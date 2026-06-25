"""Wallet funding-graph features and wash-trading ring detection.

Builds a directed graph of "funded by" relationships from
`AccountActivity.funding_account` and derives the signals used by
`feature_engineering.compute_wallet_graph_features`:

Single-wallet signals
----------------------
- `funding_source_similarity`: the highest Jaccard similarity between a
  wallet's set of *multi-hop* funding ancestors (up to `max_depth` hops) and
  any other wallet's funding-ancestor set. A high value means two wallets
  trace back to the same funding source(s) — a common pattern for
  sock-puppet / wash-trading rings.
- `network_centrality`: degree centrality of the wallet within the funding
  graph.

Topological ring signals (Issue #11)
------------------------------------
Real wash-trading rings insert 2–4 intermediate accounts between the
controlling wallet and the trading wallets to evade single-hop metrics.
`detect_wash_trading_rings` applies Louvain community detection to the
undirected funding graph to surface these clusters, and `ring_statistics`
summarises each detected ring (size, internal edge density, funding depth).
"""

import hashlib
from collections import Counter, defaultdict, deque
from collections.abc import Iterable

import networkx as nx

from config import config
from ingestion.data_models import AccountActivity

try:  # python-louvain — preferred community detector
    import community as _community_louvain
except ImportError:  # pragma: no cover - fallback path exercised only without the dep
    _community_louvain = None

DEFAULT_MAX_DEPTH = config.WALLET_GRAPH_MAX_DEPTH
DEFAULT_MIN_RING_SIZE = config.WASH_RING_MIN_SIZE
DEFAULT_RESOLUTION = config.WASH_RING_RESOLUTION
DEFAULT_SEED = config.WASH_RING_LOUVAIN_SEED

NO_RING = -1


def build_funding_graph(
    activities: Iterable[AccountActivity], max_depth: int | None = None
) -> nx.DiGraph:
    """Build a directed graph with edges `funding_account -> account_id`.

    `max_depth` (optional) is recorded on the graph as the default BFS depth
    cap used by multi-hop traversal; it does not prune edges at build time
    (every observed funding edge is kept) but lets callers thread a consistent
    traversal budget through the feature pipeline.
    """
    graph: nx.DiGraph = nx.DiGraph()
    graph.graph["max_depth"] = DEFAULT_MAX_DEPTH if max_depth is None else max_depth
    for activity in activities:
        graph.add_node(activity.account_id)
        if activity.funding_account:
            graph.add_edge(activity.funding_account, activity.account_id)
    return graph


def multi_hop_ancestors(graph: nx.DiGraph, wallet: str, max_depth: int) -> set[str]:
    """Return funding ancestors of `wallet` reachable within `max_depth` hops.

    Performs a bounded breadth-first traversal over predecessor edges. A
    `visited` set guarantees each node is expanded once, so the traversal is
    linear in the size of the reachable subgraph and safe on cyclic graphs.
    """
    if wallet not in graph or max_depth < 1:
        return set()

    visited: set[str] = set()
    frontier = {wallet}
    for _ in range(max_depth):
        next_frontier: set[str] = set()
        for node in frontier:
            for pred in graph.predecessors(node):
                if pred != wallet and pred not in visited:
                    visited.add(pred)
                    next_frontier.add(pred)
        if not next_frontier:
            break
        frontier = next_frontier
    return visited


def funding_source_similarity(
    wallet: str, graph: nx.DiGraph, max_depth: int = DEFAULT_MAX_DEPTH
) -> float:
    """Highest Jaccard similarity between `wallet`'s multi-hop funding ancestors
    and any other node's funding ancestors in `graph`.

    Returns `0.0` if `wallet` isn't in the graph or has no funding ancestors.
    """
    if wallet not in graph:
        return 0.0

    wallet_ancestors = multi_hop_ancestors(graph, wallet, max_depth)
    if not wallet_ancestors:
        return 0.0

    best = 0.0
    for other in graph.nodes:
        if other == wallet:
            continue
        other_ancestors = multi_hop_ancestors(graph, other, max_depth)
        if not other_ancestors:
            continue
        union = wallet_ancestors | other_ancestors
        if not union:
            continue
        jaccard = len(wallet_ancestors & other_ancestors) / len(union)
        best = max(best, jaccard)

    return float(best)


def network_centrality(wallet: str, graph: nx.DiGraph) -> float:
    """Degree centrality of `wallet` within the funding graph."""
    if wallet not in graph or graph.number_of_nodes() < 2:
        return 0.0
    return float(nx.degree_centrality(graph)[wallet])


def compute_wallet_graph_metrics(
    wallet: str, graph: nx.DiGraph, max_depth: int = DEFAULT_MAX_DEPTH
) -> dict:
    """Return `{funding_source_similarity, network_centrality}` for `wallet`."""
    return {
        "funding_source_similarity": funding_source_similarity(wallet, graph, max_depth),
        "network_centrality": network_centrality(wallet, graph),
    }


# ---------------------------------------------------------------------------
# Wash-trading ring detection (community detection)
# ---------------------------------------------------------------------------


def detect_wash_trading_rings(
    graph: nx.DiGraph,
    resolution: float = DEFAULT_RESOLUTION,
    min_ring_size: int = DEFAULT_MIN_RING_SIZE,
    seed: int = DEFAULT_SEED,
) -> dict[str, int]:
    """Partition `graph` into communities and label likely wash-trading rings.

    The directed funding graph is converted to an undirected graph and passed
    to the Louvain algorithm (python-louvain) with a fixed `seed` so results
    are deterministic for CI. Communities with fewer than `min_ring_size`
    members are not considered rings; their wallets are assigned `NO_RING`
    (`-1`). Returns a mapping `wallet_id -> community_id`.
    """
    if graph.number_of_nodes() == 0:
        return {}

    undirected = nx.Graph(graph.to_undirected())

    if _community_louvain is not None:
        partition = _community_louvain.best_partition(
            undirected, resolution=resolution, random_state=seed
        )
    else:  # pragma: no cover - fallback only when python-louvain is absent
        communities = nx.community.greedy_modularity_communities(undirected, resolution=resolution)
        partition = {node: cid for cid, members in enumerate(communities) for node in members}

    sizes = Counter(partition.values())
    return {
        node: (cid if sizes[cid] >= min_ring_size else NO_RING) for node, cid in partition.items()
    }


def ring_id_for_members(members: Iterable[str]) -> str:
    """Stable string ring id derived from the sorted member set."""
    digest = hashlib.sha1(",".join(sorted(members)).encode("utf-8")).hexdigest()
    return f"ring_{digest[:12]}"


def ring_statistics(community_id: int, community_map: dict[str, int], graph: nx.DiGraph) -> dict:
    """Summarise a detected community.

    Returns `{ring_id, ring_size, internal_edge_density, avg_funding_depth}`.
    `internal_edge_density` is the fraction of possible undirected edges within
    the community that actually exist (1.0 = fully connected clique).
    """
    members = sorted(node for node, cid in community_map.items() if cid == community_id)
    ring_size = len(members)

    undirected = graph.to_undirected()
    internal_edges = undirected.subgraph(members).number_of_edges()
    possible_edges = ring_size * (ring_size - 1) // 2
    density = float(internal_edges / possible_edges) if possible_edges > 0 else 0.0

    return {
        "ring_id": ring_id_for_members(members),
        "ring_size": ring_size,
        "internal_edge_density": density,
        "avg_funding_depth": _avg_funding_depth(graph, members),
    }


def build_ring_statistics(community_map: dict[str, int], graph: nx.DiGraph) -> dict[int, dict]:
    """Compute `ring_statistics` for every non-`NO_RING` community in the map."""
    stats: dict[int, dict] = {}
    for community_id in set(community_map.values()):
        if community_id == NO_RING:
            continue
        stats[community_id] = ring_statistics(community_id, community_map, graph)
    return stats


def ring_id_map(community_map: dict[str, int], graph: nx.DiGraph) -> dict[str, str | None]:
    """Map each wallet to its stable `ring_id` string (`None` if not in a ring)."""
    by_community: dict[int, list[str]] = defaultdict(list)
    for node, community_id in community_map.items():
        by_community[community_id].append(node)

    result: dict[str, str | None] = {}
    for community_id, members in by_community.items():
        rid = None if community_id == NO_RING else ring_id_for_members(members)
        for node in members:
            result[node] = rid
    return result


def _avg_funding_depth(graph: nx.DiGraph, members: list[str]) -> float:
    """Average funding depth of `members` within their induced directed subgraph.

    Depth is the number of funding hops from the nearest in-ring funding source
    (a member with no in-ring funder). Returns 0.0 when the subgraph has no such
    source (e.g. a pure cycle).
    """
    if not members:
        return 0.0

    sub = graph.subgraph(members)
    sources = [node for node in sub.nodes if sub.in_degree(node) == 0]
    if not sources:
        return 0.0

    distance = {source: 0 for source in sources}
    queue = deque(sources)
    while queue:
        node = queue.popleft()
        for successor in sub.successors(node):
            if successor not in distance:
                distance[successor] = distance[node] + 1
                queue.append(successor)

    depths = [distance.get(node, 0) for node in sub.nodes]
    return float(sum(depths) / len(depths)) if depths else 0.0
