"""
map_matching.py
=================

STEP 5: proper HMM-based map-matching, replacing the old codebase's
naive "snap every point to its single nearest OSM node/edge" approach.

WHY THE OLD APPROACH WAS WEAK
-------------------------------
Nearest-edge snapping looks at each GPS/DR point in isolation. At a
junction, alongside a parallel service road, or on a multi-level road
(exactly the tunnel/flyover scenarios this PS cares about), the single
nearest edge is very often the WRONG edge, and there's nothing in the
algorithm to notice or correct that.

THE FIX: Hidden Markov Model map matching (Newson & Krumm, 2009)
--------------------------------------------------------------------
This is the same class of algorithm used by real map-matchers (e.g.
Valhalla's Meili, GraphHopper's map-matching). Instead of picking the
nearest edge per point independently, it finds the most likely ENTIRE
PATH through the road network, using two probabilities:

  - EMISSION probability: how close is a candidate road point to the
    raw (noisy) GPS/DR point? (Gaussian in distance)
  - TRANSITION probability: is the shortest path ON THE ROAD NETWORK
    between two consecutive candidates roughly the same length as the
    raw straight-line distance between the two raw points? If the only
    way to connect two candidates requires a huge detour, that's a sign
    they're probably not both correct.

Viterbi decoding then finds the single best sequence of candidates
across the whole trajectory, not just per-point.

This is more compute than nearest-edge snapping, but still fast enough
for a hackathon demo (candidate sets are small, we downsample the
trajectory first).
"""

import numpy as np
import networkx as nx
from scipy.spatial import cKDTree

from osm_graph import haversine_m


def _project_point_to_segment(px, py, ax, ay, bx, by):
    """Project point P onto segment AB (all in local flat x,y meters).
    Returns (proj_x, proj_y, fraction_along_segment, distance)."""
    abx, aby = bx - ax, by - ay
    seg_len_sq = abx ** 2 + aby ** 2
    if seg_len_sq < 1e-9:
        return ax, ay, 0.0, np.hypot(px - ax, py - ay)

    t = ((px - ax) * abx + (py - ay) * aby) / seg_len_sq
    t = np.clip(t, 0.0, 1.0)
    proj_x, proj_y = ax + t * abx, ay + t * aby
    dist = np.hypot(px - proj_x, py - proj_y)
    return proj_x, proj_y, t, dist


class MapMatcher:
    def __init__(self, graph, sigma_z=10.0, beta=10.0, search_radius_m=50.0):
        """
        graph          : networkx graph from osm_graph.load_osm_graph()
        sigma_z        : GPS/DR position noise std-dev (meters) -- controls
                          how strongly emission probability penalizes distance
        beta           : transition probability decay constant (meters) --
                          controls how strongly route-length mismatches are
                          penalized
        search_radius_m: how far around each point to look for candidate
                          road segments
        """
        self.G = graph
        self.sigma_z = sigma_z
        self.beta = beta
        self.search_radius_m = search_radius_m

        # local flat-earth projection origin = graph centroid (fine for
        # areas up to a few tens of km)
        lats = [d["lat"] for _, d in graph.nodes(data=True)]
        lons = [d["lon"] for _, d in graph.nodes(data=True)]
        self.origin_lat = float(np.mean(lats))
        self.origin_lon = float(np.mean(lons))

        node_xy = {}
        for nid, d in graph.nodes(data=True):
            x, y = self._latlon_to_xy(d["lat"], d["lon"])
            node_xy[nid] = (x, y)
        self.node_xy = node_xy

        # spatial index for fast "which nodes are near (x,y)" queries --
        # brute-force over every node would be too slow once we're
        # calling this thousands of times (route selection, Viterbi, etc.)
        self._node_ids = list(node_xy.keys())
        self._node_xy_arr = np.array([node_xy[nid] for nid in self._node_ids])
        self._kdtree = cKDTree(self._node_xy_arr)

        # precompute all-pairs isn't feasible for a big graph; we compute
        # shortest paths on demand with a cutoff for speed
        self._sp_cache = {}

    def _latlon_to_xy(self, lat, lon):
        y = np.radians(lat - self.origin_lat) * 6371000.0
        x = np.radians(lon - self.origin_lon) * 6371000.0 * np.cos(np.radians(self.origin_lat))
        return x, y

    def _xy_to_latlon(self, x, y):
        lat = self.origin_lat + np.degrees(y / 6371000.0)
        lon = self.origin_lon + np.degrees(x / (6371000.0 * np.cos(np.radians(self.origin_lat))))
        return lat, lon

    def _candidates(self, x, y):
        """Find candidate (projected point, edge, distance) near (x, y),
        using a KD-tree for fast nearby-node lookup instead of scanning
        every node in the graph."""
        candidates = []
        seen_edges = set()

        nearby_idx = self._kdtree.query_ball_point((x, y), r=self.search_radius_m * 2)

        for idx in nearby_idx:
            nid = self._node_ids[idx]
            for neighbor in self.G.neighbors(nid):
                edge_key = tuple(sorted((nid, neighbor)))
                if edge_key in seen_edges:
                    continue
                seen_edges.add(edge_key)

                ax, ay = self.node_xy[nid]
                bx, by = self.node_xy[neighbor]
                proj_x, proj_y, frac, dist = _project_point_to_segment(x, y, ax, ay, bx, by)

                if dist <= self.search_radius_m:
                    candidates.append({
                        "edge": edge_key,
                        "proj_xy": (proj_x, proj_y),
                        "frac": frac,
                        "dist": dist,
                    })

        # keep the best few candidates only (by distance)
        candidates.sort(key=lambda c: c["dist"])
        return candidates[:5]

    def _network_distance(self, cand_a, cand_b):
        """Shortest-path network distance between two candidate points,
        accounting for exactly where each candidate sits ALONG its edge
        (not just snapping to the nearest endpoint, which was the bug --
        for short edges relative to GPS noise, ignoring the partial-edge
        offset introduced enough error to make the transition model
        actively misleading)."""
        a1, a2 = cand_a["edge"]
        b1, b2 = cand_b["edge"]

        len_a = self.G[a1][a2]["length"]
        len_b = self.G[b1][b2]["length"]

        # distance from candidate_a's projected point to each of its
        # edge's two endpoints
        dist_a_to_a1 = cand_a["frac"] * len_a
        dist_a_to_a2 = (1 - cand_a["frac"]) * len_a

        dist_b_to_b1 = cand_b["frac"] * len_b
        dist_b_to_b2 = (1 - cand_b["frac"]) * len_b

        if cand_a["edge"] == cand_b["edge"]:
            # same edge: exact, no need to route through the network at all
            return abs(cand_a["frac"] - cand_b["frac"]) * len_a

        best = np.inf
        for exit_node, dist_a_exit in ((a1, dist_a_to_a1), (a2, dist_a_to_a2)):
            for entry_node, dist_b_entry in ((b1, dist_b_to_b1), (b2, dist_b_to_b2)):
                key = (exit_node, entry_node)
                if key not in self._sp_cache:
                    try:
                        self._sp_cache[key] = nx.shortest_path_length(
                            self.G, exit_node, entry_node, weight="length"
                        )
                    except nx.NetworkXNoPath:
                        self._sp_cache[key] = np.inf
                total = dist_a_exit + self._sp_cache[key] + dist_b_entry
                if total < best:
                    best = total

        return best

    def match(self, xs, ys):
        """
        xs, ys: local flat x,y meter coordinates of the trajectory
                (already downsampled -- see match_latlon for the
                lat/lon convenience wrapper which also downsamples)

        Returns: list of (matched_lat, matched_lon) per input point,
        or (None, None) for points where no candidate was found.
        """
        n = len(xs)
        candidates_per_step = [self._candidates(xs[i], ys[i]) for i in range(n)]

        # Viterbi
        V = [{}]          # V[t][candidate_index] = log-probability
        backptr = [{}]

        for ci, c in enumerate(candidates_per_step[0]):
            emission = np.exp(-0.5 * (c["dist"] / self.sigma_z) ** 2)
            V[0][ci] = np.log(max(emission, 1e-12))
        backptr[0] = {ci: None for ci in V[0]}

        for t in range(1, n):
            V.append({})
            backptr.append({})
            straight_dist = np.hypot(xs[t] - xs[t - 1], ys[t] - ys[t - 1])

            for ci, c in enumerate(candidates_per_step[t]):
                emission = np.exp(-0.5 * (c["dist"] / self.sigma_z) ** 2)
                log_emission = np.log(max(emission, 1e-12))

                best_prev_score = -np.inf
                best_prev_idx = None

                for pi, prev_c in enumerate(candidates_per_step[t - 1]):
                    if pi not in V[t - 1]:
                        continue
                    net_dist = self._network_distance(prev_c, c)
                    if not np.isfinite(net_dist):
                        transition = 1e-12
                    else:
                        diff = abs(net_dist - straight_dist)
                        transition = np.exp(-diff / self.beta)

                    score = V[t - 1][pi] + np.log(max(transition, 1e-12))
                    if score > best_prev_score:
                        best_prev_score = score
                        best_prev_idx = pi

                if best_prev_idx is not None:
                    V[t][ci] = best_prev_score + log_emission
                    backptr[t][ci] = best_prev_idx

        # backtrack from the best final state
        if not V[-1]:
            return [(None, None)] * n

        best_last = max(V[-1], key=V[-1].get)
        path_idx = [None] * n
        path_idx[-1] = best_last
        for t in range(n - 1, 0, -1):
            path_idx[t - 1] = backptr[t][path_idx[t]]

        matched_latlon = []
        for t in range(n):
            idx = path_idx[t]
            if idx is None or idx not in range(len(candidates_per_step[t])):
                matched_latlon.append((None, None))
                continue
            px, py = candidates_per_step[t][idx]["proj_xy"]
            lat, lon = self._xy_to_latlon(px, py)
            matched_latlon.append((lat, lon))

        return matched_latlon

    def match_latlon(self, lats, lons, min_step_m=8.0):
        """Convenience wrapper: takes raw lat/lon trajectory, downsamples
        it so consecutive points are at least `min_step_m` apart (keeps
        the Viterbi trellis small and avoids near-duplicate points that
        add noise without adding information), runs match(), and returns
        (matched_lat, matched_lon, kept_indices)."""

        xs_all, ys_all = [], []
        for lat, lon in zip(lats, lons):
            x, y = self._latlon_to_xy(lat, lon)
            xs_all.append(x)
            ys_all.append(y)

        kept = [0]
        for i in range(1, len(xs_all)):
            last = kept[-1]
            if np.hypot(xs_all[i] - xs_all[last], ys_all[i] - ys_all[last]) >= min_step_m:
                kept.append(i)
        if kept[-1] != len(xs_all) - 1:
            kept.append(len(xs_all) - 1)

        xs = [xs_all[i] for i in kept]
        ys = [ys_all[i] for i in kept]

        matched = self.match(xs, ys)
        matched_lat = [m[0] for m in matched]
        matched_lon = [m[1] for m in matched]
        return matched_lat, matched_lon, kept
