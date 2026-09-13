"""
osm_graph.py
==============

Parses a cached Overpass API JSON export (raw OSM data: nodes + ways)
into a networkx graph we can route and snap points onto.

We don't have network access in this environment, but the old codebase
already cached real Overpass query results (cache/*.json) from when it
had access -- that's real, valid OSM data for a real area (near
Loughborough, UK, matching the IO-VNBD dataset's coverage). This module
uses that instead of re-querying anything.

For a real deployment, replace `load_osm_graph()`'s JSON source with a
live osmnx/Overpass call once you have network access on your own
machine -- everything downstream (map_matching.py) only depends on
getting a networkx graph with 'lat'/'lon' node attributes and 'length'
edge weights, so swapping the source is a one-function change.
"""

import json
import numpy as np
import networkx as nx

EARTH_RADIUS_M = 6371000.0

DRIVABLE_HIGHWAY_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link",
}


def haversine_m(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def load_osm_graph(json_path, drivable_only=True):
    """Returns a networkx.Graph with node attrs {lat, lon} and edge
    attrs {length (meters), name}."""

    data = json.load(open(json_path))
    elements = data["elements"]

    nodes = {}
    ways = []
    for el in elements:
        if el["type"] == "node":
            nodes[el["id"]] = (el["lat"], el["lon"])
        elif el["type"] == "way":
            ways.append(el)

    G = nx.Graph()

    n_skipped = 0
    for way in ways:
        tags = way.get("tags", {})
        highway = tags.get("highway")
        if drivable_only and highway not in DRIVABLE_HIGHWAY_TYPES:
            n_skipped += 1
            continue

        node_ids = way["nodes"]
        for a, b in zip(node_ids[:-1], node_ids[1:]):
            if a not in nodes or b not in nodes:
                continue
            lat1, lon1 = nodes[a]
            lat2, lon2 = nodes[b]
            length = haversine_m(lat1, lon1, lat2, lon2)

            G.add_node(a, lat=lat1, lon=lon1)
            G.add_node(b, lat=lat2, lon=lon2)
            G.add_edge(a, b, length=float(length), name=tags.get("name", ""))

    # keep only the largest connected component -- isolated fragments
    # (e.g. footpaths that got tagged oddly) aren't useful for routing
    if G.number_of_nodes() > 0:
        largest_cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()

    print(f"[osm_graph] loaded {G.number_of_nodes()} nodes, "
          f"{G.number_of_edges()} edges "
          f"({n_skipped} non-drivable ways skipped)")

    return G


if __name__ == "__main__":
    G = load_osm_graph("osm_cache.json")
    lats = [d["lat"] for _, d in G.nodes(data=True)]
    lons = [d["lon"] for _, d in G.nodes(data=True)]
    print(f"Coverage: lat [{min(lats):.4f}, {max(lats):.4f}], "
          f"lon [{min(lons):.4f}, {max(lons):.4f}]")
