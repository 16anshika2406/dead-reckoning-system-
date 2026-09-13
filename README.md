# Step 5: HMM Map-Matching

Replaces the old codebase's naive "snap every point to its single
nearest OSM edge, independently" approach with a proper Hidden Markov
Model map matcher (Newson & Krumm, 2009 — the same class of algorithm
used by Valhalla's Meili and GraphHopper's map matching).

## Why the old approach was weak
Nearest-edge snapping looks at each point alone. At a junction or near
parallel roads — exactly where tunnels/flyovers/underpasses put you —
the single nearest edge is often the wrong one, and there's nothing to
catch that.

## How this one works
Two probabilities feed a Viterbi search over the whole trajectory at
once, not point-by-point:
- **Emission**: how close is a candidate road point to the raw GPS/DR point?
- **Transition**: is the real shortest network-path distance between two
  consecutive candidates close to the raw straight-line distance between
  the two raw points? A candidate pair that requires a big detour to
  connect is penalized.

## Data source
No live internet access in this environment, so this runs on
`osm_cache.json` — the OLD codebase's own cached Overpass API query
result (real OpenStreetMap road data near the IO-VNBD test area,
Loughborough UK). `osm_graph.py` parses it into a routable graph. On a
machine with internet access, swap in a live osmnx/Overpass call —
`map_matching.py` only needs a networkx graph with `lat`/`lon` node
attributes, so nothing else changes.

## Validation
`demo_map_matching.py`:
1. Picks a REAL route through the real road graph — specifically
   selected to include genuine junction ambiguity (multiple nearby
   roads), since that's the actual failure mode being fixed, not an
   easy, unambiguous stretch.
2. Densifies it, adds realistic 8m position noise.
3. Runs naive nearest-edge snap AND the new HMM matcher on the *same*
   downsampled point set (an earlier version of this test accidentally
   compared different point sets — fixed).
4. Scores both against the known true route.

**Result:** naive snapping got 90.8% of points onto the correct route
edge; the HMM matcher got 96.9% — see `map_matching_comparison.png` for
the visual (the naive method visibly drifts onto a wrong nearby edge in
one spot; the HMM matcher stays on the true route through the same spot).

## Known limitation, found and fixed during testing
The first version of `_network_distance()` approximated route distance
using only edge ENDPOINTS, ignoring exactly where each candidate sits
along its edge. For short edges relative to GPS noise, this was
inaccurate enough that the HMM matcher initially performed *worse* than
naive snapping. Fixed by computing the exact partial-edge offset at
both ends before routing through the network. Worth remembering if you
extend this: always sanity-check that a "smarter" algorithm is actually
outperforming the baseline it's meant to replace, on a fair (same
point-set) comparison.

## Run it
```bash
pip install networkx scipy matplotlib numpy
python3 demo_map_matching.py
```
