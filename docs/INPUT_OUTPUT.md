# Input and output contract

## Query JSON

The public API accepts a layered query object. The minimal fields are:

- `behavior_graph.nodes`: node objects with a stable `id`, one of `Process`, `File`, `Socket`, or `Registry`, and optional attributes;
- `behavior_graph.edges`: directed edges with `id`, `source`, `target`, and a canonical relation;
- `attack_instances`: instance identifiers and their `behavior_edge_ids`;
- `instance_to_stage`: optional mapping to the seven-stage vocabulary.

Canonical relations are `connect`, `exec`, `exit`, `fork`, `mmap`, `read`, `recv`, `send`, `unlink`, and `write`.

## Candidate JSON

The candidate provenance subgraph contains:

- `nodes`: node objects; numeric DARPA-style types `0/1/2/3` and canonical type names are accepted;
- `edges`: directed provenance edges with resolvable endpoints, relation/syscall, and optional time;
- `paths`: retained path objects represented by ordered `edge_ids`.

Paths are not rediscovered from the entire candidate graph by this Stage-3 package. They must be supplied by the Stage-2 sampler or another audited bounded path generator.

## Output

The API returns the final score, entity score, behavior score, weighted NULL rate, and an explanation object containing:

- per-query-node candidate probabilities and NULL probability;
- structural weight assigned to each query node;
- selected path evidence for each attack instance;
- weak stage-order conflicts and penalty;
- feature and configuration hashes.

The CLI marks output as `scientific_inference: true` only when a trained checkpoint was loaded.
