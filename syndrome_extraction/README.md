# Distance-preserving syndrome-extraction circuits

Code for the search described in
[Syndrome-extraction circuits found with Sisyphus](../README.md#syndrome-extraction-circuits-found-with-sisyphus):
it generates depth-12 interleaved syndrome-extraction circuits (X and Z checks measured in the same
layers) for block-structured qLDPC codes and rejects every circuit for which Sisyphus finds a set of
fewer than d faults that flips a logical observable without triggering a detector.

## Requirements

`sisyphus_qec` (this repository), `numpy`, `stim`, and `python-sat` for the SAT-based search.
Mitten codes are read from the public repository [github.com/a7b/yarn](https://github.com/a7b/yarn):
set `MITTEN_CODES` to its `processor_codes` directory. The affine-permutation and GALA codes are
built from their published parameters (`codes.py`). Search outputs and logs go to `$SE_RUNS`
(default `./se_runs`).

## Files

| file | contents |
|---|---|
| `codes.py` | codes, their blocks of gates (one check block meets one data block through one permutation), the validity constraints of a circuit |
| `circuits.py` | memory-experiment circuits (sequential, interleaved, pipelined), uniform circuit noise, the reduced detector error model |
| `search.py` | circuit searches: simulated annealing, and a SAT search that learns which gate orders to avoid |
| `filter.py` | the staged round filter (Sisyphus) and witness verification |
| `schedules/circuits.json` | the best circuits found for five codes |

## Usage

Run from the repository root, inside a job on a cluster (the filter uses all `workers` cores):

```bash
export MITTEN_CODES=/path/to/yarn/processor_codes
python -m syndrome_extraction info apm96                     # code, gate blocks, constraints
python -m syndrome_extraction anneal apm96 100 1             # 100 valid depth-12 circuits
python -m syndrome_extraction filter apm96 0-100 180 16      # 180 s per circuit on 16 cores
python -m syndrome_extraction verify apm96                   # re-check every rejection
python -m syndrome_extraction sat apm96 12 6 16 --floor 11   # SAT search, 6 hours
python -m syndrome_extraction check                          # validity of schedules/circuits.json
python -m syndrome_extraction build gala2232:z6 best 16      # .stim memory circuits of a best circuit
python -m syndrome_extraction build apm96 ref 1              # .stim memory circuits, coloration circuit
```

`<code>` is one of `m150 ... m975`, `s300 ... s840` (Mitten), `apm96`, `apm192`, `apm384`,
`gala672`, `gala1752`, `gala1764`, `gala2232`. A suffix `:z<k>` selects pipelined circuits, whose
Z checks start k layers after the X checks (`gala2232:z6`).

## Circuits

A circuit is a list `T` with one layer index per gate block, in the order of `code.moves`. The
validity conditions (deterministic detectors when X and Z checks share layers) are linear over
GF(2) in the order bits [T_s < T_t] of the X/Z block pairs that meet on a data block; `codes.py`
derives them exactly from the gate lists, and every circuit is also checked with Stim.

The filter searches the r-round memory circuit for r = 1, 2, 3, 4, 6, 8, ..., d − 1. A light fault
set of a short circuit is one of every longer circuit, and a minimum-weight logical fault set of
weight w spans at most w rounds, so the (d − 1)-round circuit contains every fault set of weight at
most d − 1 of the d-round circuit. Most rejections happen at r = 1 or 2, within seconds. A rejection
is a proof (the fault set is checked against the directly built d-round model); a surviving circuit
is a candidate, since Sisyphus gives upper bounds only.

`schedules/circuits.json` holds, for each circuit, `name`, `code` (with the pipelining suffix),
`n`, `k`, `d`, `coloration_one_round` (lightest fault set of one round of the coloration circuit),
`depth`, `evidence` (the searches that found no fault set lighter than d) and `T`.
