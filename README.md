# Sisyphus — fast distance search for quantum codes and circuits

Sisyphus finds low-weight logical operators of CSS quantum codes and low-weight
undetectable fault sets of circuit-level detector error models (DEMs). It reports
the lightest ones it finds, so it gives **upper bounds on the code distance and
on the circuit-level distance**, together with an explicit, verified witness for
every bound.

Sisyphus is a **hill-climbing variant of information-set decoding (ISD)**
[[1](#ref-1)–[4](#ref-4), [11](#ref-11)]. It combines several algorithmic
improvements of Prange's algorithm [[1](#ref-1)] (sums of a few rows [[2](#ref-2)], a collision
search for such sums [[3](#ref-3)], and moving between neighbouring information sets by
single column swaps [[4](#ref-4)]) and adds **adaptive learning of the information sets**:
column weights learned from the low-weight codewords already found decide which
columns enter and leave the information set, so each search climbs toward lighter
codewords. When a climb stalls, it partially forgets what it learned and restarts,
as in iterated local search [[12](#ref-12)], and many independent climbs run in parallel
[[13](#ref-13)]. It is written in Python with
[Numba](https://numba.pydata.org/)-compiled bit-packed GF(2) kernels.

Sisyphus builds directly on two existing tools for computing the distance of quantum
codes, and we gratefully acknowledge both:
[**QDistRnd**](https://github.com/QEC-pages/QDistRnd) [[5](#ref-5)], the GAP package that established randomized information-set search as a
practical method for quantum codes, and
[**dist-m4ri**](https://github.com/QEC-pages/dist-m4ri) [[6](#ref-6)], its
fast C successor for codes and circuit-level detector error models, whose
benchmark set and exact detector-error-model reduction we use for all
circuit-level comparisons. A recent survey and benchmark of distance-finding
algorithms for quantum codes and circuits [[7](#ref-7)] was a further guide.

> **If you use Sisyphus in published work, please cite Sisyphus together with the
> QDistRnd paper and the distance-finding paper of Webster, Jacob and Higgott**
> (BibTeX in [Citing](#citing)).

Highlights (details in [Benchmarks](#benchmarks-and-comparisons)):

- On large circuit-level instances Sisyphus improves several **best published bounds**
  on circuit-level distance. For example, it finds a weight-33 undetectable fault
  set in the 34-round syndrome-extraction circuit of the [[756,16,≤34]]
  bivariate bicycle code, which is below the code's own distance bound of 34. It
  also gives 21 for the [[360,12,≤24]] circuit (published: ≤ 23), 16 for the
  double-gross [[288,12,18]] circuit (published: ≤ 18) and 10 for the gross
  [[144,12,12]] circuit with uniform noise (published: ≤ 11).
- On medium-size circuit instances it reaches the published distance **3–51×
  faster** than [dist-m4ri](https://github.com/QEC-pages/dist-m4ri) at the median
  (same job, 16 cores each), and ties on the remaining one.
- On codes it matches or beats dist-m4ri: sub-second on everything up to
  [[2304,1156]], and 2–5× faster on one side of the hardest instance tested
  ([[4608,2308,≤20]]).

---

## Contents

1. [Installation](#installation)
2. [Quick start](#quick-start)
3. [What Sisyphus computes](#what-sisyphus-computes)
4. [How the method works](#how-the-method-works)
5. [How Sisyphus improves on other randomized ISD methods](#how-sisyphus-improves-on-other-randomized-isd-methods)
6. [API](#api)
7. [Options and default settings](#options-and-default-settings)
8. [Benchmarks and comparisons](#benchmarks-and-comparisons)
9. [Limitations](#limitations)
10. [Repository layout](#repository-layout)
11. [Citing](#citing)
12. [References](#references)
13. [License](#license)

---

## Installation

```bash
git clone https://github.com/varunmenon1310/sisyphus.git
cd Sisyphus
pip install -e .            # numpy, numba, scipy
pip install -e ".[circuits]"  # adds stim, for circuit-level instances
```

Python ≥ 3.10. The first call compiles the kernels with Numba (about 30 s). The
compiled code is cached next to the package, so later runs start in about a
second.

## Quick start

### Distance of a CSS code

```python
from sisyphus_qec import css_distance

# Hx, Hz: binary parity-check matrices (numpy arrays, Hx @ Hz.T = 0 mod 2)
res = css_distance(Hx, Hz, num_trials=200_000, num_workers=8)

res.distance          # min over both sides: an upper bound on the code distance
res.x.distance        # X side (X-type logicals)
res.z.witness         # one Z logical of weight res.z.distance (0/1 vector)
res.z.codewords       # all distinct minimum-weight Z logicals found, one per row
```

Stop at the first logical of a known or target weight with `cutoff`:

```python
res = css_distance(Hx, Hz, num_trials=10_000_000, num_workers=16, cutoff=12)
```

### Circuit-level distance from a Stim circuit

```python
import stim
from sisyphus_qec import dem_to_matrices, side_distance, lift_columns, verify_fault_set

circuit = stim.Circuit.from_file("memory_experiment.stim")
dem = circuit.detector_error_model(decompose_errors=False)
H, L, columns = dem_to_matrices(dem)          # detectors x faults, observables x faults

res = side_distance(H, None, pair_rows=L, num_trials=10_000_000,
                    num_workers=16, time_limit=600)
faults = lift_columns(res.witness, columns)   # indices of errors in dem.flattened()
assert verify_fault_set(dem, faults)          # independent check on the DEM
print("circuit distance <=", res.distance)
```

Complete scripts are in [`examples/`](examples/): `code_distance.py` (gross code,
distance 12) and `circuit_distance.py` (surface-code memory circuit, distance 5).

---

## What Sisyphus computes

**Code distance.** For a CSS code with parity-check matrices `Hx` and `Hz`, an
X-type logical operator is a binary vector `c` with `Hz c = 0` that is not a sum
of rows of `Hx` (it commutes with every Z check but is not an X stabilizer). Its
weight is the number of qubits it acts on. The X distance is the minimum weight
of such a vector. The Z distance swaps the roles of `Hx` and `Hz`. The code
distance is the smaller of the two.

**Circuit-level distance.** A detector error model [[8](#ref-8)] lists the possible faults
of a noisy circuit and, for each one, the detectors and logical observables it
flips. With one column per fault, write `H` for the detector matrix and `L` for
the observable matrix. The circuit-level distance is the smallest number of
faults that together flip no detector and at least one logical observable: the
minimum weight of `c` with `H c = 0` and `L c ≠ 0`. It is usually smaller than
the code distance, because some single faults (for example "hook" errors on
syndrome-extraction ancillas) spread to several data qubits.

Both are the same problem: **find the minimum weight of a vector in the
kernel of a binary matrix that is not in a given subspace of trivial vectors**.
This problem is NP-hard in general, so practical tools search for light
solutions. Sisyphus never certifies that nothing lighter exists. Every reported
weight is an **upper bound**, and it comes with a witness vector that you can
check independently.

---

## How the method works

This section first recalls the standard ISD algorithms that Sisyphus builds on, then
describes what Sisyphus adds. No prior knowledge of ISD is assumed.

### Background 1: information sets

Let `G` be a generator matrix of the code to be searched (for X logicals, a basis
of the kernel of `Hz`), with `k` rows and `n` columns. Gaussian elimination brings
`G` into **systematic form**: after a suitable choice of `k` columns, called the
**information set**, those columns form an identity matrix. Each row of the
systematic form then has a single 1 inside the information set and arbitrary
entries on the remaining `n − k` **redundant columns**.

The key observation [[1](#ref-1)]: if a low-weight codeword happens to have
**no more than one** nonzero entry inside the information set, it is equal to a
single row of the systematic form, and it can simply be read off. The basic
algorithm repeats: pick a random information set, eliminate, look at the rows,
keep the lightest. A codeword of weight `w` is found once an information set
avoids almost all of its support, which for small `w` happens with a probability
that can be estimated in closed form.

### Background 2: sums of rows and collision search

A first refinement [[2](#ref-2)] also checks sums of `p = 2` or `3` rows, so it also finds
codewords with 2 or 3 entries in the information set. This raises the success
probability per elimination considerably, at the cost of more work per step.

A second refinement [[3](#ref-3)] makes the search for sums of rows efficient. Rows are sorted
into buckets by their values on a small random set of `ℓ` redundant columns. Two
rows in the same bucket sum to zero on those `ℓ` columns, which makes their sum
much more likely to be light. Only such pairs (a **collision search**) are
examined, instead of all pairs.

### Background 3: moving between information sets

A full Gaussian elimination per step costs `O(k n²)` bit operations.
It was later observed [[4](#ref-4)] that one can instead move to a
neighbouring information set by **swapping a single column**: one column leaves
the information set and one redundant column enters it. Updating the systematic
form costs one row operation per affected row. The search then becomes a
**random walk (Markov chain) over information sets**, with a much cheaper step.
It is the basis of most modern ISD implementations.

### What Sisyphus adds

Sisyphus runs many such random walks at once, and makes each walk **learn where the
light codewords are**.

#### 1. Learned column weights: a stochastic hill climb

Each search chain keeps a nonnegative **weight for every column** (initially
zero). Whenever the chain sees a codeword of low weight (close to the best weight
it has found so far), it increases the weights of all columns in that codeword's
support, by more for lighter codewords. All weights also decay slowly, so old
information fades.

The weights steer the walk:

- **Which column leaves the information set.** A few candidate pivot columns
  are drawn at random and the one with the **highest** weight leaves
  (tournament selection [[14](#ref-14)]).
- **Which column enters.** Among the redundant columns that can replace it,
  the one with the **lowest** weight enters.
- **Restarts.** Periodically the chain draws a completely new information set
  by Gaussian elimination. The column order is sorted by weight, with random
  noise added, so high-weight columns end up redundant.

Columns that appear in light codewords are therefore kept **out of** the
information set, which is exactly the situation in which the extraction steps of
Background 1 and 2 can read those codewords off. Each new light codeword reinforces its
own region, so the walk climbs toward ever lighter codewords: a **stochastic hill
climb**, guided by a memory of where light codewords were found. With some
probability (8–25% depending on the chain) a move is made uniformly at random
instead, which keeps the walk from becoming deterministic.

#### 2. Escaping local optima

A pure hill climb gets stuck. The learned weights tend to concentrate on the
support of the best codeword found so far, and the chain then keeps rediscovering
that one codeword. Sisyphus handles this the way iterated local search [[12](#ref-12)] does.
When a chain's best weight has not improved for a long time (both a fixed number
of iterations and a multiple of the longest gap between its previous
improvements), the chain **forgets part of its learned weights** on its best
codeword's support (30% of those columns by default) and restarts from a fresh
information set. It keeps its general sense of the region and leaves the exact
local optimum.

In addition, each group of chains contains one chain that does not learn at all
and walks uniformly at random. It keeps the population diverse and helps
enumerate many distinct minimum-weight codewords on small codes.

#### 3. Greedy weight reduction of every candidate

Every candidate codeword found by the read-off is improved by **local search**.
Sisyphus repeatedly adds any row of the current basis that lowers the weight while
keeping the vector a nontrivial solution, until no row helps. This often
turns a near-miss into a lighter codeword.

#### 4. Parallel search: the island model

Chains are organized as in the island model of parallel evolutionary
algorithms [[13](#ref-13)]. Each process (an *island*) runs four chains in lockstep: three
learning chains with different learning rates and the uniform chain. Islands are
independent and do not exchange information. Each island runs for a fixed budget
of iterations (by default at most 250,000 per island) and is then replaced by a
freshly seeded island, until the total budget `num_trials` is spent. On hard
instances the outcome varies much more between islands than within one, so many
moderate islands beat a few long ones. When any island reaches the `cutoff`
weight, all islands stop within one iteration.

#### 5. Circuit-level instances: detector time windows

In a detector error model, detectors are ordered in time (round by round), and
the light undetectable fault sets are local in time. A low-weight fault set
typically spans only a few rounds of a long memory experiment. For such
instances Sisyphus runs **half of its islands on time windows**. A window is a range
of consecutive detectors, and its sub-problem contains only the faults whose
detectors all lie inside the window. Windows come in five sizes (halves, quarters, ... down to 1/32
of the detectors, each size with 50% overlap between neighbouring windows), and
each window island moves to a new random window periodically. A solution of a window sub-problem, padded with zeros, is a solution of
the full problem, and it is verified as such.

A window is 10–50× smaller than the full model, so its islands run 10–50×
more iterations per second and concentrate on the right region immediately.
This is what lets Sisyphus solve large circuit instances such as the 34-round
[[756,16,34]] circuit (13,230 detectors × 129,276 faults after reduction). By
default (`windows="auto"`) windows are used only when the matrix has at least
6× more columns than rows, which is the shape of circuit-level models and not of
codes.

#### 6. Efficient linear algebra

- **Bit-packed GF(2) arithmetic.** All matrices are stored as 64-bit words, so
  one XOR processes 64 entries. Gaussian elimination and the batched column
  swaps use the Method of Four Russians [[9](#ref-9), [10](#ref-10)].
- **Compact systematic form.** A systematic generator matrix is `[I | P]` up to
  column order, so only the `P` part (the redundant columns) is stored. Per-step
  work and memory scale with the number of redundant columns `n − k` (the rank
  of the check matrix) instead of `n`. That is 4× less on rate-1/2 quantum codes
  and about 10× less on circuit-level models, where `n` is 8–13 times the number
  of detectors.
- **Restarts from the check matrix.** A fresh information set is computed by
  eliminating the parity-check matrix `H` in the chosen column order. The
  systematic generator matrix follows directly, without ever computing a basis
  of the kernel. On circuit-level models this is 16–35× cheaper than reducing a
  stored generator matrix.

#### 7. Verification of every reported codeword

A vector in the kernel can still be a stabilizer (a trivial solution). Sisyphus
checks nontriviality with a **randomized test**. It draws 64 random vectors from
the space that detects nontrivial solutions (for codes, random elements of the
kernel of the opposite check matrix; for circuits, random combinations of the
observable rows). Each candidate carries its 64 parities against them, updated
alongside the rows at the cost of one extra 64-bit word per row. A nonzero parity **proves** that the candidate is
nontrivial, so Sisyphus never reports a stabilizer. A genuine logical is missed only
with probability 2⁻⁶⁴.

---

## How Sisyphus improves on other randomized ISD methods

| | basic ISD [[1](#ref-1)] | sums of rows [[2](#ref-2)] | collision search [[3](#ref-3)] | column-swap walk [[4](#ref-4)] | dist-m4ri [[6](#ref-6)] | **Sisyphus** |
|---|---|---|---|---|---|---|
| Next information set | random, full elimination | random, full elimination | random, full elimination | one column swap | random (uniform or a random window of columns), full elimination | one column swap, plus periodic restarts |
| Candidates per step | rows | rows + sums of `p` rows | rows + colliding pairs | rows + colliding pairs | rows + pairs | rows + colliding pairs + colliding triples |
| Choice of columns | uniform | uniform | uniform | uniform | uniform or window | **learned from light codewords found so far** |
| Local optima | — | — | — | — | — | **partial forgetting + restart (iterated local search)** |
| Candidate improvement | — | — | — | — | — | **greedy weight reduction** |
| Circuit-level models | — | — | — | — | column windows | **detector time windows** |
| Parallelism | — | — | — | — | threads | independent islands of chains |

In words:

1. **Directed rather than blind sampling.** Classical ISD samples information
   sets uniformly. Its cost per success is fixed by the code's weight
   distribution and does not improve as the search proceeds. Sisyphus's walks
   accumulate knowledge: each light codeword makes nearby light codewords easier
   to find. On codes whose minimum-weight logicals cluster on structured
   supports (bivariate bicycle, generalized bicycle, affine-permutation codes),
   this is why Sisyphus finds the weight-20 Z logicals of [[4608,2308,≤20]] 2–5×
   faster than dist-m4ri's random sampling.
2. **Cheap steps.** Single column swaps [[4](#ref-4)] and restarts from the
   check matrix keep each step cheap. The compact `[I | P]` storage cuts memory
   and time by `n / (n − k)`.
3. **Robustness of a population.** Several learning rates, one uniform chain,
   iterated-local-search perturbations and many independent islands keep the
   search from collapsing onto one local optimum, which is the typical failure of
   a single hill climb.
4. **Locality for circuits.** Detector time windows turn one huge problem into
   many small ones that contain the light fault sets. This exploits the
   space-time structure of circuit noise, which uniform sampling ignores.

---

## API

```python
from sisyphus_qec import css_distance, side_distance
from sisyphus_qec import dem_to_matrices, lift_columns, verify_fault_set
```

### `css_distance(Hx, Hz, num_trials=20000, num_workers=1, seed=None, cutoff=0, basis="both", ...)`

Distance of the CSS code with X checks `Hx` and Z checks `Hz`. Searches Z
logicals (kernel of `Hx` outside the row space of `Hz`) first, then X logicals.
`basis="z"` or `"x"` searches one side only. With `cutoff > 0` the X side is
skipped when the Z side already reached the cutoff. `num_trials` applies per
side. Returns a `CssDistanceResult` with fields `x` and `z` (each a
`SideResult`) and `distance` (the minimum over the sides searched).

### `side_distance(check_H, dual_H, num_trials=20000, num_workers=1, seed=None, cutoff=0, pair_rows=None, ...)`

Lightest vector `c` with `check_H c = 0` outside the row space of `dual_H`. For
X logicals of a CSS code call `side_distance(Hz, Hx)`, for Z logicals
`side_distance(Hx, Hz)`. For a detector error model call
`side_distance(H, None, pair_rows=L)`: `pair_rows` is any set of vectors such
that `c` is nontrivial exactly when it has odd overlap with one of them (for a
DEM, the observable matrix).

### `SideResult`

| field | meaning |
|---|---|
| `distance` | weight of the lightest verified solution found (an upper bound); `-1` if the side was skipped |
| `witness` | one solution of that weight (uint8 vector) |
| `codewords` | all distinct solutions of that weight found (one per row; capped by `max_codewords` per island) |
| `counts` | how many times each of them was found |
| `num_codewords` | number of rows of `codewords` |
| `shots` | iterations performed |
| `wall_time` | seconds |
| `islands` | number of islands run |
| `t_best` | estimated time (s) at which `distance` was first reached |

### DEM helpers (`sisyphus_qec.dem`)

- `dem_to_matrices(dem, merge_duplicates=True)` → `(H, L, columns)`. One
  column per fault of a `stim.DetectorErrorModel`. Faults with identical effects
  are merged; `columns[j]` lists the DEM error indices behind column `j`.
- `lift_columns(codeword, columns)` maps a solution over the columns to DEM
  error indices (positions among the error instructions of `dem.flattened()`).
- `verify_fault_set(dem, error_indices)` is an independent check on the DEM:
  the faults are distinct, flip no detector, and flip at least one observable.

Matrices are passed as dense 0/1 numpy arrays and bit-packed internally. A
reduced circuit model of 13,230 × 129,276 needs about 1.7 GB as a dense
`uint8` array.

---

## Options and default settings

> **The defaults work well across codes of very different sizes and across
> circuit-level instances, and we do not recommend changing them.** They were
> chosen by equal-wall-clock comparisons on small, medium and large codes and on
> circuit-level models, and every default is the best setting found in those
> comparisons. The search parameters interact in ways that are hard to tune by
> hand. If you need to tune them for a specific family of instances, use an
> automated hyperparameter search (for example an agentic optimization loop that
> scores settings by time-to-target over several seeds), not one-off manual
> changes.

The options you normally set are the **run options**. The **search parameters**
can be passed as keyword arguments to `css_distance` and `side_distance`; they
are listed for completeness.

### Run options

| option | default | meaning |
|---|---|---|
| `num_trials` | `20000` | total number of iterations over all islands (per side for `css_distance`). One iteration = one move of one chain plus candidate extraction. |
| `num_workers` | `1` | islands run in parallel (processes). `None` = all available CPU cores. `1` runs in the calling process. |
| `cutoff` | `0` | stop as soon as a solution of weight `≤ cutoff` is found. `0` runs the whole budget, which also collects the set of minimum-weight solutions. |
| `time_limit` | `None` | wall-clock budget in seconds (`side_distance`). The search returns the best result so far when it expires. Needs `num_workers ≥ 2`. |
| `seed` | `None` | random seed (`None` = fresh randomness). |
| `island_shots` | `None` | iteration budget of one island before it is replaced by a fresh one. `None` = an equal share of `num_trials` per worker, capped at 250,000. |
| `walkers_per_island` | `4` | chains per island: three learning chains with different learning rates and one uniform chain. Larger values reuse these presets in turn. |
| `num_seeds` | `None` | older alternative to `island_shots`: split the budget into this many waves of `num_workers` islands. |
| `pair_rows` | `None` | vectors defining nontriviality (see `side_distance`). Required for detector error models (`L`). |
| `collect_codewords` | `True` | keep the set of minimum-weight solutions (otherwise one). |
| `basis` | `"both"` | `css_distance` only: `"both"`, `"z"` or `"x"`. |
| `compact` | `None` | `None` / `True`: compact `[I | P]` storage. `False`: full-width rows (same search, slower). |

### Search parameters

| parameter | default | meaning |
|---|---|---|
| `swaps_per_shot` | `8` | column swaps per iteration, applied as one batch. |
| `restart_interval` | `None` | iterations between fresh information sets. `None` = `min(2000, max(150, k/3))` with `k` the dimension of the searched space. |
| `lb_p` | `2` | candidate extraction: `0` rows only, `1` + pairs, `2` + triples. |
| `lb_hash_l` | `12` | `ℓ`, the number of random redundant columns used to bucket rows in the collision search [[3](#ref-3)]. `0` = exhaustive pairs and triples among the lightest rows instead. |
| `lb_M` | `256` | size of the list of lightest rows used for triples. Use `48` with `lb_hash_l=0`. |
| `lb_R` | `48` | rows used as the first element of a triple. |
| `lb_hash_windows` | `1` | independent random column sets for bucketing per iteration. |
| `greedy_passes` | `4` | sweeps of greedy weight reduction per candidate (`0` = off). |
| `kick_rounds` | `1000` | a learning chain whose best weight has not improved for this many iterations partially forgets and restarts (`0` = off). |
| `kick_frac` | `0.3` | fraction of the best codeword's columns whose learned weight is reset at such a restart. |
| `kick_patience` | `2.0` | also wait at least this multiple of the chain's longest gap between improvements so far before restarting. |
| `restart_from_h` | `1` | compute fresh information sets by eliminating the check matrix (cheaper; identical result). |
| `windows` | `"auto"` | detector time windows: `"auto"` = 5 levels when columns ≥ 6× rows (circuit models), otherwise off. An integer sets the number of levels; `0` = off. |
| `window_frac` | `0.5` | fraction of islands that run on a window. |
| `window_rounds` | `5000` | iterations a window island spends on one window before moving to another. |
| `window_min` | `16` | smallest window, in detectors. |
| `explorer_restart` | `0` | restart interval of the uniform chain (`0` = same as the others). `1` turns it into a fresh uniform information set every iteration. |
| `max_codewords` | `2000` | minimum-weight solutions stored per island. |
| `m4ri_k` | `8` | block size of the Method of Four Russians [[9](#ref-9)] in elimination. |
| `proj_words` | `1` | 64-bit words of random parities used for verification (64 parities per word). |
| `migrate_interval`, `migrate_boost` | `100`, `0.0` | sharing of the island's best support between its chains (off by default: it made chains converge on one codeword). |
| `stall_rounds` | `0` | restart a whole island after this many iterations without improvement (off). |
| `greedy_kicks`, `kick_size`, `mix_rate`, `lb_hash_p4` | `0`, `2`, `0.0`, `0` | experimental; not used by the default compact search. |

---

## Benchmarks and comparisons

Runs stop at the known or published distance (`cutoff`) unless stated otherwise. The two tools run on the same machine type (Intel Xeon
Platinum 8268), **16 threads for dist-m4ri vs 16 worker processes for Sisyphus**; the
small and medium circuit instances were run in the same job.
Times are min / median / max over 3 seeds. Both tools are randomized, so the spread
comes from the random seed. Every witness of both tools is verified (circuits: on
the full DEM).

dist-m4ri ([QEC-pages/dist-m4ri](https://github.com/QEC-pages/dist-m4ri),
v0.14.7) is the current reference tool for this problem. Its random-window
information-set algorithm (`method=1`) is the upper-bound search compared here,
with default parameters unless noted.

### Codes

| instance | distance | dist-m4ri | Sisyphus |
|---|---|---|---|
| gross [[144,12,12]] | 12 | < 0.01 s | 0.06 s |
| double gross [[288,12,18]] X / Z | 18 | < 0.01 / < 0.01 s | 0.09 / 0.09 s |
| [[270,54,10]] ("Mitten" family of the dist-m4ri benchmark) X / Z | 10 | < 0.01 s | 0.09 s |
| [[360,12,≤24]] bivariate bicycle X / Z | 24 | < 0.01 s | 0.09 s |
| [[756,16,≤34]] bivariate bicycle X / Z | 34 | 0.07–0.33 / 0.15–0.68 s | 0.11–0.12 / 0.10–0.12 s |
| [[1152,580,≤12]] affine-permutation (APM) | 12 | 0.00 / 0.00 / 0.01 s | 0.09 / 0.09 / 0.10 s |
| [[2304,1156,≤14]] APM | 14 | 0.11 / 0.26 / 0.45 s | 0.21 / 0.24 / 0.30 s |
| [[4608,2308,≤20]] APM, Z side | 20 | 783, 1157 s (2 seeds) | **139, 591 s** |
| [[4608,2308,≤20]] APM, X side | 20 | 1 of 3 seeds in 1800 s (102 s) | 1 of 3 seeds in 1800 s (1008 s) |

- **Small and medium codes** (up to a few thousand qubits): both tools find the
  distance in well under a second. dist-m4ri is faster below 0.1 s, where Sisyphus's
  time is dominated by starting its worker processes.
- **Large codes**: on the [[4608,2308,≤20]] code Sisyphus is 2–5× faster on the Z
  side, where the minimum-weight logicals are concentrated on few blocks and the
  learned column weights pay off. The X side is a tie: about 25 core-hours per
  weight-20 hit for both tools.

### Circuit-level instances

The circuits are the Stim circuits of the dist-m4ri benchmark set: bivariate
bicycle, gross and double-gross memory experiments with SI1000 or uniform noise,
the "Mitten" codes with two syndrome-extraction schedules each, and a toric-layout code.
Each detector error model was reduced exactly to one basis, and the reduced
sizes match those reported by dist-m4ri.

Small and medium circuits: **both tools in the same job on the same node**, 3 seeds
each, stopping at the published value (2026-11-05). dist-m4ri's time is its own
search clock, which excludes reading and reducing the DEM file. Sisyphus's time is the
whole call, including its setup (elimination, starting worker processes).

| instance | reduced size | stop at | dist-m4ri | Sisyphus | median speedup |
|---|---|---|---|---|---|
| bb72 SI1000 X / Z, uniform X | 252 × 2,232 | 6 | 0.00 s | 0.08 s | – |
| bb90 SI1000 X / Z | 495 × 4,590 | 8 | 0.01 s | 0.15 s | – |
| bb108 SI1000 X / Z | 594 × 5,508 | 8 | 0.01 s | 0.16 s | – |
| torus200 X / Z | 300 × 4,800 | 8 | 0.01–0.09 s | 0.18–0.27 s | – |
| Mitten 90–330, both schedules (10 circuits) | up to 792 × 10,890 | 4–9 | 0.00–0.17 s | 0.09–0.48 s | – |
| gross SI1000 X / Z | 936 × 8,784 | 10 | 0.03–0.10 s | 0.21–0.23 s | – |
| gross uniform X | 936 × 8,784 | 11 | 0.02–0.04 s | 0.27–0.30 s (finds **10**) | – |
| double gross SI1000 X | 2,736 × 26,208 | 18 | 2.2 / 2.4 / 7.4 s | 2.6 / 2.7 / 2.8 s | 0.9× (tie) |
| double gross uniform X | 2,736 × 26,208 | 18 | 7.2 / 8.9 / 40 s | **2.6 / 2.8 / 3.2 s** | **3.2×** |
| double gross SI1000 Z | 2,736 × 26,208 | 18 | 59 / 71 / 81 s | **2.6 / 3.3 / 3.3 s** | **22×** |
| [[360,12,24]] SI1000 X | 4,500 × 43,560 | 23 | 41 / 192 / 482 s | **3.9 / 3.9 / 7.7 s** (finds **22**) | **49×** |
| [[360,12,24]] SI1000 Z | 4,500 × 43,560 | 23 | 139 / 195 / 291 s | **3.9 / 3.9 / 3.9 s** | **51×** |

Large circuit (16 threads vs 16 workers, separate runs):

| instance | reduced size | dist-m4ri | Sisyphus |
|---|---|---|---|
| [[756,16,34]] SI1000 X, 34 rounds | 13,230 × 129,276 | 54 at 72 s; 50 at 238–553 s; best 49 in 1800 s | **47–49 at ~80 s; 39 at ~82 s; 33 at 199 s** |

- **Small circuits** (up to about 10⁴ faults): both tools reach every published
  value in well under a second. dist-m4ri's search is faster in absolute terms
  (0.00–0.17 s vs 0.08–0.48 s); Sisyphus's time there is mostly fixed setup cost.
- **Medium circuits** (2.7k–4.5k detectors, 26k–44k faults): Sisyphus reaches the
  published distance **3–51× faster at the median** on four of the five
  instances and ties on the fifth. Its run-to-run spread is also much smaller
  (bb360 X: 3.9–7.7 s vs 41–482 s).
- **Large circuits**: on the 34-round [[756,16,34]] circuit Sisyphus finds lower
  weights sooner: 47–49 in about 80 s, then 39 and 33, where dist-m4ri's best
  in 30 minutes is 49.

### Bounds tighter than the best published

All found by Sisyphus. Each witness is in
[`benchmarks/witnesses/`](benchmarks/witnesses/) as a list of faults (the
detectors and observables each fault flips) and can be checked against the
original circuits with Stim:

```bash
python benchmarks/verify_witnesses.py /path/to/dist-m4ri/benchmark
```

| circuit (dist-m4ri benchmark) | published | **Sisyphus** | notes |
|---|---|---|---|
| `bb_756_16_34_si1000_r34`, X basis | not computed; the code's distance is ≤ 34 and the lightest reported circuit fault set has weight 55 | **33** | below the code distance bound: the syndrome-extraction schedule loses distance to hook errors. dist-m4ri's best in 30 min: 49 |
| `bb_360_12_24_si1000_r24`, X basis | [12, 23] | **21** | found in 120 s (2 of 2 seeds) |
| `bb_360_12_24_si1000_r24`, Z basis | not computed | **21** | found in 120 s (2 of 2 seeds) |
| `double_gross_288_12_18_si1000_r18`, X basis | [12, 18] | **16** | found in 120 s (2 of 2 seeds) |
| `double_gross_288_12_18_si1000_r18`, Z basis | not computed | **16** | found in 120 s (2 of 2 seeds) |
| `double_gross_288_12_18_uniform_r18`, X basis | not computed | **16** | found in 120 s (2 of 2 seeds) |
| `gross_144_12_12_uniform_r12`, X basis | [7, 11] | **10** | found in under 1 s |

At the code level, no published bound was improved. Sisyphus reaches the published
weight 20 on [[4608,2308,≤20]] and 34 on [[756,16,≤34]].

### Where dist-m4ri is still ahead

- Small instances (below about 10⁴ columns), where both finish in under a second
  but Sisyphus's fixed setup cost (0.1–0.5 s) dominates.
- dist-m4ri also computes **lower bounds** (its connected-cluster algorithm,
  `method=2`), and with them exact distances on small instances. Sisyphus only
  gives upper bounds.
- One-round circuits of very large codes: on a one-round syndrome-extraction
  circuit of a [[2232,1120,16]] code, dist-m4ri reached 13 (Z basis) where
  Sisyphus's best run reached 14.

---

## Limitations

- **Upper bounds only.** Sisyphus finds light solutions and verifies them; it cannot
  prove that nothing lighter exists. Use an exact method (for example
  dist-m4ri's connected-cluster search) for lower bounds.
- **Dense input.** Matrices are passed as dense arrays (1 byte per entry before
  packing), which matters above about 10⁵ × 10⁵.
- **`time_limit` needs `num_workers ≥ 2`**; a single in-process search runs
  until its iteration budget is spent.
- **First call compiles** the kernels (about 30 s; cached afterwards).

## Repository layout

```
Sisyphus/
├── sisyphus_qec/
│   ├── search.py         public entry points (css_distance, side_distance) and the
│   │                     parallel island search (process pool, scheduling, early stop)
│   ├── setup.py          per-side setup: packed check matrix, verification parities,
│   │                     starting basis
│   ├── chain_compact.py  search chain on the compact [I | P] form (default)
│   ├── chain_full.py     search chain on full-width rows (compact=False)
│   ├── extraction.py     candidate extraction (single rows, sums of rows, collision search)
│   │                     and greedy weight reduction
│   ├── windows.py        detector time windows for circuit-level models
│   ├── gf2.py            bit-packed GF(2) linear algebra (Four-Russians row reduction)
│   ├── params.py         default search parameters
│   ├── results.py        result types
│   └── dem.py            Stim detector error model helpers
├── examples/             code_distance.py, circuit_distance.py, codes.py
├── tests/                pytest suite (known distances, brute force on random codes,
│                         a circuit)
└── benchmarks/
    ├── witnesses/        fault sets behind the new circuit-level bounds
    └── verify_witnesses.py
```

Run the tests with `python -m pytest tests/`.

## Citing

If you use Sisyphus in published work, please cite Sisyphus, the QDistRnd paper, and the
distance-finding paper of Webster, Jacob and Higgott:

```bibtex
@software{sisyphus,
  title   = {{Sisyphus}: fast randomized distance search for quantum codes and circuits},
  author  = {{Sisyphus developers}},
  year    = {2026},
  url     = {https://github.com/varunmenon1310/sisyphus}
}

@article{qdistrnd,
  title   = {{QDistRnd}: A {GAP} package for computing the distance of quantum
             error-correcting codes},
  author  = {Pryadko, Leonid P. and Shabashov, Vadim A. and Kozin, Valerii K.},
  journal = {Journal of Open Source Software},
  volume  = {7},
  number  = {71},
  pages   = {4120},
  year    = {2022},
  doi     = {10.21105/joss.04120}
}

@article{webster2026distance,
  title   = {Distance-Finding Algorithms for Quantum Codes and Circuits},
  author  = {Webster, Mark and Jacob, Abraham and Higgott, Oscar},
  journal = {arXiv preprint arXiv:2603.22532},
  year    = {2026}
}
```

## References

Information-set decoding and low-weight codeword search:

1. <a id="ref-1"></a>E. Prange, "The use of information sets in decoding cyclic codes", *IRE
   Transactions on Information Theory* 8(5), S5–S9 (1962). Information sets;
   reading codewords off a systematic generator matrix.
2. <a id="ref-2"></a>P. J. Lee and E. F. Brickell, "An observation on the security of McEliece's
   public-key cryptosystem", *EUROCRYPT 1988*, LNCS 330, 275–280 (1988). Sums of
   several rows.
3. <a id="ref-3"></a>J. Stern, "A method for finding codewords of small weight", *Coding Theory and
   Applications*, LNCS 388, 106–113 (1989). Collision search on a subset of
   redundant columns.
4. <a id="ref-4"></a>A. Canteaut and F. Chabaud, "A new algorithm for finding minimum-weight words
   in a linear code: application to McEliece's cryptosystem and to narrow-sense
   BCH codes of length 511", *IEEE Transactions on Information Theory* 44(1),
   367–378 (1998). Moving between information sets by single column swaps.

Distance of quantum codes and circuits:

5. <a id="ref-5"></a>L. P. Pryadko, V. A. Shabashov and V. K. Kozin, "QDistRnd: A GAP package for
   computing the distance of quantum error-correcting codes", *Journal of Open
   Source Software* 7(71), 4120 (2022).
6. <a id="ref-6"></a>L. P. Pryadko and W. Zeng, dist-m4ri, https://github.com/QEC-pages/dist-m4ri.
   Random-window information sets and connected-cluster search for codes and
   detector error models; the circuit-level benchmark set used here.
7. <a id="ref-7"></a>M. Webster, A. Jacob and O. Higgott, "Distance-Finding Algorithms for Quantum
   Codes and Circuits", arXiv:2603.22532 (2026).
8. <a id="ref-8"></a>C. Gidney, "Stim: a fast stabilizer circuit simulator", *Quantum* 5, 497
   (2021). Detector error models.

Linear algebra over GF(2):

9. <a id="ref-9"></a>V. L. Arlazarov, E. A. Dinic, M. A. Kronrod and I. A. Faradzev, "On economical
   construction of the transitive closure of an oriented graph", *Soviet
   Mathematics Doklady* 11, 1209–1210 (1970). The Method of Four Russians.
10. <a id="ref-10"></a>M. R. Albrecht, G. V. Bard and W. Hart, "Algorithm 898: Efficient
    multiplication of dense matrices over GF(2)", *ACM Transactions on
    Mathematical Software* 37(1), 9 (2010). The M4RI library.

Search heuristics:

11. <a id="ref-11"></a>H. H. Hoos and T. Stützle, *Stochastic Local Search: Foundations and
    Applications*, Morgan Kaufmann (2004). Stochastic hill climbing with restarts.
12. <a id="ref-12"></a>H. R. Lourenço, O. C. Martin and T. Stützle, "Iterated local search",
    *Handbook of Metaheuristics*, 320–353 (2003). Perturbation and restart from a
    local optimum.
13. <a id="ref-13"></a>D. Whitley, S. Rana and R. B. Heckendorn, "The island model genetic
    algorithm: on separability, population size and convergence", *Journal of
    Computing and Information Technology* 7(1), 33–47 (1999). Independent
    subpopulations in parallel.
14. <a id="ref-14"></a>D. E. Goldberg and K. Deb, "A comparative analysis of selection schemes used
    in genetic algorithms", *Foundations of Genetic Algorithms*, 69–93 (1991).
    Tournament selection.

Codes used in the benchmarks:

15. <a id="ref-15"></a>S. Bravyi, A. W. Cross, J. M. Gambetta, D. Maslov, P. Rall and T. J. Yoder,
    "High-threshold and low-overhead fault-tolerant quantum memory", *Nature*
    627, 778–782 (2024). Bivariate bicycle codes.

## License

Sisyphus is released under the Apache License 2.0 (see [LICENSE](LICENSE)).
