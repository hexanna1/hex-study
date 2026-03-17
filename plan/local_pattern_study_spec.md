# Local Pattern Representative Study Spec

## Purpose

Given one HexWorld source position, estimate local reply strength robustly by:

1. generating symmetry/balance representatives,
2. evaluating paired with/without-pattern candidate sets for each representative,
3. canonicalizing candidates by orbit key within each representative,
4. computing pass-anchored post-ablation corrected values per representative,
5. pooling by canonical candidate identity, then normalizing once at the end.

This is written for someone implementing the workflow from scratch.

---

## Core Idea

For each representative position:

- start with a **base local candidate set** (near-pattern cells),
- always include a **pass anchor** candidate (`pass_proxy`),
- optionally include one **outside-local** candidate (`tenuki`) from a raw-NN root probe,
- run paired candidate-mode evaluation on:
  - the with-pattern representative,
  - the matched without-pattern representative,
- compute corrected values on orbit-canonical keys with `pass_proxy = 0`,
- pool those corrected values across representatives by canonical key,
- normalize once at the end so the pooled best key is `1`.

The final reported stone-fraction values are therefore pooled-relative scores, not per-representative normalized scores.

## Why This Workflow Is Needed

The target question is narrow:

- “How good is this move because of the local pattern?”

Direct evaluations can mix in:

- broad edge/corner geometry effects,
- search evaluation noise,
- calibration noise (if a correction layer is added).

Practical consequence:

- keep the local-pattern score and the calibration signal conceptually separate,
- and keep all calibration terms in additive space (log-odds / Elo) until the final normalization after representative pooling.

Method note:

- this workflow uses raw-NN candidate evaluations as a fixed policy because they are faster, avoid search evaluation noise, and make reruns deterministic.
- when additional compute is available, it is preferable to spend it on averaging over more balance-stone contexts than on search.

## Why Add a Calibration Correction Layer?

The representative interpolation step answers:

- “How good is this move relative to `pass_proxy` and the best candidate in this representative?”

That is useful, but it can still mix in broad board-shape effects.

Example motivation:

- holding the local pattern fixed, moves closer to your own edge are weaker,
- so a move in that region looks weak in a local pattern evaluation,
- not because it is weak relative to the pattern,
- but because it is weak relative to the global board geometry (especially edge proximity).

Here, "global board geometry" means context such as distance/orientation to edges and corners, independent of the local pattern itself.

The calibration correction layer is meant to subtract part of that broad geometric bias, so the local study is less dominated by effects that are present even without the pattern.

## Terminology

- `absolute candidate`:
  - a board cell in one specific representative position (for example `l13` in rep #3).
- `rep-relative candidate`:
  - that same move mapped into the shared pattern-local coordinate frame so candidates from different representatives are comparable.
- `orbit-canonical key`:
  - a deterministic symmetry-coalesced key for pooling equivalent rep-relative candidates.

## Symmetry Flow (High-Level)

This workflow has two separate symmetry steps that should not be mixed up:

- representative symmetry step:
  - start from the selected transform family (for `d6`, that is 12 rotation/reflection transforms),
  - keep only distinct representative positions after deduplication.
- candidate symmetry step:
  - within each retained representative, still evaluate all generated candidates,
  - map each absolute candidate to a rep-relative candidate, then coalesce symmetry-equivalent rep-relative candidates into one orbit-canonical key for pooling.

Practical intuition for symmetric interior cases:

- if there are `R` distinct representatives under `d6`, then a generic orbit-canonical class is expected to have about `12 / R` equivalent absolute candidates per representative.
- example: `R = 6` typically implies paired equivalents (`2` cells) per class in each representative.

Edge-conditioned studies use a smaller symmetry family because the representative context must preserve the chosen edge regime. A good default is bilateral reflection that preserves that edge context. These studies therefore retain fewer representatives than centered `d6` studies.

Worked example:

- source: `https://hexworld.org/board/#21c1,k11l11`.
- local candidate rule: `pass_proxy` plus adjacent empty candidates only (`candidate_Δ_max=1`, as defined later).
- retained representatives (`R=6`): `rep1`, `rep2`, `rep3`, `rep4`, `rep5`, `rep6`.
  - `rep1`: `https://hexworld.org/board/#21c1,j11k11`
  - `rep2`: `https://hexworld.org/board/#21c1,k10k11`
  - `rep3`: `https://hexworld.org/board/#21c1,k11j12`
  - `rep4`: `https://hexworld.org/board/#21c1,k11j11`
  - `rep5`: `https://hexworld.org/board/#21c1,k11k10`
  - `rep6`: `https://hexworld.org/board/#21c1,j12k11`
- absolute candidates for `rep1` (9 total):
  - `k1` (`pass_proxy` move), `i11`, `i12`, `j10`, `j12`, `k10`, `k12`, `l10`, `l11`.
- orbit-canonical key -> absolute candidates in `rep1`:
  - `pass_proxy -> k1`
  - `-1,0 -> i11`
  - `-1,1 -> i12,j10`
  - `0,1 -> j12,k10`
  - `1,1 -> k12,l10`
  - `2,0 -> l11`

---

## Inputs

Required:

- source HexWorld URL/hash.

Typical optional controls:

- board size (default: source size),
- placement context (`centered`, default, or `edge`),
- symmetry policy (default: `d6`),
- candidate radius threshold (`candidate_Δ_max`, default `7`),
- tenuki minimum squared distance from pattern (`tenuki_Δ_min`, default `21`),
- `top_n` (optional).

Recommended defaults:

- for katahex `20240812`, prefer board sizes around `20-22` when the source context does not force another size.
- balance profile: `a1,d2`.
- candidate evaluation mode: raw-NN.
- use a balanced non-empty ablation context rather than an empty board.
- without-pattern branch policy:
  - red to play: remove pattern stones from the representative (local matched ablation),
  - blue to play: use canonical near-50% root-eval baseline `a1,d2,move3` (blue to play), where `move3` is the top-prior red move from a raw-NN probe of `a1,d2` on the same board size.

For katahex `20240812`, this board-size range keeps central studies away from edge effects and unstable near-`0%` / near-`100%` regions. It also avoids the larger-board bias regime that appears beyond about the `22 -> 23` transition for this net.

A balanced non-empty ablation context avoids the empty-board compression/expansion artifact, where the additive gap between `pass_proxy` and the best move is inflated relative to ordinary study positions. It also avoids the stronger first-move imbalance of the empty board. That imbalance makes additive comparisons noisier.

---

## Hex-Specific Policy Notes

These choices are domain-specific (not generic ML/reporting defaults):

- `pass_proxy` is not a literal pass token. It is a real board move used as a stable `0` anchor.
- `pass_proxy` location policy:
  - red to play: middle of first row (scan center-out if occupied),
  - blue to play: middle of first column (scan center-out if occupied).
- Reason for this proxy: pass handling is often unstable/inconvenient in engine workflows; a deterministic weak edge move is a practical substitute anchor.
- `tenuki` means “best move outside the local candidate set”, not “best move overall”.
- Candidate locality uses hex-axial Euclidean distance with 60° basis:
  - `n = dq^2 + dq*dr + dr^2`.
  - Write `Δn` for squared distance `n`.

---

## End-to-End Algorithm

### 1) Parse and extract pattern

From source position:

- parse occupancy and side-to-play,
- define pattern stones as `plus` (side to play) and `minus` (opponent),
- convert to relative coordinates anchored to a deterministic local origin.

### 2) Generate representatives

- expand under selected symmetry policy,
- place each orientation in the chosen representative context, either middle-board or edge-conditioned,
- apply fixed balancing stones,
- discard invalid placements (out of bounds, overlaps, duplicates).

Representative grid is `balance x orientation` (no translation jitter in this flow).

### 3) Build base local candidates

Per representative, generate candidate cells by local proximity to pattern stones:

- include empty cells within `candidate_Δ_max` distance of pattern,
- prepend `pass_proxy` candidate.

`pass_proxy` is always explicit and is used as the `0` anchor for stone fractions.

### 4) Root-probe tenuki augmentation

Per representative:

- run a raw-NN root probe (no candidate restriction),
- define `base_local_candidates` as the pre-augmentation set,
- from root-probe moves outside `base_local_candidates`, keep only moves with minimum squared distance to all pattern stones at least `tenuki_Δ_min`,
- choose the top-ranked remaining root-probe move (policy prior order),
- if found, append it as `tenuki`.

If no such move is found, skip tenuki augmentation for that representative.

### 5) Candidate-mode evaluation

Per representative, run candidate mode with:

- candidates = base local candidates + optional `tenuki`,
- evaluate candidate children in raw-NN mode.

### 6) Canonicalize candidates

Map each candidate to a canonical key:

- absolute candidates:
  - first map absolute candidates to rep-relative candidates,
  - then collapse symmetry-equivalent rep-relative candidates into one orbit-canonical key
    (choose one deterministic representative per symmetry orbit),
- pass anchor candidate: key `pass_proxy`,
- outside-local augmentation candidate: key `tenuki`.

This keying is used both for paired ablation construction (within representative) and for final cross-representative pooling.

### 7) Calibration correction (paired with/without pattern) on canonical keys

Per representative:

- build the **without-pattern** position:
  - red to play: remove pattern stones while keeping non-pattern stones fixed (including balancing stones),
  - blue to play: use canonical near-50% root-eval baseline `a1,d2,move3` (same board size),
- evaluate the root and candidate children for both:
  - with-pattern representative,
  - without-pattern representative,
- coalesce candidate rows by orbit-canonical key,
- compare with/without differences in log-odds space to estimate a pattern-specific interaction term per canonical key.

### 7a) Semantics and model

Semantics first (what this bounded score should mean):

- `0` = pass-like floor (`-1`-class move: effectively gives up one move of value),
- `1` = best local move under the pattern (within the evaluated paired canonical-key set),
- `1 - score(c)` = local-pattern mistake size relative to that best local move.

This is a pass-anchored local move-quality score, not a pure interaction score. In particular, `0` is **not** meant to mean “interaction-neutral.”

Motivating axioms:

- work in an additive evaluation space (log-odds / Elo),
- pattern-independent move effects should cancel in candidate comparisons,
- the generic move/pass gap must remain present in the bounded score (pure ablation cancels it),
- anchors and candidates must be defined in the same corrected quantity.

Why pure ablation is not enough:

- let `L(.)` denote the additive evaluation scale used for calibration (log-odds / Elo),
- let `Δw(c) = L(W+c) - L(W)` for the with-pattern root `W`,
- let `Δu(c) = L(U+c) - L(U)` for the ablated root `U`,
- the pure ablation term `I(c) = Δw(c) - Δu(c)` isolates pattern interaction,
- but for a pass-like move it cancels the generic “spent a move vs pass” effect and makes pass-like play look near interaction-neutral (`0`-class), which is the wrong bounded-score semantics.
- in practice with raw-NN, this paired-difference structure is useful because it compares the same ply transition in both branches (`child - root`), which helps cancel side-to-move offsets in `I(c)`.

Edgeless / infinite-board symmetry prior (explicit):

- once the local pattern is ablated, there is no global edge/corner geometry to distinguish one ordinary move location from another,
- therefore the generic (non-pattern) component is treated as a move-class effect:
  - one shared value for ordinary non-pass moves,
  - a distinct class for pass.

This motivates the decomposition:

- `Q(c) = g * m(c) + I(c)`,
- where `m(c)=1` for ordinary moves and `m(c)=0` for pass.

Here `g` is the generic non-pass move value (in the same units as `I`, i.e. log-odds / Elo).

Practical estimator choice for `g`:

- let `u(c) = L(U+c)` be the ablated-branch child value in additive space,
- use
  - `ĝ = max_c [u(c) - u(pass_proxy)]`,
- this keeps the generic class term on same-ply child-to-child comparisons in the ablated branch, which is less susceptible to side-to-move offsets in raw-NN outputs.

Alternative estimator (not used here):

- `g_root = -Δu(pass_proxy) = L(U) - L(U+pass_proxy)`,
- this is a root-vs-child estimate and is more exposed to raw-NN side-to-move/root-child inconsistency artifacts.
- if the evaluator were Bellman-consistent in the ablated branch (`L(U) = max_c L(U+c)`), then `ĝ` and `g_root` coincide.

### 7b) Practical score construction

Practical bounded score construction:

- define the corrected pass-referenced value
  - `V(c) = ĝ * m(c) + [I(c) - I(pass_proxy)]`,
- treat `pass_proxy` as the pass-class anchor for scoring (`m(pass_proxy)=0`), so `V(pass_proxy)=0` by construction,
- keep `V(c)` as the per-representative corrected additive quantity for pooling.

This preserves the intended semantics (`0` = pass-like floor, positive values = locally useful moves) while still using ablation to remove pattern-independent artifacts from candidate comparisons.

Caveat:

- unequal red/blue pattern stone counts are acceptable for within-pattern candidate comparison,
- but cross-pattern absolute comparisons should treat pattern stone counts as part of the pattern definition unless a separate normalization policy is specified.

### 8) Pool corrected values across representatives

Use `V(c)` values from section 7b:

- aggregate one corrected value per `(representative, canonical key)` row,
- compute pooled corrected values by canonical key,
- keep `pass_proxy` at `0` on this pooled additive scale.

### 9) Final normalization after pooling

After pooling corrected values by canonical key:

- let `best_pooled = argmax_c pooled_V(c)`,
- define
  - `score(c) = pooled_V(c) / pooled_V(best_pooled)`,

This gives the final reported stone-fraction values after representative averaging rather than before it.

### 10) Optional pre-ablation diagnostic view

Use candidate-mode results only:

- convert winrates in log-odds/logit space,
- anchor `pass_proxy` at `0`,
- anchor best candidate in that representative at `1`,
- interpolate all candidates to stone-fraction values.

These values are diagnostic only. They are not the final reported local-pattern score.

### 11) Optional exhaustive pattern sweep

Instead of starting from one supplied HexWorld position, this workflow can start from exhaustive interior-pattern enumeration.

In that variant:

- enumerate canonical simple labeled patterns up to a chosen move cap,
- keep only the supported labeled count families:
  `plus = minus` (red to move),
  `minus = plus + 1` (blue to move),
  `plus = minus + 1` (red to move),
  or `minus = plus + 2` (blue to move),
- keep only connected shapes with maximum pairwise span bounded by the chosen `Δ`,
- reject only patterns with immediate same-color adjacency (`min(min_rr, min_bb) = 1`) when the nearest red/blue contact is at least `Δ4`,
- then run the previous steps for each retained pattern.

This family split is easy to misread if one starts from actual red/blue counts instead of labeled notation.

- The stored pattern key is always labeled relative to the side to move: `plus` means “to move”, `minus` means “opponent”.
- A one-tenuki state can therefore create either:
  - a red-to-move family with one more `plus` stone than `minus` (`plus = minus + 1`),
  - or a blue-to-move family with two more `minus` stones than `plus` (`minus = plus + 2`).
- Both are needed. For example, `+[0,0]-[]` and `+[]-[0,0]` are different labeled patterns with different continuations:
  - `+[0,0]-[]` can continue to `+[0,0:0,1]-[]`,
  - `+[]-[0,0]` can continue to `+[0,1]-[0,0]`.
- The move cap counts one tenuki as one move. So `m5` means:
  - up to 5 local moves for no-tenuki families,
  - up to 4 visible local stones for one-tenuki families.

Recommended sweep settings:

- `max_moves = 5` (`m5`), counting one tenuki as one move,
- span bounded by `Δ13`.

This produces a full sweep over one explicitly defined family of small local patterns rather than one hand-selected source position.
