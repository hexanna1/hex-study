# Local Study Working Notes

This file records reusable principles for local-pattern study setup.

## Evaluation Framing

- Use a consistent quantitative frame across studies:
  - compare positions/moves in logistic-odds space,
  - map to "fractions of a stone" for interpretation.
- When doing balancing studies, define an explicit target in that same frame.
  - Example: with one red and one blue balancing stone, a target like
    `red_value - blue_value ~= -0.5` is a good way to offset first-move advantage.

## Raw-NN Side-to-Move Bias

- Raw-NN outputs usually carry a systematic side-to-move offset (a few percentage points in practical runs).
- Do not treat raw root eval and best-child eval as Bellman-consistent by default.
  - In practice, root winrate can be noticeably above or below all sampled child winrates.
- Prefer comparison structures that naturally cancel this bias:
  - paired with/without differences using the same ply transition (`child - root` in both branches),
  - and same-ply child-to-child gaps when estimating generic move-vs-pass terms.
- Keep this caveat explicit in writeups when reporting percentages from raw-NN studies.

## Prior Stability

- Distinguish carefully between raw-NN policy priors and live-search priors.
- Raw-NN policy priors are deterministic and are the default choice when a study needs a stable candidate-generation signal.
- Live-search priors with `analysisWideRootNoise = 0` can be stable enough to use empirically, but they should still be treated as search outputs rather than as the canonical policy.
- Live-search priors with `analysisWideRootNoise > 0` are intentionally nondeterministic exploration outputs and should not be used as stable threshold signals.

## Symmetry Discipline

- Always remove redundant work from symmetry-equivalent cases.
- At minimum, dedupe under 180-degree board rotation for corner-placement sweeps.
- For interior-only pattern work, stronger symmetry collapsing (rotations/reflections) may be appropriate.
- When needed, enforce one canonical orientation (for example fixing which corner family red is sampled from).

## Geometry and Corner Families

- Corner context is not interchangeable; distinguish acute and obtuse families explicitly.
- Edge context is inherently asymmetric with respect to color goals.
  - In opening-like contexts, moves near your own edge are much weaker than analogous moves near the opponent's edge.
  - Example on 19x19 (ignoring swap): `d10` is strong for red while `j4` is quite weak.
- Do not assume coordinate-swapped offsets are equivalent in same-color same-corner comparisons.
  - In particular, `4-2` and `2-4` in obtuse-corner contexts are not equivalent.
- Scaled analogies are useful but should be validated empirically.
  - Example: "3-3 obtuse" on 21x21 (such as `s3`) is equivalent to `q3` on 19x19.

## Local vs Global Bias (Calibration Motivation)

- A local-pattern score is often depressed by global geometry even when the pattern-relative move choice is good.
  - Example: a move closer to your own edge may look weak partly because your edge already supplies strength there, not because it is a poor reply to the local pattern.
- This is a reason to study local patterns first in an edge/corner-neutral setting, then apply a separate baseline correction for generic center/edge geometry effects when needed.
- Keep the conceptual separation clear:
  - local-pattern evaluation asks "how good is this move for this pattern?",
  - baseline geometry calibration asks "how much of that score comes from a generic board-shape prior?"

## Coordinate Equivalences and Near-Fungibility

- Remember color/goal-equivalent coordinate transforms when comparing red-vs-blue corner openings.
  - Example: red `a3` is comparable to blue `c1` (row/col swap under side-role symmetry).
- Some opening cells are near-fungible and belong in the same bucket in coarse sweeps.
  - Example (red perspective): `a1`/`b1`/`c1` are very similar, while `d1` is meaningfully different.
  - Example (red perspective): `a2`/`b2`/`c2` are very similar, while `d2` is meaningfully different.
  - Counterexample: `a3` and `b3` are not similar and should stay separate.
- Practical coarse-graining heuristic:
  - many mid-row points on the same early row (for example red first row) are approximately fungible,
  - similarly for many mid-row points on second and third rows,
  - but treat near-corner transitions as potential regime changes and re-check empirically.

## Acute-Corner Candidate Reductions

- With red `b4`, blue `c2`, and `a1-a4,b1-b3` all empty, KataHex has a tendency to spend policy on useless moves in that dead region.
- Those cells should not consume candidate budget.
- With red `b4/c4`, blue `c3`, and `a1-a4,b1-b3,c1-c2` all empty, blue `c2` and blue `b3` are provably equivalent local continuations, so only one representative should be evaluated.
- Apply these reductions across the corresponding acute-corner rotations and color/goal-equivalent variants.
- Treat these as candidate-generation policy rules, not as a claim that the study databases globally quotient out all such positions.

## Board-Size Transfer

- Small-board runs can be used as fast-pass proxies for larger-board studies.
- Expect baseline skew to change with board size (for example, 19x19 is more skewed than 21x21).
- Prefer a two-stage flow:
  1. coarse filter on a cheaper board size,
  2. confirmation on target board sizes.

## Expansion/Compression Artifact

- The spread between `pass_proxy` and the best move (viewed in winrate/logit/Elo space) is not stable across contexts.
- Empty-board contexts are relatively expanded, while other contexts can be strongly compressed.
- Compression/expansion is not explained only by move count; specific position geometry matters.
- This shows up as a real context artifact in normalization denominators based on worst/pass vs best gaps.

## Search Budgeting

- In candidate mode, treat `search_seconds` as per-candidate budget.
- Practical heuristic:
  - choose a seconds-per-candidate value,
  - and scale that per-candidate value by board size (smaller boards generally need less time per candidate).
  - example defaults:
    - around `1` second per candidate on 19x19 proxy sweeps,
    - around `2` seconds per candidate on 21x21 confirmation sweeps.
- Adjust upward for final confirmation passes after coarse filtering.

## Sweep Design Hygiene

- Keep objective function explicit before sweeping (what counts as "good").
- Track enough metadata to reproduce and compare runs:
  - board size,
  - symmetry policy,
  - candidate-generation rule,
  - time budget,
  - noise/root settings.
