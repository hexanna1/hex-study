# Opening + Joseki Database Spec

## Purpose

Build compact continuation databases by expanding plausible moves from a current frontier and pruning lines that do not deserve further study budget.

This document covers two related workflows:

- a full-board opening database,
- and a local-corner joseki database.

The goal is to describe the shared model and the important pruning choices at a high level.

---

## Core Idea

Both workflows treat database construction as repeated frontier expansion.

For each node on the current frontier:

1. realize the current study state,
2. run a KataHex raw-NN root probe,
3. admit a small candidate set,
4. evaluate the admitted continuations,
5. convert the checked results into comparable continuation weights,
6. multiply those weights into cumulative importance,
7. keep only the continuations that still deserve downstream budget,
8. continue one frontier layer deeper.

The important question is not “what is the single best move here?” It is “which continuations deserve to remain in a compact study database?”

---

## Why This Workflow Exists

A plain best-line search is too narrow for study purposes, while an unrestricted continuation tree grows too quickly to stay interpretable.

The workflow therefore uses:

- raw-NN policy to decide which moves are worth checking,
- child-position evaluation only as a local continuation score,
- cumulative importance to control future tree budget,
- and explicit pruning rules to keep the database focused.

The result is a study tree that stays broad near the root while cutting away continuations that are unimportant, redundant, or locally inert.

---

## Two Study Modes

The shared workflow supports two different node models.

### Openings

The opening database is position-first.

Each node corresponds to:

- one realized whole-board position,
- plus the ordered move sequence used to reach it.

The realized position is the expansion state.
Different move orders that reach the same position are not treated as different opening identities merely because the order differed.

### Joseki

The joseki database is line-first.

Each node corresponds to:

- one ordered corner-local line,
- plus the realized position produced by that line inside a fixed family-specific balanced context.

The ordered local line is the primary identity.
Different lines may transpose to the same realized position and still remain different joseki nodes.

---

## Shared Expansion Workflow

### 1) Realize the frontier

Each frontier node is converted into one concrete study position.
For openings, that is the whole-board position itself. For joseki, that is the realized corner line inside its fixed balance context.

### 2) Probe the position

Run a KataHex raw-NN root probe on the realized position.
Use it to rank candidate continuations by policy prior and to supply local child-position scores for the continuations that are later checked.

### 3) Admit a small candidate set

Only a small subset of continuations is allowed to consume child-evaluation budget.

The admission rule differs by study mode:

- openings use a small whole-board top-`k` policy, with a programmatically derived fair-opening root set,
- joseki uses local-corner continuations plus at most one tenuki continuation, with a curated family-specific root set.

### 4) Evaluate admitted continuations

For each admitted continuation:

- realize the child position,
- evaluate it as needed,
- and interpret the result as a local continuation score for the player who just moved.
These scores are used only locally at the current node, not backed up through the tree as Bellman-consistent values.

### 5) Convert to continuation weights

Checked continuations are mapped onto a comparable scale at the current node.
The exact normalization differs between the two study modes, but in both cases the result is a continuation weight that feeds the cumulative importance rule.

### 6) Apply cumulative-importance pruning

Child importance is computed by multiplying:

- the parent node importance,
- the continuation weight,
- and a global per-ply decay.
Only continuations whose resulting child importance remains above the configured floor are kept for further expansion.

### 7) Expand by frontier layers

Expansion proceeds one layer at a time rather than taking one line to full depth before moving on.
This keeps partial outputs useful and naturally preserves more breadth near the top of the study tree.

---

## Pruning Philosophy

The most important design choice is not the exact evaluation metric.
It is how study budget is withheld.

The pruning philosophy is:

- do not spend child-evaluation budget on too many candidates,
- do not keep continuations whose cumulative importance has already collapsed,
- do not give tenuki the same downstream budget as sustained local joseki play,
- and do not spend separate budget on continuations that are provably dead or redundant.

Example: in the acute-corner case with red `b4`, blue `c2`, and `a1-a4,b1-b3` all empty, moves in that dead region should not consume separate candidate budget.

These decisions are intentional study-policy choices.
They are not claims that every omitted move is globally bad or that every retained move is uniquely correct.

---

## Output

Both workflows produce:

- a retained continuation tree,
- node-level importance values,
- checked-candidate metadata explaining which continuations were considered,
- and a frontier-layered database that remains usable even before expansion is complete.

The opening output is best thought of as a compact whole-board opening corpus.
The joseki output is best thought of as a compact local-corner line tree.

---

## Non-Goals

This workflow is not intended to:

- solve openings,
- prove exact equivalence classes in general,
- replace swap analysis with continuation-tree search,
- enumerate every plausible legal continuation,
- or treat raw-NN child evaluations as game-theoretic backed-up values.
