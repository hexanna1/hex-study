# Match Database Spec

## Purpose

Define the value convention used when the match database reads `hexata` `batch` and `match` JSON.

The key rule is simple: graph point `N` means the value of the position after move `N` has been played.

The awkward part is that `batch` and `match` expose different evidence for that same value.

## Records and Cursor

A searched ply is a pre-move search. It says what the engine saw from the current position and which move was actually played.

The website cursor is position-based:

- record ply `N` is the search before move `N`,
- cursor step `N` is the board after move `N`.

At a cursor position, the website uses the searched record for that same position. That record supplies the candidates and marks the actual next move.

`pass` and `swap` are played actions. `final` is analysis after the last played move, not another move. A resignation result is outcome metadata; if shown as a terminal cursor step, it has the same board as the previous step.

## Graph Semantics

In `batch`, the played move may not be analyzed as a useful candidate at all. It can be missing from the retained candidate list, or present with too few visits to trust. But the analyzed positions come from the same engine, so the website uses the next position's analysis as the value after the played move. This relies on the Bellman-consistency assumption: the value of a move's child position is the value of that move.

For the last batch move, the next position is supplied by `final`.

In `match`, the played move is analyzed by the mover, so the website uses the mover's own candidate row for `played`. It cannot use the next ply in the batch style, because the next search belongs to the other engine and would measure the opponent's view of the reply.

That is why `final` can supply a batch graph point but is not graphed for match games. Terminal resignation steps also do not add graph points.
