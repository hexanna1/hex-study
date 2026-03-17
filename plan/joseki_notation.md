## Corner Joseki Notation

A joseki line is an ordered sequence of corner-family plies. Coordinates are corner-relative, color-agnostic, and quotient out the specific physical board corner.

Two rigid forms are supported:

### 1) Track block

`A[PLIES]` or `O[PLIES]`

* `A` means acute-corner track.
* `O` means obtuse-corner track.
* `PLIES` is a `:`-separated list of entries.
* Each entry is either a move `x,y` or empty.

Examples:
`A[5,4]`
`A[5,4:3,7:]`
`O[4,4:2,4:2,3]`

### 2) Full line

A full joseki line is a concatenation of one or more track blocks.

Examples:
`A[5,4]`
`A[5,4]O[4,4]A[3,7]`
`O[4,4]A[]O[2,4]`

---

## Semantics

* A line is rooted at its first non-empty ply.
* Ply ownership is determined by global line order.
* Empty entry means no move on that track at that global ply.
* Coordinates in `A[...]` are relative to the acute track in the initiator's canonical frame.
* Coordinates in `O[...]` are relative to the obtuse track in the initiator's canonical frame.
* Coordinates do not change with side to move.
* Coordinates do not change with actual color assignment.

So `A[5,4:3,7:]` refers to one initiator-relative acute track sequence, independent of which actual color made the first move and independent of which acute corner realized it on the board.

---

## Merge Sugar

Consecutive blocks from the same track may be merged by concatenating their entries with `:`.

Examples:

* `A[1,1]A[2,2]` is equivalent to `A[1,1:2,2]`
* `A[1,1]A[]A[2,2]` is equivalent to `A[1,1::2,2]`

This sugar is only for consecutive blocks on the same track. Different tracks remain explicit in the global order.

---

## Coordinate Meaning

A move token `x,y` is a family-local corner coordinate pair.

It is not:

* a board coordinate such as `d5`,
* the repo's existing axial local-pattern coordinate system.

The exact normalization from concrete board cells to initiator-relative family-local coordinates is a separate specification.

---

## Grammar

### Lexical

* `int := digit+`
* `move := int "," int`
* `entry := move | ""`

### Blocks and lines

* `acute := "A[" entry ( ":" entry )* "]"`
* `obtuse := "O[" entry ( ":" entry )* "]"`
* `block := acute | obtuse`
* `line := block+`

Constraints:

* Coordinates are positive integers.
* The first non-empty entry in a line fixes the initiator.
* Canonical output has no spaces.
* Empty entries are allowed.

---

## Relationship to Pattern Notation

Joseki notation records an ordered line. Pattern notation records a current position. Position snapshots can be derived from any joseki prefix when needed.
