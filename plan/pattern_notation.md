## Pattern Notation (interior-only)

A pattern is two disjoint finite sets of axial coordinates `(q,r)` on the infinite hex grid. Coordinates are written as `q,r`. Points inside a block are separated by `:`.

Two rigid forms are supported:

### 1) Labeled (side-to-move relative)

`+[PLUS]-[MINUS]`

* `PLUS` is the side-to-move stones, `MINUS` is the opponent stones.
* Either list may be empty: `+[]-[...]` or `+[...]-[]`.
* The notation itself does not encode which player is to move; that must be supplied externally.

Example:
`+[0,0:0,1:1,-1]-[1,0:2,0]`

### 2) Unlabeled (swap-invariant by syntax)

`[A][B]`

* Two unlabeled color-classes; swapping the two blocks does not change the meaning.
* Either list may be empty: `[A][]`.

Example:
`[0,0:0,1][1,0:2,0]`

---

## Grammar

### Lexical

* `int  := "-"? digit+`
* `point := int "," int`
* `points := point ( ":" point )*`  (may be empty)
* `block := "[" points? "]"`

### Patterns

* `labeled := "+" block "-" block`
* `unlabeled := block block`
* `pattern := labeled | unlabeled`

Constraints:

* No spaces in canonical output (you may allow them in input if you want, but then strip them before parsing).
* Duplicate points within the same block are invalid.
* Cross-block overlap is invalid (a point cannot appear in both color-classes).
