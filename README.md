# softcube

**Disclaimer:** This is an experiment, and it probably won't work!

This is an attempt to solve (or partially solve) the Rubik's cube using gradient descent. Essentially, a distribution over solutions is represented with a softmax for each move, and the final cube is represented as a mean vector over all those cubes. The gradient descent objective is to pull the final cube mean towards the solved state.

# The idea

Moves on a Rubik's cube can be seen as permutation matrices, and states of a Rubik's cube can be represented as arrays of one-hot vectors (one per sticker). Thus, we can represent an "expected" Rubik's cube as an expectation over cube vectors. We can apply a move to an expected cube by permuting the expected vector. We can then get a new expectation for the next move by averaging these vectors for each move.

# Results

The program can solve two-move scrambles like "R U" on pretty much every attempt. The program can also solve four move scrambles like "R U F B", but a bit less than half of the starting initializations result in the correct solution.

I ran 100 random inits on the sune ("R U R' U R U2 R'"). None of the inits found the solution. Here are the top 10 results:

```python
(b"B B B2 U U B B'", 11.907935871759161)
(b"U2 U2 U B2 B' B' U", 11.915506755198425)
(b"L' L2 L' U U U' U", 11.917694614822491)
(b"U L2 L' L L L U", 11.924781099237155)
(b"U' D2 D' U D' U U", 11.927204057829663)
(b"U' D2 D2 U2 U' U U", 11.934922604095824)
(b"U R' R2 R' U U' U", 11.935606596792471)
(b"D2 D2 U2 D2 D2 U' U", 11.93965739975992)
(b"D' U2 U' D U' U U", 11.944093857038414)
(b"U U2 U2 U' U' U2 U", 11.95040498751858)
```
