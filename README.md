# softcube

**Disclaimer:** This is an experiment, and it probably won't work!

This is an attempt to solve (or partially solve) the Rubik's cube using gradient descent. Essentially, a distribution over solutions is represented with a softmax for each move, and the final cube is represented as a mean vector over all those cubes. The gradient descent objective is to pull the final cube mean towards the solved state.

# The idea

Moves on a Rubik's cube can be seen as permutation matrices, and states of a Rubik's cube can be represented as arrays of one-hot vectors (one per sticker). Thus, we can represent an "expected" Rubik's cube as an expectation over cube vectors. We can apply a move to an expected cube by permuting the expected vector. We can then get a new expectation for the next move by averaging these vectors for each move.

# Results

Here, I shall report back on how well this works. My prediction: not at all.
