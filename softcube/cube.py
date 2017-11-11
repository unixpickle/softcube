"""
Manipulations for Rubik's cubes.
"""

import tensorflow as tf

# Taken from https://github.com/unixpickle/pll-order.
# pylint: disable=C0301
RAW_PERMUTATIONS = {
    'U': [6, 3, 0, 7, 4, 1, 8, 5, 2, 9, 10, 11, 12, 13, 14, 15, 16, 17, 36, 37, 38, 21, 22, 23, 24, 25, 26, 45, 46, 47, 30, 31, 32, 33, 34, 35, 27, 28, 29, 39, 40, 41, 42, 43, 44, 18, 19, 20, 48, 49, 50, 51, 52, 53],
    'D': [0, 1, 2, 3, 4, 5, 6, 7, 8, 15, 12, 9, 16, 13, 10, 17, 14, 11, 18, 19, 20, 21, 22, 23, 51, 52, 53, 27, 28, 29, 30, 31, 32, 42, 43, 44, 36, 37, 38, 39, 40, 41, 24, 25, 26, 45, 46, 47, 48, 49, 50, 33, 34, 35],
    'F': [0, 1, 2, 3, 4, 5, 53, 50, 47, 42, 39, 36, 12, 13, 14, 15, 16, 17, 24, 21, 18, 25, 22, 19, 26, 23, 20, 27, 28, 29, 30, 31, 32, 33, 34, 35, 6, 37, 38, 7, 40, 41, 8, 43, 44, 45, 46, 9, 48, 49, 10, 51, 52, 11],
    'B': [38, 41, 44, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 45, 48, 51, 18, 19, 20, 21, 22, 23, 24, 25, 26, 33, 30, 27, 34, 31, 28, 35, 32, 29, 36, 37, 17, 39, 40, 16, 42, 43, 15, 2, 46, 47, 1, 49, 50, 0, 52, 53],
    'R': [0, 1, 20, 3, 4, 23, 6, 7, 26, 9, 10, 33, 12, 13, 30, 15, 16, 27, 18, 19, 11, 21, 22, 14, 24, 25, 17, 8, 28, 29, 5, 31, 32, 2, 34, 35, 42, 39, 36, 43, 40, 37, 44, 41, 38, 45, 46, 47, 48, 49, 50, 51, 52, 53],
    'L': [35, 1, 2, 32, 4, 5, 29, 7, 8, 18, 10, 11, 21, 13, 14, 24, 16, 17, 0, 19, 20, 3, 22, 23, 6, 25, 26, 27, 28, 15, 30, 31, 12, 33, 34, 9, 36, 37, 38, 39, 40, 41, 42, 43, 44, 51, 48, 45, 52, 49, 46, 53, 50, 47]
}

def identity_cube(dtype=tf.float64):
    """
    Create a cube Tensor representing the solved state.
    """
    res = []
    for i in range(6 * 9):
        one_hot = [0] * 6
        one_hot[i // 9] = 1
        res.extend(one_hot)
    return tf.constant(res, dtype=dtype)

def apply_move(move, cube):
    """
    Apply a move to the cube Tensor.

    Args:
      move: a string representing the move in WCA
        notation. Only face turns are supported.
      cube: a 324-component 1-D Tensor representing a cube
        state or a distribution of cube states.

    Returns:
      A new cube Tensor.
    """
    if len(move) not in [1, 2] or move[0] not in RAW_PERMUTATIONS:
        raise ValueError('invalid move: ' + move)
    move_count = 1
    if len(move) == 2:
        if move[1] == "'":
            move_count = 3
        elif move[1] == '2':
            move_count = 2
        else:
            raise ValueError('invalid move: ' + move)
    raw_perm = RAW_PERMUTATIONS[move[0]]
    one_hot_perm = []
    for src_idx in raw_perm:
        for i in range(6):
            one_hot_perm.append(raw_perm[6*src_idx + i])

    perm = tf.constant(one_hot_perm)
    res = cube
    for i in range(move_count):
        res = tf.gather(res, perm)
    return res
