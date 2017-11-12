"""
SGD-based cube solving.
"""

import tensorflow as tf

from .cube import ALL_MOVES, identity_cubes, apply_move

EPSILON = 1e-4

def solve(scramble, batch_size, solution_moves=20):
    """
    Setup an objective that can be optimized to find a
    solution to a cube.

    Args:
      scramble: a sequence of moves representing the state
        to be solved.
      batch_size: the number of solutions to optimize in
        parallel.

    Returns:
      A tuple containing:
        objective: a 1-D Tensor of losses.
        moves: a 1-D Tensor of solution strings.
    """
    identities = identity_cubes(batch_size)
    cube = identities
    for move in scramble:
        cube = apply_move(move, cube)
    result_moves = []
    for _ in range(solution_moves):
        cube, move_str = apply_distribution(cube)
        result_moves.append(move_str)
    move_strs = tf.string_join(result_moves, separator=' ')
    log_probs = tf.log((cube + EPSILON) / (1 + EPSILON))
    return -tf.reduce_sum(log_probs * identities, axis=-1), move_strs

def apply_distribution(cubes, initializer=tf.truncated_normal_initializer()):
    """
    Apply a learned move distribution to each cube.
    The distributions are separate for each cube, meaning
    that the batch size must be known beforehand.

    Args:
      cubes: a batch of cube Tensors.
      initializer: the initializer for the softmax
        parameters.

    Returns:
      A tuple containing:
        cubes: a batch of resultant cube distributions.
        moves: a batch of strings representing the current
          maximum likelihood moves.

    Creates a Variable in the current scope.
    """
    dist_params = tf.Variable(initializer(shape=(int(cubes.get_shape()[0]), len(ALL_MOVES)),
                                          dtype=cubes.dtype))
    move_probs = tf.nn.softmax(dist_params)
    results = [move_probs[:, i:i+1] * apply_move(move, cubes)
               for i, move in enumerate(ALL_MOVES)]
    max_indices = tf.argmax(dist_params, axis=-1)
    move_names = tf.gather(tf.constant(ALL_MOVES), max_indices)
    return tf.reduce_sum(tf.stack(results), axis=0), move_names
