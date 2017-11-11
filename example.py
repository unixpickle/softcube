"""
Example of solving a simple cube.
"""

from softcube import solve
import tensorflow as tf

def main():
    """
    Compute a solution to a two-move scramble.
    """
    objective, solutions = solve(['R', 'U'], 10, solution_moves=2)
    mean_obj = tf.reduce_mean(objective)
    minimize = tf.train.AdamOptimizer(learning_rate=1e-2).minimize(mean_obj)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for i in range(1000):
            _, obj = sess.run((minimize, mean_obj))
            if i % 10 == 0:
                print('step %d: loss=%f' % (i, obj))
        solutions = zip(*sess.run((solutions, objective)))
        print('\n'.join([str(x) for x in sorted(solutions, key=lambda x: x[1])]))

if __name__ == '__main__':
    main()
