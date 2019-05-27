"""
Example of solving a simple cube.
"""

import torch
import torch.optim as optim

from softcube.cube import algorithm_cubes
from softcube.solve import SoftmaxSolver


def main():
    """
    Compute a solution to a two-move scramble.
    """
    solver = SoftmaxSolver(num_moves=2, batch_size=10)
    opt = optim.Adam(solver.parameters(), lr=1e-2)
    start = algorithm_cubes('R U', solver.batch_size)
    for i in range(1000):
        obj = torch.mean(solver.losses(start))
        opt.zero_grad()
        obj.backward()
        opt.step()
        if i % 10 == 0:
            print('step %d: loss=%f' % (i, obj.item()))
    solutions = zip(solver.strings(), solver.losses(start).detach().cpu().numpy())
    print('\n'.join([str(x) for x in sorted(solutions, key=lambda x: x[1])]))


if __name__ == '__main__':
    main()
