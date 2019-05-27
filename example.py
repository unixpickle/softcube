"""
Example of solving a simple cube.
"""

import torch
import torch.optim as optim

from softcube.cube import algorithm_cubes
from softcube.solve import NNSolver


def main():
    """
    Compute a solution to a simple scramble.
    """
    solver = NNSolver(num_moves=3, batch_size=10)
    opt = optim.Adam(solver.parameters(), lr=1e-3)
    start = algorithm_cubes('R U R\'', solver.batch_size)
    for i in range(100):
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
