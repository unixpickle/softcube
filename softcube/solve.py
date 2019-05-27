"""
SGD-based cube solving.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .cube import ALL_MOVES, identity_cubes, move_matrix

EPSILON = 1e-4


def solution_loss(cubes, epsilon=EPSILON):
    """
    Compute a loss for each cube in a batch.

    The loss is zero for solved cubes, and positive for
    unsolved cubes.
    """
    log_probs = torch.log((cubes + epsilon) / (1 + epsilon))
    return -torch.sum(log_probs * identity_cubes(cubes.shape[0]), dim=-1)


class MoveCombiner:
    """
    A cache-based routine for linearly combining move
    matrices.
    """

    def __init__(self):
        self.matrix_size = move_matrix(ALL_MOVES[0]).shape[0]
        self.matrices = torch.stack([move_matrix(m).view(-1) for m in ALL_MOVES])

    def __call__(self, probs):
        combined = torch.matmul(probs, self.matrices)
        return combined.view(-1, self.matrix_size, self.matrix_size)


class SoftmaxSolver:
    """
    A solver that finds solutions by learning logits.
    """

    def __init__(self, num_moves, batch_size):
        self.num_moves = num_moves
        self.batch_size = batch_size
        self.combiner = MoveCombiner()
        self.move_logits = []
        for _ in range(num_moves):
            logits = nn.Parameter(torch.randn([batch_size, len(ALL_MOVES)]))
            self.move_logits.append(logits)

    def parameters(self):
        return self.move_logits

    def apply_solutions(self, cubes):
        for logits in self.move_logits:
            probs = F.softmax(logits, dim=-1)
            matrices = self.combiner(probs)
            cubes = torch.matmul(cubes[:, None, :], matrices)[:, 0]
        return cubes

    def losses(self, cubes):
        return solution_loss(self.apply_solutions(cubes))

    def strings(self):
        res = []
        for i in range(self.batch_size):
            moves = []
            for tensor in self.move_logits:
                idx = np.argmax(tensor[i].detach().cpu().numpy())
                moves.append(ALL_MOVES[idx])
            res.append(' '.join(moves))
        return res
