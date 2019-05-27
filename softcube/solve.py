"""
SGD-based cube solving.
"""

from abc import ABC, abstractmethod

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

    def to(self, target):
        self.matrices = self.matrices.to(target)

    def __call__(self, probs):
        combined = torch.matmul(probs, self.matrices)
        return combined.view(-1, self.matrix_size, self.matrix_size)


class Solver(ABC):
    def __init__(self, num_moves, batch_size):
        self.num_moves = num_moves
        self.batch_size = batch_size
        self.combiner = MoveCombiner()

    @abstractmethod
    def parameters(self):
        pass

    @abstractmethod
    def probabilities(self):
        """
        Get a batch of move probability batches, one per
        move.
        """
        pass

    def to(self, target):
        for p in self.parameters:
            p.to(target)
        self.combiner.to(target)

    def apply_solutions(self, cubes):
        for probs in self.probabilities():
            matrices = self.combiner(probs)
            cubes = torch.matmul(cubes[:, None, :], matrices)[:, 0]
        return cubes

    def losses(self, cubes):
        return solution_loss(self.apply_solutions(cubes))

    def strings(self):
        res = []
        probs = [x.detach().cpu().numpy() for x in self.probabilities()]
        for i in range(self.batch_size):
            moves = []
            for arr in probs:
                idx = np.argmax(arr[i])
                moves.append(ALL_MOVES[idx])
            res.append(' '.join(moves))
        return res


class SoftmaxSolver(Solver):
    """
    A solver that finds solutions by learning logits.
    """

    def __init__(self, num_moves, batch_size):
        super().__init__(num_moves, batch_size)
        self.move_logits = []
        for _ in range(num_moves):
            logits = nn.Parameter(torch.randn([batch_size, len(ALL_MOVES)]))
            self.move_logits.append(logits)

    def parameters(self):
        return self.move_logits

    def probabilities(self):
        return [F.softmax(logits, dim=-1) for logits in self.move_logits]


class NNSolver(Solver):
    """
    A solver that finds solutions by learning neural
    network parameters.
    """

    def __init__(self, num_moves, batch_size):
        super().__init__(num_moves, batch_size)
        self.base = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
        )
        self.inputs = nn.Parameter(torch.randn([batch_size, 256]))
        self.output_layers = []
        for _ in range(num_moves):
            self.output_layers.append(nn.Sequential(
                nn.Linear(256, 256),
                nn.BatchNorm1d(256),
                nn.ReLU(),
                nn.Linear(256, len(ALL_MOVES)),
            ))

    def parameters(self):
        return list(self.base.parameters()) + [self.inputs] + [x for l in self.output_layers
                                                               for x in l.parameters()]

    def probabilities(self):
        features = self.base(self.inputs)
        return [F.softmax(layer(features), dim=-1) for layer in self.output_layers]
