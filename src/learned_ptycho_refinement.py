"""Finite internal updates with a shared positive learned preconditioner."""
import torch
from torch import nn

from src.global_ptycho_response import invariant_features


class SpatialPreconditioner(nn.Module):
    def __init__(self, kind='spatial', width=16):
        super().__init__()
        if kind not in ('spatial', 'pointwise'):
            raise ValueError('Expected spatial or pointwise preconditioner')
        kernel = 3 if kind == 'spatial' else 1
        self.kind = kind
        self.net = nn.Sequential(
            nn.Conv2d(9, width, kernel, padding=kernel//2), nn.SiLU(),
            nn.Conv2d(width, width, kernel, padding=kernel//2), nn.SiLU(),
            nn.Conv2d(width, 1, 1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, columns, problem, evaluated, fraction):
        if len(columns) != 1:
            raise ValueError('This pilot learns coherent single-field updates only')
        # invariant_features expects an unnormalized summed intensity gradient.
        # The factor of two is fixed and common to learned controls.
        summed = evaluated['gradient'][0]*problem.normalizer/2
        features = invariant_features(columns[0], problem.coverage, summed, fraction)
        raw = self.net(features[None])[0, 0]
        # Finite dynamic range; positivity is supplied by the problem's
        # softplus map, and coverage normalization removes its uniform scale.
        return 6*torch.tanh(raw/6)


def refine_columns(initial, problem, fraction, steps=3, model=None,
                   differentiable=True, upper=1.):
    if steps < 0:
        raise ValueError('Nonnegative update count required')
    columns = initial
    records = []
    expected_loss = None
    for index in range(steps):
        evaluated = problem.evaluate(columns)
        actual_loss = float(evaluated['loss'].detach())
        if expected_loss is not None and abs(actual_loss-expected_loss) > 5e-7+2e-5*abs(expected_loss):
            raise FloatingPointError('Updated physical loss disagrees with its intensity polynomial')
        logits = (torch.zeros_like(columns[0].real) if model is None
                  else model(columns, problem, evaluated, fraction))
        columns, detail = problem.step(columns, logits, upper=upper,
            differentiable=differentiable, evaluated=evaluated)
        expected_loss = float(sum(detail['coefficients'][i]*detail['step']**i for i in range(5)).detach())
        if expected_loss > actual_loss+5e-7+2e-5*abs(actual_loss):
            raise FloatingPointError('Selected update increased the visible intensity objective')
        records.append(dict(index=index, certificate=detail['certificate'],
            visible_loss_before=actual_loss, visible_loss_after_polynomial=expected_loss,
            step=float(detail['step'].detach()), operator_work=detail['operator_work']))
    return columns, records
