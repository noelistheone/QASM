"""Measurement-conditioned refinement through the checked scan operator.

This module contains no file or label loading. Its caller supplies only the
permitted visible measurements. A positive pixel preconditioner proposes a
direction, and the exact bounded intensity-loss line solve chooses its size.
"""
import math
import torch

from src.ptycho_intensity_steps import (
    choose_quartic_step, differentiable_quartic_step,
    intensity_coefficients, positive_diagonal_direction,
)


class VisibleIntensityProblem:
    def __init__(self, operator, ids, target, weights=None, batch=256,
                 coverage_floor=1e-4):
        self.operator = operator
        self.ids = torch.as_tensor(ids, device=operator.probe.device, dtype=torch.long)
        if (self.ids.ndim != 1 or not len(self.ids)
                or len(self.ids.unique()) != len(self.ids)
                or torch.any(self.ids < 0) or torch.any(self.ids >= len(operator.rounded))):
            raise ValueError('Unique, nonempty, in-range measurement IDs required')
        if not isinstance(batch, int) or batch < 1:
            raise ValueError('Positive integer batch size required')
        if not math.isfinite(coverage_floor) or coverage_floor <= 0:
            raise ValueError('Positive finite coverage floor required')
        expected = (len(self.ids), operator.size, operator.size)
        if (tuple(target.shape) != expected or target.is_complex()
                or not torch.isfinite(target).all() or torch.any(target < 0)
                or target.device != operator.probe.device):
            raise ValueError('Finite nonnegative intensities on the operator device required')
        self.target = target
        self.weights = torch.ones_like(target) if weights is None else torch.broadcast_to(weights, expected)
        if (self.weights.device != target.device or not torch.isfinite(self.weights).all()
                or torch.any(self.weights < 0) or self.weights.sum() <= 0):
            raise ValueError('Nonnegative weights with a positive sum required')
        self.normalizer = self.weights.sum()
        self.batch = batch
        self.coverage = operator.coverage(self.ids)
        positive = self.coverage[self.coverage > 0]
        if not len(positive):
            raise ValueError('No illuminated pixels')
        self.denominator = (self.coverage + coverage_floor*positive.mean())/self.normalizer
        self.coverage_sum = self.coverage.sum()
        self.chunks = []
        for start in range(0, len(self.ids), batch):
            stop = min(start+batch, len(self.ids))
            weight = self.weights[start:stop]
            mass = weight.sum()
            if mass > 0:
                self.chunks.append((self.ids[start:stop], self.target[start:stop], weight, mass))

    def evaluate(self, columns):
        """Visible loss and real complex gradient; retain fields for the line solve.

        For dL=Re<G,dB>, G=2 A* [w (sum|AB|²-y) AB] / sum(w).
        These operations retain the graph, including second derivatives needed
        when differentiating a finite sequence of updates.
        """
        if columns.device != self.target.device:
            raise ValueError('Columns and observations must share a device')
        gradient = torch.zeros_like(columns)
        loss = self.target.new_zeros(())
        cached = []
        for ids, target, weight, mass in self.chunks:
            fields = self.operator.fields(columns, ids)
            residual = fields.abs().square().sum(1)-target
            loss = loss + .5*(weight*residual.square()).sum()/self.normalizer
            gradient = gradient + 2*self.operator.adjoint(
                (weight*residual)[:, None]*fields, ids)/self.normalizer
            cached.append((ids, target, weight, mass, fields))
        return dict(loss=loss, gradient=gradient, fields=cached)

    def loss(self, columns):
        """Forward-only diagnostic; additional work outside an update step."""
        total = self.target.new_zeros(())
        for ids, target, weight, mass in self.chunks:
            fields = self.operator.fields(columns, ids)
            residual = fields.abs().square().sum(1)-target
            total = total + .5*(weight*residual.square()).sum()/self.normalizer
        return total

    def direction(self, gradient, log_weights, normalize_weights=True):
        """Common-mode positive spatial preconditioning, with no learned scalar.

        Coverage-weighted normalization removes the redundant uniform weight
        scale: the bounded line solve remains the only scalar step-size rule.
        """
        if tuple(log_weights.shape) != self.operator.shape:
            raise ValueError('One shared spatial weight map on the object canvas required')
        denominator = self.denominator
        if normalize_weights:
            positive = torch.nn.functional.softplus(log_weights)+1e-3
            mean = (self.coverage*positive).sum()/self.coverage_sum
            denominator = denominator*mean
        return positive_diagonal_direction(gradient, log_weights, denominator)

    def coefficients(self, evaluated, direction):
        coefficients = self.target.new_zeros(5)
        for ids, target, weight, mass, fields in evaluated['fields']:
            propagated = self.operator.fields(direction, ids)
            coefficients = coefficients + (mass/self.normalizer)*intensity_coefficients(
                fields.permute(1, 0, 2, 3), propagated.permute(1, 0, 2, 3), target, weight)
        return coefficients

    def step(self, columns, log_weights, upper=1., differentiable=True,
             normalize_weights=True, evaluated=None):
        evaluated = self.evaluate(columns) if evaluated is None else evaluated
        direction = self.direction(evaluated['gradient'], log_weights, normalize_weights)
        coefficients = self.coefficients(evaluated, direction)
        certificate = choose_quartic_step(coefficients.detach().double().cpu().numpy(), upper)
        step = (differentiable_quartic_step(coefficients, upper) if differentiable
                else coefficients.new_tensor(certificate['step']))
        return columns+step*direction, dict(
            coefficients=coefficients, step=step, certificate=certificate,
            visible_loss_before=evaluated['loss'], gradient=evaluated['gradient'],
            direction=direction,
            operator_work=dict(forward_batches=2*len(self.chunks),
                               adjoint_batches=len(self.chunks),
                               propagated_frames=2*len(self.ids),
                               backprojected_frames=len(self.ids)))
