"""Wasserstein trust-region method for empirical measures (NumPy).

A field on the particles is an array S of shape (N, d), and the tangent space
carries the inner product <S, T> = (1/N) sum_i S_i . T_i of L^2(rho_N).
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable

import numpy as np

Array = np.ndarray
HessianAction = Callable[[Array], Array]
VARIANTS = ('full', 'cauchy', 'no_growth')


def particle_inner(u: Array, v: Array) -> float:
    return float(np.mean(np.sum(u * v, axis=1)))


def particle_norm(u: Array) -> float:
    return float(np.sqrt(max(0.0, particle_inner(u, u))))


def boundary_intersection(z: Array, p: Array, radius: float) -> float:
    """Positive tau with |z + tau p| = radius."""
    a = particle_inner(p, p)
    if a <= 0.0:
        return 0.0
    b = 2.0 * particle_inner(z, p)
    c = particle_inner(z, z) - radius * radius
    discriminant = max(0.0, b * b - 4.0 * a * c)
    return float(max(0.0, (-b + np.sqrt(discriminant)) / (2.0 * a)))


def steihaug_toint(gradient: Array, radius: float, hessian_action: HessianAction,
                   maximum_iterations: int, relative_tolerance: float) -> tuple[Array, dict]:
    """Truncated conjugate gradients (Steihaug--Toint) for the model subproblem.

    Exits at nonpositive curvature, at the trust-region boundary, when the
    residual falls below relative_tolerance times its initial value, or after
    maximum_iterations iterations.
    """
    step = np.zeros_like(gradient)
    residual = gradient.copy()
    direction = -residual
    residual_squared = particle_inner(residual, residual)
    initial_residual = np.sqrt(max(0.0, residual_squared))
    if initial_residual == 0.0:
        return step, {'exit_mode': 'zero_gradient', 'iterations': 0}

    for iteration in range(maximum_iterations):
        hessian_direction = hessian_action(direction)
        curvature = particle_inner(direction, hessian_direction)
        if curvature <= 0.0:
            tau = boundary_intersection(step, direction, radius)
            return step + tau * direction, {'exit_mode': 'negative_curvature', 'iterations': iteration + 1}
        alpha = residual_squared / curvature
        proposed_step = step + alpha * direction
        if particle_norm(proposed_step) >= radius:
            tau = boundary_intersection(step, direction, radius)
            return step + tau * direction, {'exit_mode': 'boundary', 'iterations': iteration + 1}
        step = proposed_step
        residual = residual + alpha * hessian_direction
        new_residual_squared = particle_inner(residual, residual)
        if np.sqrt(max(0.0, new_residual_squared)) <= relative_tolerance * max(initial_residual, 1.0e-30):
            return step, {'exit_mode': 'residual', 'iterations': iteration + 1}
        beta = new_residual_squared / max(residual_squared, 1.0e-300)
        direction = -residual + beta * direction
        residual_squared = new_residual_squared

    return step, {'exit_mode': 'iteration_cap', 'iterations': maximum_iterations}


@dataclass(frozen=True)
class TrustRegionConfig:
    initial_radius: float
    maximum_radius: float
    acceptance_threshold: float = 0.1     # eta_1
    expansion_threshold: float = 0.75     # eta_2
    shrink_factor: float = 0.25           # gamma_1
    growth_factor: float = 2.0            # gamma_2
    boundary_fraction: float = 0.8        # theta
    cg_maximum_iterations: int = 30       # J_max
    cg_relative_tolerance: float = 1.0e-6  # epsilon_cg


def run_trust_region(x_initial: Array, energy, config: TrustRegionConfig, maximum_iterations: int,
                     gradient_tolerance: float, variant: str = 'full',
                     diagnostic: Callable[[Array], float] | None = None) -> dict:
    """Run WTR on the particle energy ``energy`` from ``x_initial``.

    ``energy`` provides value(x), gradient(x) and linearize(x), which returns
    the gradient and the model operator S -> B S at x.  Variants:

    * ``full``: the method of the paper;
    * ``cauchy``: B = 0, so every step is the boundary step -Delta g/|g|;
    * ``no_growth``: the full method with radius expansion disabled.

    ``diagnostic`` is evaluated at every accepted iterate (here, the holdout
    loss) and is not used by the algorithm.
    """
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant: {variant}')

    def linearize(x):
        if variant == 'cauchy':
            return energy.gradient(x), None
        return energy.linearize(x)

    x = np.asarray(x_initial, dtype=float).copy()
    radius = float(config.initial_radius)
    objective = float(energy.value(x))
    gradient, hessian_action = linearize(x)
    objective_history = [objective]
    gradient_history = [particle_norm(gradient)]
    diagnostic_history = [float(diagnostic(x))] if diagnostic is not None else []
    accepted_history, exit_modes = [], []
    termination = 'iteration_cap'

    for _ in range(maximum_iterations):
        gradient_norm = particle_norm(gradient)
        if gradient_norm <= gradient_tolerance:
            termination = 'gradient_tolerance'
            break

        if variant == 'cauchy':
            step = -(radius / max(gradient_norm, 1.0e-300)) * gradient
            predicted_reduction = -particle_inner(gradient, step)
            exit_mode = 'cauchy_boundary'
        else:
            step, info = steihaug_toint(gradient, radius, hessian_action,
                                        config.cg_maximum_iterations, config.cg_relative_tolerance)
            predicted_reduction = -(particle_inner(gradient, step)
                                    + 0.5 * particle_inner(step, hessian_action(step)))
            exit_mode = info['exit_mode']

        if not np.isfinite(predicted_reduction) or predicted_reduction <= 1.0e-16:
            termination = 'nonpositive_model_reduction'
            break

        trial = x + step
        trial_objective = float(energy.value(trial))
        ratio = (objective - trial_objective) / predicted_reduction
        accepted = bool(np.isfinite(ratio) and ratio >= config.acceptance_threshold)

        if accepted:
            x = trial
            objective = trial_objective
            gradient, hessian_action = linearize(x)
            if (variant != 'no_growth' and ratio >= config.expansion_threshold
                    and particle_norm(step) >= config.boundary_fraction * radius):
                radius = min(config.growth_factor * radius, config.maximum_radius)
        else:
            radius *= config.shrink_factor

        objective_history.append(objective)
        gradient_history.append(particle_norm(gradient))
        if diagnostic is not None:
            diagnostic_history.append(float(diagnostic(x)) if accepted else diagnostic_history[-1])
        accepted_history.append(accepted)
        exit_modes.append(exit_mode)

    if np.max(np.diff(objective_history), initial=0.0) > 5.0e-11 * max(1.0, abs(objective_history[0])):
        raise RuntimeError('accepted objective values are not monotone')

    return {
        'x_final': x,
        'iterations': len(accepted_history),
        'accepted_iterations': int(np.sum(accepted_history)),
        'termination': termination,
        'objective_history': objective_history,
        'gradient_norm_history': gradient_history,
        'diagnostic_history': diagnostic_history,
        'accepted_history': accepted_history,
        'exit_mode_counts': dict(sorted(Counter(exit_modes).items())),
    }
