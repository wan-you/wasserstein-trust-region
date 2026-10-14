"""Empirical maximum mean discrepancy (MMD) energies and benchmark targets.

For a target sample Y_1, ..., Y_M and particles X_1, ..., X_N,

    (1/2) MMD_k^2 = -(1/NM) sum_{i,m} k(X_i - Y_m) + (1/2N^2) sum_{i,j} k(X_i - X_j) + C,

where k is the equal-weight mixture of Gaussian kernels with the bandwidths
below and C does not depend on X.  Gradients and model operators use the
inner product of L^2(rho_N).
"""
from __future__ import annotations

import numpy as np

from .trust_region import Array

BANDWIDTHS = (0.22, 0.45, 0.90, 1.80)
TARGETS = ('banana', 'two_moons', 'ring')


def sample_target(name: str, count: int, seed: int) -> Array:
    """Draw from the banana, two-moons or ring distribution."""
    rng = np.random.default_rng(seed)
    if name == 'banana':
        first = rng.standard_normal(count)
        second = rng.standard_normal(count)
        return np.column_stack((0.90 * first, 0.30 * second + 0.34 * (first * first - 1.0)))
    if name == 'two_moons':
        component = rng.integers(0, 2, size=count)
        angle = rng.uniform(0.0, np.pi, size=count)
        points = np.empty((count, 2), dtype=float)
        first = component == 0
        points[first, 0] = np.cos(angle[first]) - 0.50
        points[first, 1] = np.sin(angle[first]) - 0.25
        points[~first, 0] = 0.50 - np.cos(angle[~first])
        points[~first, 1] = -np.sin(angle[~first]) + 0.25
        points += 0.055 * rng.standard_normal(points.shape)
        return 1.15 * points
    if name == 'ring':
        angle = rng.uniform(0.0, 2.0 * np.pi, size=count)
        radius = 1.35 + 0.08 * rng.standard_normal(count)
        return np.column_stack((radius * np.cos(angle), radius * np.sin(angle)))
    raise ValueError(f'unknown target: {name}')


def pairwise_differences(x: Array, y: Array) -> tuple[Array, Array]:
    difference = x[:, None, :] - y[None, :, :]
    return difference, np.einsum('ijk,ijk->ij', difference, difference)


def gaussian_kernel_average(x: Array, y: Array, bandwidths=BANDWIDTHS) -> float:
    _, squared_distance = pairwise_differences(x, y)
    value = np.zeros_like(squared_distance)
    weight = 1.0 / len(bandwidths)
    for bandwidth in bandwidths:
        value += weight * np.exp(-squared_distance / (2.0 * bandwidth ** 2))
    return float(np.mean(value))


class HalfSquaredMMD:
    """(1/2) MMD_k^2 between the particles and a fixed target sample."""

    def __init__(self, target: Array, bandwidths=BANDWIDTHS):
        self.target = np.asarray(target, dtype=float).copy()
        self.bandwidths = tuple(float(value) for value in bandwidths)
        self.target_constant = gaussian_kernel_average(self.target, self.target, self.bandwidths)

    def value(self, x: Array) -> float:
        source_source = gaussian_kernel_average(x, x, self.bandwidths)
        source_target = gaussian_kernel_average(x, self.target, self.bandwidths)
        return float(0.5 * (source_source - 2.0 * source_target + self.target_constant))

    def gradient(self, x: Array) -> Array:
        return self._linearize(x, with_hessian=False)[0]

    def linearize(self, x: Array):
        return self._linearize(x, with_hessian=True)

    def hvp(self, x: Array, direction: Array) -> Array:
        return self.linearize(x)[1](direction)

    def _linearize(self, x: Array, with_hessian: bool):
        source_difference, source_squared = pairwise_differences(x, x)
        target_difference, target_squared = pairwise_differences(x, self.target)
        source_two = np.zeros_like(source_squared)
        target_two = np.zeros_like(target_squared)
        source_four = np.zeros_like(source_squared)
        target_four = np.zeros_like(target_squared)
        weight = 1.0 / len(self.bandwidths)
        for bandwidth in self.bandwidths:
            inverse_two = 1.0 / (bandwidth * bandwidth)
            source_kernel = weight * np.exp(-0.5 * inverse_two * source_squared)
            target_kernel = weight * np.exp(-0.5 * inverse_two * target_squared)
            source_two += source_kernel * inverse_two
            target_two += target_kernel * inverse_two
            if with_hessian:
                source_four += source_kernel * inverse_two ** 2
                target_four += target_kernel * inverse_two ** 2

        gradient = (-np.mean(source_two[:, :, None] * source_difference, axis=1)
                    + np.mean(target_two[:, :, None] * target_difference, axis=1))
        if not with_hessian:
            return gradient, None

        def hessian_action(direction: Array) -> Array:
            direction_difference = direction[:, None, :] - direction[None, :, :]
            source_projection = np.sum(source_difference * direction_difference, axis=2, keepdims=True)
            source_action = np.mean(
                source_four[:, :, None] * source_difference * source_projection
                - source_two[:, :, None] * direction_difference, axis=1)
            target_projection = np.sum(target_difference * direction[:, None, :], axis=2, keepdims=True)
            target_action = np.mean(
                target_four[:, :, None] * target_difference * target_projection
                - target_two[:, :, None] * direction[:, None, :], axis=1)
            return source_action - target_action

        return gradient, hessian_action


class MMDState:
    """Particle state with the kernel matrices formed by its evaluation."""

    __slots__ = ('x', 'value', 'source_kernels', 'target_kernels', 'gradient',
                 'source_two', 'source_two_rowsum', 'target_two_rowsum', 'operator')

    def __init__(self, x, value, source_kernels, target_kernels):
        self.x = x
        self.value = value
        self.source_kernels = source_kernels
        self.target_kernels = target_kernels
        self.gradient = None
        self.source_two = None
        self.source_two_rowsum = None
        self.target_two_rowsum = None
        self.operator = None


class CachedMMDEnergy:
    """(1/2) MMD_k^2 without the constant C, with per-state caches.

    For a large target sample (M >> N) the particle--target terms dominate.
    An evaluation stores the kernel matrices of its state, the gradient
    reuses them, and the first model-operator product at a state prepares
    the per-particle Hessian blocks diag(nabla^2 V(X_i)) and the weights of
    the interaction term.  Later products at that state reuse them.
    """

    def __init__(self, target, bandwidths=BANDWIDTHS):
        self.target = np.array(target, dtype=np.float64, order='C')
        self.bandwidths = tuple(float(h) for h in bandwidths)
        inverse_two = np.array([h ** -2 for h in self.bandwidths])
        self.mixture_weight = 1.0 / len(self.bandwidths)
        self.exponent_factors = -0.5 * inverse_two
        self.weights_two = self.mixture_weight * inverse_two
        self.weights_four = self.mixture_weight * inverse_two ** 2
        self.calls = dict(objective=0, gradient=0, preparation=0, application=0)

    @staticmethod
    def squared_distances(x, y):
        result = np.zeros((x.shape[0], y.shape[0]))
        for axis in range(x.shape[1]):
            difference = np.subtract.outer(x[:, axis], y[:, axis])
            np.multiply(difference, difference, out=difference)
            result += difference
        return result

    def _kernel_stack(self, squared_distance):
        stack = np.empty((len(self.bandwidths),) + squared_distance.shape)
        for index, factor in enumerate(self.exponent_factors):
            np.multiply(squared_distance, factor, out=stack[index])
            np.exp(stack[index], out=stack[index])
        return stack

    def value(self, x) -> MMDState:
        self.calls['objective'] += 1
        x = np.ascontiguousarray(x, dtype=np.float64)
        source = self._kernel_stack(self.squared_distances(x, x))
        target = self._kernel_stack(self.squared_distances(x, self.target))
        count, target_count = x.shape[0], self.target.shape[0]
        source_average = self.mixture_weight * float(np.sum(source)) / (count * count)
        target_average = self.mixture_weight * float(np.sum(target)) / (count * target_count)
        return MMDState(x, 0.5 * (source_average - 2.0 * target_average), source, target)

    def _weighted_products(self, stack, right, weights):
        """Return sum_h weights_h K_h [right, 1] in one pass over the stack."""
        count = stack.shape[1]
        features = np.concatenate((right, np.ones((right.shape[0], 1))), axis=1)
        products = (stack.reshape(-1, stack.shape[2]) @ features).reshape(
            stack.shape[0], count, features.shape[1])
        combined = np.tensordot(weights, products, axes=(0, 0))
        return combined[:, :-1], combined[:, -1]

    def gradient(self, state: MMDState):
        if state.gradient is not None:
            return state.gradient
        self.calls['gradient'] += 1
        x = state.x
        count, target_count = x.shape[0], self.target.shape[0]
        source_product, source_rowsum = self._weighted_products(state.source_kernels, x, self.weights_two)
        target_product, target_rowsum = self._weighted_products(state.target_kernels, self.target,
                                                                self.weights_two)
        state.gradient = ((source_product - source_rowsum[:, None] * x) / count
                          + (target_rowsum[:, None] * x - target_product) / target_count)
        state.source_two_rowsum = source_rowsum
        state.target_two_rowsum = target_rowsum
        return state.gradient

    @staticmethod
    def _second_moment_blocks(weights, x, y):
        """Return sum_j weights_ij (x_i - y_j)(x_i - y_j)^T, of shape (n, d, d)."""
        count, dimension = x.shape
        features = np.concatenate(
            (np.ones((y.shape[0], 1)), y,
             (y[:, :, None] * y[:, None, :]).reshape(y.shape[0], dimension * dimension)), axis=1)
        moments = weights @ features
        total = moments[:, 0]
        first = moments[:, 1:1 + dimension]
        second = moments[:, 1 + dimension:].reshape(count, dimension, dimension)
        return (total[:, None, None] * x[:, :, None] * x[:, None, :]
                - x[:, :, None] * first[:, None, :] - first[:, :, None] * x[:, None, :] + second)

    def _prepare(self, state: MMDState):
        if state.operator is not None:
            return state.operator
        self.gradient(state)
        self.calls['preparation'] += 1
        count, dimension = state.x.shape
        target_count = self.target.shape[0]
        state.source_two = np.tensordot(self.weights_two, state.source_kernels, axes=(0, 0))
        source_four = np.tensordot(self.weights_four, state.source_kernels, axes=(0, 0))
        target_four = np.tensordot(self.weights_four, state.target_kernels, axes=(0, 0))
        # The blocks depend only on differences; centering reduces cancellation.
        center = state.x.mean(axis=0)
        x = state.x - center
        y = self.target - center
        source_blocks = self._second_moment_blocks(source_four, x, x)
        target_blocks = self._second_moment_blocks(target_four, x, y)
        identity = np.eye(dimension)
        diagonal = ((source_blocks - state.source_two_rowsum[:, None, None] * identity) / count
                    + (state.target_two_rowsum[:, None, None] * identity - target_blocks) / target_count)
        state.operator = (source_four, x, diagonal)
        return state.operator

    def action(self, state: MMDState, direction):
        source_four, x, diagonal = self._prepare(state)
        self.calls['application'] += 1
        count, dimension = x.shape
        projection = np.einsum('ij,ij->i', x, direction)
        features = np.concatenate(
            (direction, projection[:, None],
             (x[:, :, None] * direction[:, None, :]).reshape(count, dimension * dimension),
             x * projection[:, None]), axis=1)
        moments = source_four @ features
        weighted_direction = moments[:, :dimension]
        weighted_projection = moments[:, dimension]
        cross = moments[:, dimension + 1:dimension + 1 + dimension * dimension].reshape(
            count, dimension, dimension)
        weighted_position = moments[:, dimension + 1 + dimension * dimension:]
        # sum_j A_ij (x_i - x_j)(x_i - x_j)^T s_j, expanded into matrix products.
        coupling = (x * np.einsum('ij,ij->i', x, weighted_direction)[:, None]
                    - x * weighted_projection[:, None]
                    - np.einsum('iab,ib->ia', cross, x) + weighted_position)
        return np.einsum('iab,ib->ia', diagonal, direction) - (coupling - state.source_two @ direction) / count

    def counts(self):
        return dict(self.calls)
