"""Gaussian core model in a harmonic trap: PyTorch oracle and WTR solver.

The empirical energy of N particles X_1, ..., X_N in R^2 is

    F(X) = kappa/(2N) sum_i |X_i|^2
           + a/(2N^2) sum_{i,j} exp(-|X_i - X_j|^2 / (2 sigma^2)),

with the diagonal terms i = j included.  Gradients and Hessian-vector
products are taken in the L^2(rho_N) inner product <u, v> = (1/N) sum_i u_i.v_i,
so the gradient norm is the root-mean-square force.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
import time
import traceback

import numpy as np
import torch

SIGMA = 1 / math.sqrt(2)
TOLERANCE = 2e-4

# Trust-region parameters used for every Gaussian-core run in the paper.
WTR_PARAMETERS = dict(initial_radius=2., maximum_radius=8., eta_accept=.1,
                      eta_expand=.75, gamma_shrink=.25, gamma_expand=2.,
                      boundary_fraction=.8, inner_cap=3, cg_rtol=.1)


class Deadline(Exception):
    pass


class Finished(Exception):
    pass


def synchronize(device):
    device = torch.device(device)
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def ip(x, y):
    """Inner product of L^2(rho_N), returned as a tensor on the device."""
    return (x * y).sum() / x.shape[0]


def norm_tensor(x):
    return torch.sqrt(torch.clamp(ip(x, x), min=0))


def norm(x):
    return float(norm_tensor(x))


def array_sha(x):
    return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()


class Timer:
    def __init__(self, budget, device='cuda'):
        self.device = torch.device(device)
        synchronize(self.device)
        self.start = time.perf_counter()
        self.cpu_start = time.process_time()
        self.budget = float(budget)

    def elapsed(self, sync=False):
        if sync:
            synchronize(self.device)
        return time.perf_counter() - self.start

    def check(self):
        if self.elapsed() >= self.budget:
            raise Deadline


@dataclass
class Counts:
    objective: int = 0
    gradient: int = 0
    hessian_vector: int = 0
    kernel_builds: int = 0
    model_precomputations: int = 0


@dataclass
class State:
    x: torch.Tensor
    kernel: torch.Tensor
    energy_tensor: torch.Tensor
    gradient: torch.Tensor | None = None
    centered: torch.Tensor | None = None
    rows: torch.Tensor | None = None
    hessian: torch.Tensor | None = None
    _energy: float | None = None

    @property
    def energy(self):
        if self._energy is None:
            self._energy = float(self.energy_tensor)
        return self._energy


class GaussianCore:
    """Energy, gradient and model operator of the Gaussian core model.

    The kernel matrix is computed once per state and reused by the gradient.
    The first Hessian-vector product at a state assembles the 2N x 2N matrix
    of the model operator; later products at that state reuse it.
    """

    def __init__(self, kappa, a=1., sigma=SIGMA, timer=None):
        self.kappa, self.a = float(kappa), float(a)
        self.b = float(sigma) ** -2
        self.timer = timer
        self.count = Counts()

    def check(self):
        if self.timer is not None:
            self.timer.check()

    @torch.no_grad()
    def value(self, x):
        self.check()
        self.count.objective += 1
        self.count.kernel_builds += 1
        x = x.detach().clone(memory_format=torch.contiguous_format)
        # Direct differences avoid the cancellation in |x|^2 + |y|^2 - 2 x.y.
        kernel = torch.cdist(x, x, p=2, compute_mode='donot_use_mm_for_euclid_dist')
        kernel.square_().mul_(-.5 * self.b).exp_()
        energy = .5 * self.kappa * x.square().sum() / len(x)
        energy = energy + .5 * self.a * kernel.mean()
        state = State(x, kernel, energy)
        self.check()
        return state

    @torch.no_grad()
    def gradient(self, state):
        self.check()
        if state.gradient is None:
            self.count.gradient += 1
            state.centered = state.x - state.x.mean(dim=0)
            state.rows = state.kernel.sum(dim=1)
            pair = state.rows[:, None] * state.centered - state.kernel @ state.centered
            state.gradient = self.kappa * state.x - self.a * self.b / len(state.x) * pair
        self.check()
        return state.gradient

    @torch.no_grad()
    def _assemble(self, state):
        x, k, n = state.centered, state.kernel, len(state.x)
        dx = x[:, 0, None] - x[None, :, 0]
        dy = x[:, 1, None] - x[None, :, 1]
        hxx = self.a * self.b * k * (self.b * dx.square() - 1)
        hyy = self.a * self.b * k * (self.b * dy.square() - 1)
        hxy = self.a * self.b * self.b * k * dx * dy
        # Coordinates are ordered (x_1, ..., x_N, y_1, ..., y_N).
        hessian = -torch.cat((torch.cat((hxx, hxy), dim=1),
                              torch.cat((hxy, hyy), dim=1)), dim=0) / n
        i = torch.arange(n, device=x.device)
        hessian[i, i] += self.kappa + hxx.sum(dim=1) / n
        hessian[n + i, n + i] += self.kappa + hyy.sum(dim=1) / n
        cross = hxy.sum(dim=1) / n
        hessian[i, n + i] += cross
        hessian[n + i, i] += cross
        state.hessian = hessian

    @torch.no_grad()
    def action(self, state, direction):
        self.check()
        self.gradient(state)
        self.count.hessian_vector += 1
        if state.hessian is None:
            self.count.model_precomputations += 1
            self._assemble(state)
        result = state.hessian @ direction.T.contiguous().view(-1)
        self.check()
        return result.view(2, len(direction)).T.contiguous()

    def counts(self):
        return asdict(self.count)


def boundary(z, p, radius):
    """Positive root tau of |z + tau p| = radius, computed stably."""
    a, b = ip(p, p), ip(z, p)
    c = torch.clamp(ip(z, z) - radius * radius, max=0)
    discriminant = torch.sqrt(torch.clamp(b * b - a * c, min=0))
    tiny = torch.finfo(z.dtype).tiny
    return torch.where(b > 0, -c / torch.clamp(discriminant + b, min=tiny),
                       (-b + discriminant) / torch.clamp(a, min=tiny))


@torch.no_grad()
def steihaug(g, radius, action, maxiter, rtol, initial_squared_norm=None):
    """Truncated conjugate gradients for the trust-region subproblem.

    Returns the step s, the product B s (accumulated along the iteration, so
    the predicted decrease needs no extra product) and the exit information.
    """
    z = torch.zeros_like(g)
    hz = torch.zeros_like(g)
    r, p = g.clone(), -g
    rr = ip(r, r)
    initial2 = float(rr) if initial_squared_norm is None else initial_squared_norm
    if not math.isfinite(initial2):
        raise FloatingPointError('nonfinite initial CG residual')
    if initial2 == 0:
        return z, hz, dict(iterations=0, exit_mode='zero_gradient')
    for iteration in range(1, maxiter + 1):
        hp = action(p)
        curvature = float(ip(p, hp))
        if not math.isfinite(curvature):
            raise FloatingPointError('nonfinite CG curvature')
        if curvature <= 0:
            tau = boundary(z, p, radius)
            return z + tau * p, hz + tau * hp, dict(iterations=iteration, exit_mode='negative_curvature')
        alpha = rr / curvature
        candidate = z + alpha * p
        rnew = r + alpha * hp
        rr_new = ip(rnew, rnew)
        candidate2, residual2 = torch.stack((ip(candidate, candidate), rr_new)).tolist()
        if not math.isfinite(candidate2) or not math.isfinite(residual2):
            raise FloatingPointError('nonfinite CG state')
        if candidate2 >= radius * radius:
            tau = boundary(z, p, radius)
            return z + tau * p, hz + tau * hp, dict(iterations=iteration, exit_mode='boundary')
        z, r = candidate, rnew
        hz = hz + alpha * hp
        if residual2 <= rtol * rtol * initial2:
            return z, hz, dict(iterations=iteration, exit_mode='residual')
        p = -r + (rr_new / rr) * p
        rr = rr_new
    return z, hz, dict(iterations=maxiter, exit_mode='inner_cap')


class Run:
    """Timing and recording shared by WTR and the L-BFGS baselines.

    The timer starts after the initial configuration is generated on the host
    and includes the transfer to the device, all setup and every evaluation.
    Each accepted state is recorded after a device synchronization.
    """

    def __init__(self, x0, kappa, method, params, device='cuda', budget=60.,
                 tolerance=TOLERANCE, max_iterations=100000):
        self.x0 = np.asarray(x0, dtype=np.float64).copy()
        self.method, self.params = method, dict(params)
        self.device = torch.device(device)
        self.budget, self.tolerance = float(budget), float(tolerance)
        self.max_iterations = int(max_iterations)
        self.history, self.events, self.accepted = [], [], None
        self.timer = Timer(budget, self.device)
        self.x = torch.as_tensor(self.x0, dtype=torch.float64, device=self.device).clone()
        self.oracle = GaussianCore(kappa, timer=self.timer)

    def record(self, state, iteration):
        gradient = self.oracle.gradient(state)
        energy, force = torch.stack((state.energy_tensor, norm_tensor(gradient))).tolist()
        if not math.isfinite(energy) or not math.isfinite(force):
            raise FloatingPointError('nonfinite energy or force')
        if self.accepted is not None and energy > self.accepted.energy + 1e-12 * max(1., abs(self.accepted.energy)):
            raise FloatingPointError('accepted energy increased')
        self.timer.check()
        state._energy = energy
        self.history.append(dict(iteration=iteration, energy=energy, force=force,
                                 time=self.timer.elapsed(), counts=self.oracle.counts()))
        self.accepted = state
        if force <= self.tolerance:
            raise Finished

    def finish(self, reason, iterations, message=None):
        wall = self.timer.elapsed(sync=True)
        last = self.history[-1] if self.history else None
        xf = self.accepted.x.cpu().numpy().copy() if self.accepted is not None else self.x0.copy()
        row = dict(method=self.method, params=self.params, device=str(self.device),
                   stop_reason=reason, message=message, budget_seconds=self.budget,
                   tolerance=self.tolerance, iterations=iterations,
                   accepted_iterations=max(0, len(self.history) - 1), wall_seconds=wall,
                   time_to_tolerance=last['time'] if reason == 'force_tolerance' else None,
                   terminal_energy=last['energy'] if last else None,
                   terminal_force=last['force'] if last else None,
                   input_sha256=array_sha(self.x0), final_sha256=array_sha(xf),
                   counts=self.oracle.counts(), history=self.history, events=self.events)
        return row, xf


def stop_reason(error):
    if isinstance(error, Finished):
        return 'force_tolerance', None
    if isinstance(error, Deadline):
        return 'time_budget', None
    if isinstance(error, (FloatingPointError, ArithmeticError)):
        return 'numerical_failure', str(error)
    return 'solver_failure', traceback.format_exc()


@torch.no_grad()
def run_wtr(x0, kappa, params=None, device='cuda', budget=60., tolerance=TOLERANCE,
            max_iterations=100000):
    """Algorithm 2.1 of the paper with the truncated-CG subsolver."""
    config = dict(WTR_PARAMETERS, **(params or {}))
    run = Run(x0, kappa, 'WTR', config, device, budget, tolerance, max_iterations)
    radius, iterations, reason, message = config['initial_radius'], 0, None, None
    try:
        state = run.oracle.value(run.x)
        run.record(state, 0)
        for iterations in range(1, run.max_iterations + 1):
            run.timer.check()
            g = run.oracle.gradient(state)
            step, hstep, info = steihaug(
                g, radius, lambda v: run.oracle.action(state, v), config['inner_cap'],
                config['cg_rtol'], initial_squared_norm=run.history[-1]['force'] ** 2)
            predicted, step_norm = torch.stack(
                (-(ip(g, step) + .5 * ip(step, hstep)), norm_tensor(step))).tolist()
            if not math.isfinite(predicted) or predicted <= 0:
                reason = 'nonpositive_prediction'
                break
            candidate = run.oracle.value(state.x + step)
            if not math.isfinite(candidate.energy):
                raise FloatingPointError('nonfinite trial energy')
            ratio = (state.energy - candidate.energy) / predicted
            accepted = math.isfinite(ratio) and ratio >= config['eta_accept']
            radius_before = radius
            if accepted:
                if ratio >= config['eta_expand'] and step_norm >= config['boundary_fraction'] * radius:
                    radius = min(config['gamma_expand'] * radius, config['maximum_radius'])
            else:
                radius *= config['gamma_shrink']
            run.events.append(dict(iteration=iterations, ratio=ratio if math.isfinite(ratio) else None,
                                   radius=radius_before, radius_next=radius, step_norm=step_norm,
                                   accepted=bool(accepted), cg=info))
            if accepted:
                state = candidate
                run.record(state, iterations)
        if reason is None:
            reason = 'iteration_cap'
    except Exception as error:
        reason, message = stop_reason(error)
    return run.finish(reason, iterations, message)


def explicit_reference(x, v, kappa, a=1., sigma=SIGMA):
    """Energy, gradient and Hessian-vector product by explicit double loops."""
    n, d = x.shape
    energy = .5 * kappa * float(np.sum(x * x)) / n
    gradient, action = kappa * x.copy(), kappa * v.copy()
    for i in range(n):
        for j in range(n):
            z, dv = x[i] - x[j], v[i] - v[j]
            kernel = a * math.exp(-float(z @ z) / (2 * sigma * sigma))
            energy += .5 * kernel / (n * n)
            gradient[i] -= kernel * z / (n * sigma * sigma)
            action[i] += kernel * (np.outer(z, z) / sigma ** 4 - np.eye(d) / sigma ** 2) @ dv / n
    return energy, gradient, action


def check_oracle(device='cpu'):
    """Compare the oracle with explicit sums and finite differences."""
    rng = np.random.default_rng(20261012)
    errors = {}
    for kappa in (.02, .08, .32):
        x_np = rng.normal(size=(9, 2))
        v_np = rng.normal(size=x_np.shape)
        x, v = (torch.tensor(a, dtype=torch.float64, device=device) for a in (x_np, v_np))
        f_ref, g_ref, hv_ref = explicit_reference(x_np, v_np, kappa)
        oracle = GaussianCore(kappa)
        state = oracle.value(x)
        g, hv = oracle.gradient(state), oracle.action(state, v)
        h = 1e-5
        plus, minus = oracle.value(x + h * v), oracle.value(x - h * v)
        errors[kappa] = dict(
            energy=abs(state.energy - f_ref),
            gradient=float(np.max(np.abs(g.cpu().numpy() - g_ref))),
            hessian_vector=float(np.max(np.abs(hv.cpu().numpy() - hv_ref))),
            gradient_fd=abs((plus.energy - minus.energy) / (2 * h) - float(ip(g, v))),
            hessian_vector_fd=norm((oracle.gradient(plus) - oracle.gradient(minus)) / (2 * h) - hv))
        assert max(errors[kappa].values()) < 2e-7, errors[kappa]
    return errors
