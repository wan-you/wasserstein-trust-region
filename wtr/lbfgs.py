"""L-BFGS baselines for the Gaussian core model.

Both solvers work in the coordinates y = x / sqrt(N), in which the Euclidean
gradient norm equals the RMS force used by WTR, and both stop with the same
test on recorded accepted states.  PyTorch's L-BFGS runs entirely on the
device; SciPy's L-BFGS-B keeps its recursion on the host and evaluates the
energy and gradient on the device, with all transfers inside the timer.
"""
from __future__ import annotations

import math

import numpy as np
import torch
from scipy.optimize import minimize

from .gaussian_core import TOLERANCE, Run, stop_reason


def candidates():
    """The twelve L-BFGS variants among which the baseline is tuned."""
    result = {}
    for memory in (10, 20, 50):
        for lr in (.25, 1., 4.):
            result[f'torch-m{memory}-lr{lr:g}'] = dict(backend='torch', history_size=memory, lr=lr)
        result[f'scipy-m{memory}'] = dict(backend='scipy', history_size=memory)
    return result


def run_lbfgs(x0, kappa, params, device='cuda', budget=60., tolerance=TOLERANCE,
              max_iterations=100000):
    if params['backend'] == 'torch':
        return _run_torch(x0, kappa, params, device, budget, tolerance, max_iterations)
    return _run_scipy(x0, kappa, params, device, budget, tolerance, max_iterations)


def _run_torch(x0, kappa, params, device, budget, tolerance, max_iterations):
    """PyTorch L-BFGS with strong-Wolfe line search, one iteration per step()."""
    run = Run(x0, kappa, 'L-BFGS', params, device, budget, tolerance, max_iterations)
    iterations, reason, message = 0, None, None
    try:
        scale = math.sqrt(len(x0))
        y = (run.x / scale).reshape(-1).detach().clone().requires_grad_(True)
        state = run.oracle.value(y.detach().reshape(run.x.shape) * scale)
        run.record(state, 0)
        # PyTorch's strong-Wolfe search makes at most 25 evaluations.
        optimizer = torch.optim.LBFGS(
            [y], lr=float(params.get('lr', 1.)), max_iter=1, max_eval=26,
            tolerance_grad=0., tolerance_change=0., history_size=int(params['history_size']),
            line_search_fn='strong_wolfe')
        accepted_y = y.detach().clone()
        latest_y, latest_state = accepted_y, state
        first_closure = True

        def closure():
            nonlocal first_closure, latest_y, latest_state
            run.timer.check()
            with torch.no_grad():
                if first_closure:
                    current = run.accepted
                    first_closure = False
                else:
                    current = run.oracle.value(y.detach().reshape(run.x.shape) * scale)
                    if not math.isfinite(current.energy):
                        raise FloatingPointError('nonfinite line-search energy')
                    latest_y, latest_state = y.detach().clone(), current
                y.grad = (run.oracle.gradient(current) / scale).reshape(-1).detach()
                run.timer.check()
                return current.energy_tensor.detach()

        for iterations in range(1, run.max_iterations + 1):
            run.timer.check()
            first_closure = True
            optimizer.step(closure)
            run.timer.check()
            if torch.equal(y.detach(), accepted_y):
                reason, message = 'stagnation', 'step left the iterate unchanged'
                break
            # The line search may return a point other than its last trial.
            if torch.equal(y.detach(), latest_y):
                next_state = latest_state
            else:
                next_state = run.oracle.value(y.detach().reshape(run.x.shape) * scale)
            run.record(next_state, iterations)
            accepted_y = y.detach().clone()
            latest_y, latest_state = accepted_y, next_state
        if reason is None:
            reason = 'iteration_cap'
    except Exception as error:
        reason, message = stop_reason(error)
    return run.finish(reason, iterations, message)


def _run_scipy(x0, kappa, params, device, budget, tolerance, max_iterations):
    """SciPy L-BFGS-B (unconstrained) with the device energy and gradient."""
    run = Run(x0, kappa, 'L-BFGS', params, device, budget, tolerance, max_iterations)
    iterations, reason, message = 0, None, None
    try:
        scale = math.sqrt(len(x0))
        initial_y = (np.asarray(x0, dtype=np.float64) / scale).reshape(-1).copy()
        first_x = torch.as_tensor(initial_y.reshape(run.x.shape), dtype=torch.float64,
                                  device=run.device) * scale
        first_state = run.oracle.value(first_x)
        run.record(first_state, 0)
        # Each cache holds [host coordinates, device state, host gradient].
        accepted_cache = [initial_y.copy(), first_state, None]
        latest_cache = accepted_cache

        def get_cache(host_y):
            nonlocal latest_cache
            run.timer.check()
            if np.array_equal(host_y, accepted_cache[0]):
                return accepted_cache
            if not np.array_equal(host_y, latest_cache[0]):
                x = torch.as_tensor(np.asarray(host_y).reshape(run.x.shape), dtype=torch.float64,
                                    device=run.device) * scale
                candidate = run.oracle.value(x)
                if not math.isfinite(candidate.energy):
                    raise FloatingPointError('nonfinite line-search energy')
                latest_cache = [host_y.copy(), candidate, None]
            return latest_cache

        def fun(host_y):
            result = get_cache(host_y)[1].energy
            run.timer.check()
            return result

        def jac(host_y):
            cache = get_cache(host_y)
            if cache[2] is None:
                cache[2] = (run.oracle.gradient(cache[1]) / scale).reshape(-1).cpu().numpy().copy()
            run.timer.check()
            return cache[2]

        def callback(host_y):
            nonlocal iterations, accepted_cache
            iterations += 1
            cache = get_cache(host_y)
            run.record(cache[1], iterations)
            accepted_cache = cache

        result = minimize(fun, initial_y, jac=jac, method='L-BFGS-B', callback=callback,
                          options=dict(maxcor=int(params['history_size']), maxls=50, ftol=0., gtol=0.,
                                       maxiter=run.max_iterations, maxfun=10_000_000))
        reason, message = 'native_termination', str(result.message)
    except Exception as error:
        reason, message = stop_reason(error)
    return run.finish(reason, iterations, message)
