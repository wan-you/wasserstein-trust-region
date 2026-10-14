"""Equal-weight integration rules from MMD minimization (Section 4.4).

Nodes X_1, ..., X_N minimize the empirical MMD energy against M = 8192 target
points.  WTR is compared with L-BFGS (memory 20 and 50) and SciPy's
trust-krylov; accuracy is the RMSE of Q_N f over 75 Gaussian test functions.

    python -m experiments.integration prepare --out results/integration
    python -m experiments.integration run     --out results/integration
    python -m experiments.integration figures --out results/integration

``run`` executes the solvers one at a time with single-threaded numerical
libraries; times are measured with time.perf_counter.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import gc
import json
from pathlib import Path
import time

import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.special import ndtri, roots_legendre
from scipy.stats import qmc

from wtr.mmd import TARGETS, CachedMMDEnergy
from wtr.trust_region import particle_inner, particle_norm, steihaug_toint

METHODS = ('WTR', 'LBFGS20', 'LBFGS50', 'TR-Krylov')
SIZES = (64, 128, 256)
SEEDS = (301, 302, 303)
INITIAL_RADIUS, MAXIMUM_RADIUS, CG_ITERATIONS, CG_TOLERANCE = .075, .70, 24, 2e-6
TOLERANCE = 2e-6
BUDGET = 60.
WIDTHS = (.3, .6, 1.2)
CENTERS = np.array([(a, b) for a in (-1.8, -.9, 0., .9, 1.8) for b in (-1.8, -.9, 0., .9, 1.8)])
THRESHOLDS = (1e-2, 5e-3, 2e-3, 1e-3, 5e-4, 2e-4, 1e-4)
COLORS = {'WTR': '#1465A3', 'LBFGS20': '#D47723', 'LBFGS50': '#9252A0', 'TR-Krylov': '#23836D'}
LABELS = {'WTR': 'WTR', 'LBFGS20': 'L-BFGS20', 'LBFGS50': 'L-BFGS50', 'TR-Krylov': 'TR-Krylov'}
NAMES = {'banana': 'Banana', 'two_moons': 'Two moons', 'ring': 'Ring'}


class Finished(Exception):
    pass


class Deadline(Exception):
    pass


def target_points(name, exponent=13, seed=None):
    """Apply the sampling map of the target to a scrambled Sobol sequence."""
    seed = 71300 + 10 * TARGETS.index(name) if seed is None else seed
    u = qmc.Sobol(d=4 if name == 'two_moons' else 2, scramble=True,
                  rng=np.random.default_rng(seed)).random_base2(exponent)
    if name == 'banana':
        a, z = ndtri(u).T
        return np.column_stack((.90 * a, .30 * z + .34 * (a * a - 1)))
    if name == 'two_moons':
        theta = np.pi * u[:, 1]
        base = np.column_stack((np.cos(theta) - .50, np.sin(theta) - .25))
        base *= np.where(u[:, :1] < .5, 1., -1.)
        return 1.15 * (base + .055 * ndtri(u[:, 2:]))
    theta = 2 * np.pi * u[:, 0]
    radius = 1.35 + .08 * ndtri(u[:, 1])
    return radius[:, None] * np.column_stack((np.cos(theta), np.sin(theta)))


def test_values(x):
    """Values of the 75 test functions at the points x, ordered by width, then center."""
    distance = ((x[:, None, :] - CENTERS[None, :, :]) ** 2).sum(axis=2)
    return np.concatenate([np.exp(-.5 * distance / h ** 2) for h in WIDTHS], axis=1)


def reference_integrand(name, c, h):
    """One-dimensional integrand left after integrating the Gaussian factors exactly."""
    if name == 'banana':
        def integrand(t):
            return (h / np.sqrt(h * h + .30 ** 2) / np.sqrt(2 * np.pi)
                    * np.exp(-.5 * t * t - .5 * (.90 * t - c[0]) ** 2 / h ** 2
                             - .5 * (.34 * (t * t - 1) - c[1]) ** 2 / (h * h + .30 ** 2)))
        return integrand, -np.inf, np.inf
    if name == 'two_moons':
        def integrand(t):
            z = 1.15 * np.array([np.cos(t) - .50, np.sin(t) - .25])
            den = h * h + (1.15 * .055) ** 2
            return h * h / den / (2 * np.pi) * (np.exp(-((z - c) ** 2).sum() / (2 * den))
                                                + np.exp(-((z + c) ** 2).sum() / (2 * den)))
        return integrand, 0., np.pi

    def integrand(t):
        q = c[0] * np.cos(t) + c[1] * np.sin(t)
        orthogonal = max(0., float(c @ c - q * q))
        return (h / np.sqrt(h * h + .08 ** 2) / (2 * np.pi)
                * np.exp(-orthogonal / (2 * h * h) - (1.35 - q) ** 2 / (2 * (h * h + .08 ** 2))))
    return integrand, 0., 2 * np.pi


def reference_integrals(name):
    """Exact integrals by adaptive quadrature, checked by a 1024-point Gauss--Legendre rule."""
    values, difference = [], 0.
    nodes, weights = roots_legendre(1024)
    for h in WIDTHS:
        for c in CENTERS:
            f, lo, hi = reference_integrand(name, c, h)
            value, _ = quad(f, lo, hi, epsabs=2e-13, epsrel=2e-13, limit=300)
            left, right = (-10., 10.) if name == 'banana' else (lo, hi)
            points = (left + right) / 2 + (right - left) / 2 * nodes
            fixed = (right - left) / 2 * np.dot(weights, [f(t) for t in points])
            values.append(value)
            difference = max(difference, abs(value - fixed))
    assert difference < 2e-11, (name, difference)
    return values, difference


def prepare(out):
    out.mkdir(parents=True, exist_ok=True)
    targets = {name: target_points(name) for name in TARGETS}
    reference = {}
    for name in TARGETS:
        exact, difference = reference_integrals(name)
        empirical = test_values(targets[name]).mean(axis=0)
        reference[name] = dict(exact=exact, empirical=empirical.tolist(),
                               rule_difference=difference,
                               target_rmse=float(np.sqrt(np.mean((empirical - exact) ** 2))))
        print(f'{name}: quadrature check {difference:.1e}, '
              f'RMSE of the {len(targets[name])}-point rule {reference[name]["target_rmse"]:.3e}')
    np.savez_compressed(out / 'targets.npz', **targets)
    (out / 'reference.json').write_text(json.dumps(reference, indent=1) + '\n')


def solve(target, n, seed, method, budget=BUDGET, maxiter=10000):
    """Minimize the MMD energy from the initial configuration of ``seed``.

    Every solver records each distinct accepted iterate and stops when the
    gradient norm in L^2(rho_N) is at most TOLERANCE.  The quasi-Newton and
    Krylov baselines work in y = x / sqrt(N), where the Euclidean gradient
    norm equals that norm.
    """
    x0 = .95 * np.random.default_rng(23000 + seed).normal(size=(n, 2))
    gc.collect()
    gc.disable()
    start = time.perf_counter()
    oracle = CachedMMDEnergy(target)
    history, events, snapshots = [], [], []
    accepted, trials, reason, message = None, 0, None, None
    radius, scale = INITIAL_RADIUS, np.sqrt(n)

    def check():
        if time.perf_counter() - start >= budget:
            raise Deadline

    def evaluate(x, gradient=True):
        check()
        state = oracle.value(np.ascontiguousarray(x))
        check()
        if gradient:
            oracle.gradient(state)
            check()
        return state

    def record(state):
        nonlocal accepted
        check()
        energy, gradient_norm = state.value, particle_norm(oracle.gradient(state))
        if history:
            assert energy <= history[-1]['energy'] + 1e-11
        if accepted is None or not np.array_equal(accepted.x, state.x):
            snapshots.append(state.x.copy())
            history.append(dict(energy=energy, gradient=gradient_norm, counts=oracle.counts(),
                                time=time.perf_counter() - start))
        accepted = state
        if gradient_norm <= TOLERANCE:
            raise Finished

    try:
        state = evaluate(x0.copy())
        record(state)
        if method == 'WTR':
            for trials in range(1, maxiter + 1):
                g = oracle.gradient(state)

                def action(v):
                    check()
                    result = oracle.action(state, v)
                    check()
                    return result
                step, info = steihaug_toint(g, radius, action, CG_ITERATIONS, CG_TOLERANCE)
                predicted = -(particle_inner(g, step) + .5 * particle_inner(step, action(step)))
                if not np.isfinite(predicted) or predicted <= 0:
                    reason = 'nonpositive_prediction'
                    break
                candidate = evaluate(state.x + step, False)
                ratio = (state.value - candidate.value) / predicted
                ok = bool(np.isfinite(ratio) and ratio >= .1)
                events.append(dict(trial=trials, radius=radius, ratio=float(ratio), accepted=ok, cg=info))
                if ok:
                    if ratio >= .75 and particle_norm(step) >= .8 * radius:
                        radius = min(2 * radius, MAXIMUM_RADIUS)
                    state = candidate
                    record(state)
                else:
                    radius *= .25
            if reason is None:
                reason = 'iteration_cap'
        else:
            last_y, last_state = (x0 / scale).ravel(), state

            def get_state(y):
                nonlocal last_y, last_state
                check()
                if not np.array_equal(y, last_y):
                    last_state, last_y = evaluate(y.reshape(x0.shape) * scale), y.copy()
                return last_state

            def fg(y):
                st = get_state(y)
                return st.value, (oracle.gradient(st) / scale).ravel()

            def hessp(y, v):
                result = (oracle.action(get_state(y), v.reshape(x0.shape) * scale) / scale).ravel()
                check()
                return result

            def callback(y):
                nonlocal trials
                trials += 1
                record(get_state(y))

            if method.startswith('LBFGS'):
                result = minimize(fg, last_y.copy(), jac=True, method='L-BFGS-B', callback=callback,
                                  options=dict(maxcor=int(method[5:]), maxls=50, ftol=0., gtol=0.,
                                               maxiter=maxiter, maxfun=1000000))
            else:
                result = minimize(fg, last_y.copy(), jac=True, hessp=hessp, method='trust-krylov',
                                  callback=callback,
                                  options=dict(gtol=0., maxiter=maxiter, initial_trust_radius=INITIAL_RADIUS,
                                               max_trust_radius=MAXIMUM_RADIUS, eta=.1, inexact=True))
            reason, message = 'native_termination', str(result.message)
    except Finished:
        reason = 'gradient_tolerance'
    except Deadline:
        reason = 'time_budget'
    finally:
        wall = time.perf_counter() - start
        gc.enable()
    row = dict(target=None, method=method, n=n, seed=seed, stop_reason=reason, message=message,
               trials=trials, accepted_iterations=len(history) - 1, wall_seconds=wall,
               time_to_tolerance=history[-1]['time'] if reason == 'gradient_tolerance' else None,
               terminal_energy=history[-1]['energy'], terminal_gradient=history[-1]['gradient'],
               counts=oracle.counts(), history=history, events=events)
    return row, np.array(snapshots)


def add_integration_errors(row, states, reference):
    """Integration error at every recorded iterate, computed after the run."""
    exact = np.array(reference['exact'])
    for state, entry in zip(states, row['history'], strict=True):
        error = test_values(state).mean(axis=0) - exact
        entry['integration_rmse'] = float(np.sqrt(np.mean(error * error)))
    row['integration_threshold_times'] = {
        str(e): next((h['time'] for h in row['history'] if h['integration_rmse'] <= e), None)
        for e in THRESHOLDS}
    row['terminal_integration_rmse'] = row['history'][-1]['integration_rmse']


def run(out):
    targets = np.load(out / 'targets.npz')
    reference = json.loads((out / 'reference.json').read_text())
    rng = np.random.default_rng(20261020)
    tasks = [(t, n, s, [str(m) for m in rng.permutation(METHODS)])
             for t in TARGETS for n in SIZES for s in SEEDS]
    runs = out / 'runs'
    runs.mkdir(exist_ok=True)
    warmup = target_points('banana', exponent=7, seed=71999)
    for method in METHODS:
        solve(warmup, 16, 298, method, budget=2, maxiter=2)
    for target, n, seed, methods in tasks:
        for method in methods:
            row, states = solve(targets[target], n, seed, method)
            row['target'] = target
            add_integration_errors(row, states, reference[target])
            stem = f'{target}-n{n}-s{seed}-{method}'
            (runs / f'{stem}.json').write_text(json.dumps(row) + '\n')
            np.save(runs / f'{stem}-final.npy', states[-1])
            print(f"{stem}: {row['stop_reason']}, {row['wall_seconds']:.3f} s, "
                  f"RMSE {row['terminal_integration_rmse']:.4e}", flush=True)


def load_rows(out):
    rows = [json.loads(p.read_text()) for p in sorted((out / 'runs').glob('*.json'))]
    assert len(rows) == len(TARGETS) * len(SIZES) * len(SEEDS) * len(METHODS), len(rows)
    return rows


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.labelsize': 10, 'axes.titlesize': 11,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    rows = load_rows(out)
    reference = json.loads((out / 'reference.json').read_text())
    index = {(r['target'], r['n'], r['seed'], r['method']): r for r in rows}

    # Figure 4: error against time at N = 128, and time to the gradient tolerance.
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.0))
    fig.subplots_adjust(left=.085, right=.995, bottom=.08, top=.89, wspace=.25, hspace=.33)
    for j, target in enumerate(TARGETS):
        ax = axes[0, j]
        for method in METHODS:
            selected = [index[target, 128, s, method] for s in SEEDS]
            start = max(r['history'][0]['time'] for r in selected)
            end = max(r['history'][-1]['time'] for r in selected)
            grid = np.geomspace(start, end, 250)
            values = []
            for r in selected:
                times = np.array([h['time'] for h in r['history']])
                errors = np.array([h['integration_rmse'] for h in r['history']])
                values.append(errors[np.maximum(0, np.searchsorted(times, grid, side='right') - 1)])
            values = np.array(values)
            ax.plot(grid, np.median(values, axis=0), color=COLORS[method], lw=1.8, label=LABELS[method],
                    linestyle='--' if method == 'LBFGS50' else '-')
            ax.fill_between(grid, values.min(axis=0), values.max(axis=0), color=COLORS[method], alpha=.12, lw=0)
        ax.axhline(reference[target]['target_rmse'], color='.5', linestyle=':', lw=1,
                   label='8,192-point input RMSE' if j == 0 else None)
        ax.set(xscale='log', yscale='log', title=NAMES[target], xlabel='Node construction time (s)')
        ax.grid(True, alpha=.15, which='major')
        if j == 0:
            ax.set_ylabel('Integral RMSE (128 equal-weight nodes)')
        ax = axes[1, j]
        for method in METHODS:
            times = [[index[target, n, s, method]['time_to_tolerance'] for s in SEEDS] for n in SIZES]
            assert all(t is not None for group in times for t in group)
            median = np.array([np.median(t) for t in times])
            spread = np.array([median - [min(t) for t in times], [max(t) for t in times] - median])
            ax.errorbar(SIZES, median, yerr=spread, color=COLORS[method], marker='o', markersize=4, lw=1.6,
                        capsize=3, label=LABELS[method])
        ax.set(xscale='log', yscale='log', xlabel='Number of integration nodes', xticks=SIZES)
        ax.set_xticklabels([str(n) for n in SIZES])
        ax.minorticks_off()
        ax.grid(True, alpha=.15)
        if j == 0:
            ax.set_ylabel('Time to gradient tolerance (s)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.53, .985), ncol=5, frameon=False)
    save(fig, out / 'fig-integration-performance')

    # Figure 5: speedup of WTR at each threshold, two moons and ring.
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    cases = [(t, n, s) for t in ('two_moons', 'ring') for n in SIZES for s in SEEDS]
    for method in METHODS[1:]:
        ratios = np.array([[index[(*c, method)]['integration_threshold_times'][str(e)]
                            / index[(*c, 'WTR')]['integration_threshold_times'][str(e)] for c in cases]
                           for e in THRESHOLDS])
        ax.plot(THRESHOLDS, np.median(ratios, axis=1), color=COLORS[method], lw=2, marker='o', ms=4,
                label='vs ' + LABELS[method])
        ax.fill_between(THRESHOLDS, np.quantile(ratios, .25, axis=1), np.quantile(ratios, .75, axis=1),
                        color=COLORS[method], alpha=.12, lw=0)
        print(f'median speedup over {LABELS[method]} at RMSE 1e-4: {np.median(ratios[-1]):.2f}')
    ax.axhline(1, color='.4', linestyle=':', lw=1)
    ax.text(.000098, 1.015, 'Equal time', ha='right', fontsize=8, color='.35')
    ax.set_xscale('log')
    ax.invert_xaxis()
    ax.set(xlabel='Requested integral RMSE (stricter accuracy →)',
           ylabel='Speedup = baseline time / WTR time')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(True, alpha=.15)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, out / 'fig-integration-thresholds')

    # Table 2: median construction time to the stated RMSE at N = 128.
    lines = [r'\begin{tabular}{lrrrrr}', r'\hline',
             r'Target & RMSE & WTR & L-BFGS20 & L-BFGS50 & TR-Krylov \\', r'\hline']
    for target, threshold, label in (('banana', 1e-3, r'$10^{-3}$'), ('two_moons', 1e-4, r'$10^{-4}$'),
                                     ('ring', 1e-4, r'$10^{-4}$')):
        medians = []
        for method in METHODS:
            times = [index[target, 128, s, method]['integration_threshold_times'][str(threshold)] for s in SEEDS]
            assert all(t is not None for t in times)
            medians.append(f'{np.median(times):.3f}')
        lines.append(' & '.join([NAMES[target], label] + medians) + r' \\')
    lines += [r'\hline', r'\end{tabular}']
    (out / 'table-integration-time.tex').write_text('\n'.join(lines) + '\n')

    for method in METHODS[1:]:
        ratios = [index[t, n, s, method]['time_to_tolerance'] / index[t, n, s, 'WTR']['time_to_tolerance']
                  for t in TARGETS for n in SIZES for s in SEEDS]
        print(f'median speedup over {LABELS[method]} to the gradient tolerance: {np.median(ratios):.2f}')
    print('wrote figures and table to', out)


def save(fig, stem):
    import matplotlib.pyplot as plt
    fig.savefig(stem.with_suffix('.png'), dpi=200, bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('prepare', 'run', 'figures'))
    parser.add_argument('--out', type=Path, default=Path('results/integration'))
    args = parser.parse_args()
    {'prepare': prepare, 'run': run, 'figures': figures}[args.command](args.out)


if __name__ == '__main__':
    main()
