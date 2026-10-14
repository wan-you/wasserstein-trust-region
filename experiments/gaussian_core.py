"""Gaussian core model of dendrimers in a harmonic trap (Section 4.2).

Time to the RMS-force tolerance 2e-4 for WTR and a tuned L-BFGS baseline,
both on one GPU in float64, for N in {2048, 4096} and kappa in
{0.02, 0.08, 0.32}, from the initial configurations of seeds 47, 53, 59, 61.

    python -m experiments.gaussian_core check   --device cuda:0
    python -m experiments.gaussian_core tune    --device cuda:0 --out results/gaussian-core
    python -m experiments.gaussian_core run     --device cuda:0 --out results/gaussian-core
    python -m experiments.gaussian_core figures --out results/gaussian-core

``tune`` selects the L-BFGS variant for each (kappa, N) on seeds 101 and 103
and writes lbfgs-selection.json; ``run`` uses that file if present and the
selection reported in the paper otherwise.  Runs are executed one at a time.
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS', 'BLIS_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
import gc
import itertools
import json
import math
from pathlib import Path

import numpy as np

KAPPAS = (.02, .08, .32)
SIZES = (2048, 4096)
SEEDS = (47, 53, 59, 61)
TUNING_SEEDS = (101, 103)
REPEATS = 3
TOLERANCE = 2e-4
METHODS = ('WTR', 'LBFGS')
# L-BFGS variant selected on the tuning seeds for each (kappa, N), as in the paper.
PAPER_SELECTION = {
    'k0.02-n2048': 'scipy-m10', 'k0.02-n4096': 'scipy-m10',
    'k0.08-n2048': 'torch-m50-lr1', 'k0.08-n4096': 'scipy-m10',
    'k0.32-n2048': 'scipy-m20', 'k0.32-n4096': 'scipy-m20',
}


def key(kappa, n):
    return f'k{kappa:g}-n{n}'


def initial_configuration(n, seed):
    return np.random.default_rng(seed).normal(scale=2.4, size=(n, 2))


def configure(device):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def warmup(device):
    """Untimed warm-up of every solver and of the large matrix operations."""
    import torch
    from wtr.gaussian_core import GaussianCore, run_wtr, synchronize
    from wtr.lbfgs import run_lbfgs
    x0 = initial_configuration(16, 20261012)
    run_wtr(x0, .08, device=device, budget=30., max_iterations=2)
    for params in (dict(backend='torch', history_size=20, lr=1.), dict(backend='scipy', history_size=20)):
        run_lbfgs(x0, .08, params, device=device, budget=30., max_iterations=2)
    large = torch.as_tensor(initial_configuration(4096, 20261012), device=device, dtype=torch.float64)
    oracle = GaussianCore(.08)
    state = oracle.value(large)
    oracle.gradient(state)
    oracle.action(state, torch.ones_like(large))
    synchronize(device)


def solve(method, params, kappa, n, seed, device, budget):
    from wtr.gaussian_core import run_wtr, synchronize
    from wtr.lbfgs import run_lbfgs
    x0 = initial_configuration(n, seed)
    synchronize(device)
    gc.collect()
    if method == 'WTR':
        row, xf = run_wtr(x0, kappa, params, device=device, budget=budget, tolerance=TOLERANCE)
    else:
        row, xf = run_lbfgs(x0, kappa, params, device=device, budget=budget, tolerance=TOLERANCE)
    synchronize(device)
    row.update(kappa=kappa, n=n, seed=seed, method=method)
    return row, x0, xf


def score(rows, budget):
    """Failures, then geometric mean of time/budget, then of terminal force/tolerance."""
    failures, times, forces = 0, [], []
    for row in rows:
        success = row['stop_reason'] == 'force_tolerance'
        failures += not success
        times.append(max(1e-12, min(row['time_to_tolerance'] / budget, 1.)) if success else 1.)
        force = row['terminal_force']
        forces.append(max(1e-12, force / TOLERANCE) if force is not None and math.isfinite(force) else 1e100)
    return failures, float(np.exp(np.mean(np.log(times)))), float(np.exp(np.mean(np.log(forces))))


def tune(out, device, budget=2.):
    from wtr.lbfgs import candidates
    out.mkdir(parents=True, exist_ok=True)
    configure(device)
    warmup(device)
    variants = candidates()
    selection, scores = {}, {}
    for index, (kappa, n) in enumerate(itertools.product(KAPPAS, SIZES)):
        rows = {name: [] for name in variants}
        for seed in TUNING_SEEDS:
            order = np.random.default_rng(20261012 + 1000 * index + seed).permutation(sorted(variants))
            for name in order:
                row, _, _ = solve('LBFGS', variants[name], kappa, n, seed, device, budget)
                rows[name].append(row)
        scores[key(kappa, n)] = {name: score(r, budget) for name, r in rows.items()}
        selection[key(kappa, n)] = min(variants, key=lambda name: (*scores[key(kappa, n)][name], name))
        print(key(kappa, n), '->', selection[key(kappa, n)], flush=True)
    (out / 'lbfgs-selection.json').write_text(json.dumps(dict(selection=selection, scores=scores), indent=1))


def run(out, device, budget):
    from wtr.gaussian_core import WTR_PARAMETERS
    from wtr.lbfgs import candidates
    out.mkdir(parents=True, exist_ok=True)
    runs = out / 'runs'
    runs.mkdir(exist_ok=True)
    selection_file = out / 'lbfgs-selection.json'
    selection = (json.loads(selection_file.read_text())['selection'] if selection_file.exists()
                 else PAPER_SELECTION)
    variants = candidates()
    configure(device)
    warmup(device)
    blocks = [(kappa, n, seed, repeat) for kappa, n in itertools.product(KAPPAS, SIZES)
              for seed in SEEDS for repeat in range(REPEATS)]
    rng = np.random.default_rng(2026101247)
    for block in rng.permutation(len(blocks)):
        kappa, n, seed, repeat = blocks[int(block)]
        for method in rng.permutation(METHODS):
            method = str(method)
            params = WTR_PARAMETERS if method == 'WTR' else variants[selection[key(kappa, n)]]
            row, x0, xf = solve(method, params, kappa, n, seed, device, budget)
            row['repeat'] = repeat
            stem = f'{key(kappa, n)}-s{seed}-r{repeat}-{method}'
            (runs / f'{stem}.json').write_text(json.dumps(row) + '\n')
            np.savez_compressed(runs / f'{stem}.npz', x0=x0, xf=xf)
            print(f"{stem}: {row['stop_reason']}, time to tolerance {row['time_to_tolerance']}", flush=True)


def load(out):
    rows = {}
    for path in sorted((out / 'runs').glob('*.json')):
        row = json.loads(path.read_text())
        row['arrays'] = path.with_suffix('.npz')
        rows[row['kappa'], row['n'], row['seed'], row['method'], row['repeat']] = row
    expected = set(itertools.product(KAPPAS, SIZES, SEEDS, METHODS, range(REPEATS)))
    assert set(rows) == expected, expected - set(rows)
    assert all(r['stop_reason'] == 'force_tolerance' for r in rows.values())
    return rows


def figures(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.titlesize': 10.5,
                         'axes.labelsize': 9.5, 'xtick.labelsize': 9, 'ytick.labelsize': 9,
                         'pdf.fonttype': 42, 'ps.fonttype': 42})
    colors = {'WTR': '#176f77', 'LBFGS': '#cc703d'}
    styles = {'WTR': '-', 'LBFGS': '--'}
    names = {'WTR': 'WTR', 'LBFGS': 'L-BFGS'}
    rows = load(out)

    # For each case, the median of the three repeats; the speedup is the ratio of these medians.
    cases = []
    for kappa, n, seed in itertools.product(KAPPAS, SIZES, SEEDS):
        times = {m: float(np.median([rows[kappa, n, seed, m, r]['time_to_tolerance'] for r in range(REPEATS)]))
                 for m in METHODS}
        energies = [rows[kappa, n, seed, m, 0]['terminal_energy'] for m in METHODS]
        cases.append(dict(kappa=kappa, n=n, seed=seed, wtr=times['WTR'], lbfgs=times['LBFGS'],
                          speedup=times['LBFGS'] / times['WTR'],
                          energy_difference=abs(energies[0] - energies[1]) / abs(energies[1])))
    speedups = [c['speedup'] for c in cases]
    print(f'WTR faster in {sum(s > 1 for s in speedups)}/{len(cases)} cases; '
          f'median speedup {np.median(speedups):.3f}; maximum relative energy difference '
          f'{max(c["energy_difference"] for c in cases):.2e}')

    # Table 1.
    lines = [r'\begingroup', r'\small', r'\setlength{\tabcolsep}{3.5pt}', r'\begin{tabular}{rr rr r}', r'\hline',
             r'$\kappa$ & $N$ & \multicolumn{2}{c}{Time to tolerance (s)} & Speedup \\',
             r' & & WTR & L-BFGS & L-BFGS/WTR \\', r'\hline']
    for kappa, n in itertools.product(KAPPAS, SIZES):
        group = [c for c in cases if c['kappa'] == kappa and c['n'] == n]
        lines.append(f"{kappa:.2f} & {n} & {np.median([c['wtr'] for c in group]):.3f} & "
                     f"{np.median([c['lbfgs'] for c in group]):.3f} & "
                     f"{np.median([c['speedup'] for c in group]):.2f} " + r'\\')
    lines += [r'\hline', r'\end{tabular}', r'\endgroup']
    (out / 'table-gaussian-core-performance.tex').write_text('\n'.join(lines) + '\n')

    # Figure 1: speedup for every case.
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    regimes = list(itertools.product(KAPPAS, SIZES))
    for i, seed in enumerate(SEEDS):
        group = [c for c in cases if c['seed'] == seed]
        ax.scatter([regimes.index((c['kappa'], c['n'])) + (i - 1.5) * .11 for c in group],
                   [c['speedup'] for c in group], color='#216890', marker='o', s=27,
                   edgecolors='white', linewidths=.35, zorder=3)
    ax.axhline(1, color='#333333', ls='--', lw=1.0)
    ax.set(xlim=(-.5, 5.5), ylim=(.9, max(speedups) * 1.15), xlabel=r'$\kappa$ / $N$',
           ylabel='L-BFGS time / WTR time')
    ax.set_xticks(range(6), [f'{k:g}\n{n}' for k, n in regimes], fontsize=8)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', color='#dddddd', lw=.6)
    fig.subplots_adjust(left=.14, right=.98, bottom=.22, top=.96)
    save(fig, out / 'fig-gaussian-core-speedup')

    # Figure 2: final configurations, radial distributions and force histories at N = 4096.
    fig, axes = plt.subplots(3, 3, figsize=(7.2, 7.0), gridspec_kw={'height_ratios': [1, .85, 1.0]})
    for col, kappa in enumerate(KAPPAS):
        with np.load(rows[kappa, 4096, 47, 'WTR', 0]['arrays']) as data:
            x0, xf = data['x0'], data['xf']
        ax = axes[0, col]
        ax.scatter(xf[:, 0], xf[:, 1], s=2.8, color=colors['WTR'], alpha=.5, edgecolors='none', rasterized=True)
        ax.set(xlim=(-3.05, 3.05), ylim=(-3.05, 3.05), xlabel=r'$x_1$', ylabel=r'$x_2$', title=rf'$\kappa={kappa:g}$')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks((-3, 0, 3))
        ax.set_yticks((-3, 0, 3))
        ax = axes[1, col]
        curves = [(x0, '#999999', '-.', 1.1)]
        for method in METHODS:
            with np.load(rows[kappa, 4096, 47, method, 0]['arrays']) as data:
                curves.append((data['xf'], colors[method], styles[method], 1.7))
        for points, color, style, width in curves:
            radius = np.sort(np.linalg.norm(points, axis=1))
            ax.step(radius, np.arange(1, len(radius) + 1) / len(radius), where='post', color=color, ls=style, lw=width)
        ax.set(xlim=(0, 9.8), ylim=(-.02, 1.04), xlabel=r'Radius $|x|$', ylabel='Radial CDF')
        ax.set_xticks((0, 3, 6, 9))
        ax = axes[2, col]
        for seed in SEEDS[1:] + SEEDS[:1]:
            for method in METHODS:
                history = rows[kappa, 4096, seed, method, 0]['history']
                ax.plot([h['time'] for h in history], [h['force'] for h in history],
                        color=colors[method], ls=styles[method], lw=1.0, alpha=.8)
        ax.axhline(TOLERANCE, color='#555555', lw=.8, ls=(0, (2, 3)), zorder=0)
        ax.set(yscale='log', ylim=(6e-5, 2), xlim=(0, None), xlabel='Elapsed time (s)', ylabel='RMS force')
        ax.grid(which='major', color='#dedede', lw=.5)
    for i, ax in enumerate(axes.flat):
        ax.text(-.18, 1.035, f'({chr(97 + i)})', transform=ax.transAxes, fontweight='bold', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    handles = [Line2D([0], [0], color=colors[m], ls=styles[m], lw=1.8, label=names[m]) for m in METHODS]
    handles.append(Line2D([0], [0], color='#999999', ls='-.', lw=1.2, label='Initial (radial CDF)'))
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .007), ncol=3, frameon=False, fontsize=9.5)
    fig.subplots_adjust(left=.085, right=.99, top=.96, bottom=.15, hspace=.50, wspace=.44)
    save(fig, out / 'fig-gaussian-core-dynamics')
    print('wrote figures and table to', out)


def save(fig, stem):
    import matplotlib.pyplot as plt
    fig.savefig(stem.with_suffix('.png'), dpi=240, facecolor='white', bbox_inches='tight')
    fig.savefig(stem.with_suffix('.pdf'), dpi=240, facecolor='white', bbox_inches='tight',
                metadata={'CreationDate': None, 'ModDate': None})
    plt.close(fig)


def check(device):
    from wtr.gaussian_core import check_oracle, run_wtr
    configure(device)
    for kappa, errors in check_oracle(device).items():
        print(f'kappa={kappa}: maximum error {max(errors.values()):.2e}')
    row, _ = run_wtr(initial_configuration(128, 1), .08, device=device)
    print(f"N=128: {row['stop_reason']} after {row['iterations']} iterations, "
          f"RMS force {row['terminal_force']:.2e}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('check', 'tune', 'run', 'figures'))
    parser.add_argument('--out', type=Path, default=Path('results/gaussian-core'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--budget', type=float, default=60., help='time limit per run in seconds')
    args = parser.parse_args()
    if args.command == 'check':
        check(args.device)
    elif args.command == 'tune':
        tune(args.out, args.device)
    elif args.command == 'run':
        run(args.out, args.device, args.budget)
    else:
        figures(args.out)


if __name__ == '__main__':
    main()
