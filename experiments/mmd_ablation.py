"""MMD fitting of the banana, two-moons and ring targets (Section 4.3, Figure 3).

Compares the full method with a Cauchy-only variant (B = 0) and a variant
without radius growth.  For every target and initial configuration the three
variants share the initialization and both target samples.

    python -m experiments.mmd_ablation run --out results/mmd-ablation --workers 8
    python -m experiments.mmd_ablation figure --out results/mmd-ablation
"""
from __future__ import annotations

import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
              'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[_name] = '1'

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

import numpy as np

from wtr.mmd import TARGETS, HalfSquaredMMD, sample_target
from wtr.trust_region import VARIANTS, TrustRegionConfig, run_trust_region

SEEDS = range(8)
PARTICLES = 1000
TRAIN_COUNT, HOLDOUT_COUNT = 384, 768
TARGET_SEED, INITIAL_SEED = 33000, 23000
INITIAL_SCALE = 0.95
TOLERANCE = 2e-4
MAXIMUM_ITERATIONS = 70
CONFIG = TrustRegionConfig(initial_radius=0.075, maximum_radius=0.70,
                           cg_maximum_iterations=24, cg_relative_tolerance=2e-6)
LABELS = {'banana': 'Banana', 'two_moons': 'Two moons', 'ring': 'Ring'}
COLORS = {'full': '#3567a8', 'cauchy': '#d28b38', 'no_growth': '#4d8b63'}
ZOOM = {'banana': ((5.0, 20.0), (1.35e-3, 4.50e-3)),
        'two_moons': ((5.0, 20.0), (1.60e-3, 4.80e-3)),
        'ring': ((5.0, 20.0), (8.00e-4, 3.30e-3))}


def samples(target):
    seed = TARGET_SEED + 10 * TARGETS.index(target)
    return sample_target(target, TRAIN_COUNT, seed), sample_target(target, HOLDOUT_COUNT, seed + 1)


def initial_configuration(seed):
    rng = np.random.default_rng(INITIAL_SEED + seed)
    return INITIAL_SCALE * rng.standard_normal((PARTICLES, 2))


def run_case(task):
    target, seed = task
    train, holdout = samples(target)
    energy, holdout_energy = HalfSquaredMMD(train), HalfSquaredMMD(holdout)
    x0 = initial_configuration(seed)
    rows, finals = [], {}
    for variant in VARIANTS:
        result = run_trust_region(x0, energy, CONFIG, MAXIMUM_ITERATIONS, TOLERANCE,
                                  variant=variant, diagnostic=holdout_energy.value)
        finals[variant] = result.pop('x_final')
        rows.append(dict(target=target, seed=seed, variant=variant, **result))
    return rows, finals, x0


def check_hessian(seed=99001):
    """Compare analytic model-operator products with gradient differences."""
    rng = np.random.default_rng(seed)
    errors = {}
    for target in TARGETS:
        x = rng.normal(size=(16, 2))
        energy = HalfSquaredMMD(samples(target)[0][:28])
        direction = rng.standard_normal(x.shape)
        analytic = energy.hvp(x, direction)
        eps = 2e-6
        numeric = (energy.gradient(x + eps * direction) - energy.gradient(x - eps * direction)) / (2 * eps)
        errors[target] = float(np.linalg.norm(analytic - numeric) / np.linalg.norm(analytic))
        assert errors[target] < 3e-5, errors
    return errors


def run(out, workers):
    out.mkdir(parents=True, exist_ok=True)
    print('relative model-operator errors:', check_hessian())
    tasks = [(target, seed) for target in TARGETS for seed in SEEDS]
    rows, arrays = [], {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for case_rows, finals, x0 in pool.map(run_case, tasks):
            rows += case_rows
            target, seed = case_rows[0]['target'], case_rows[0]['seed']
            arrays[f'{target}-s{seed}-initial'] = x0
            for variant, x in finals.items():
                arrays[f'{target}-s{seed}-{variant}'] = x
            for row in case_rows:
                print(f"{target:9s} seed {seed} {row['variant']:9s} iterations {row['iterations']:3d} "
                      f"holdout {row['diagnostic_history'][-1]:.6e} ({row['termination']})", flush=True)
    (out / 'results.json').write_text(json.dumps(dict(rows=rows), indent=1) + '\n')
    np.savez_compressed(out / 'particles.npz', **arrays)
    for target in TARGETS:
        medians = {v: np.median([r['diagnostic_history'][-1] for r in rows
                                 if r['target'] == target and r['variant'] == v]) for v in VARIANTS}
        print(f'{target}: median final holdout loss ' +
              ', '.join(f'{v} {m:.4e}' for v, m in medians.items()))


def ragged_summary(histories, stops):
    """Quartiles of histories that stop at different iterations.

    At each iteration only the runs that reached it contribute.  The curve
    ends at the median stopping iteration; if that is a half-integer, the end
    point is interpolated geometrically between its two neighbours.
    """
    histories = [np.asarray(h, dtype=float) for h in histories]
    median_stop = float(np.median(stops))
    left, right = int(np.floor(median_stop)), int(np.ceil(median_stop))
    quantiles = np.array([np.quantile([h[t] for h in histories if t < len(h)], [.25, .5, .75])
                          for t in range(right + 1)]).T
    trials = np.arange(left + 1, dtype=float)
    shown = quantiles[:, :left + 1]
    if median_stop > left:
        alpha = median_stop - left
        pair = np.maximum(quantiles[:, left:left + 2], 1e-300)
        end = np.exp((1 - alpha) * np.log(pair[:, 0]) + alpha * np.log(pair[:, 1]))
        trials = np.append(trials, median_stop)
        shown = np.column_stack((shown, end))
    return trials, shown, median_stop


def draw_target(axis, target):
    """Draw the target law as orange geometry."""
    from matplotlib.patches import Circle, Wedge
    face, edge = '#F28E2B', '#E87500'
    if target == 'banana':
        g1, g2 = np.meshgrid(np.linspace(-2.45, 2.45, 260), np.linspace(-1.85, 3.15, 260))
        z1 = g1 / 0.90
        z2 = (g2 - 0.34 * (z1 * z1 - 1.0)) / 0.30
        density = np.exp(-0.5 * (z1 * z1 + z2 * z2)) / (2.0 * np.pi * 0.90 * 0.30)
        axis.contourf(g1, g2, density, levels=(0.018, 0.045, 0.10, 0.20, 0.36, 0.52, 0.60),
                      colors=('#FDE7CD', '#FBD4A1', '#F8BF79', '#F5AA51', face, '#DE7000'),
                      alpha=0.78, zorder=1)
        axis.contour(g1, g2, density, levels=(0.045, 0.20), colors=edge,
                     linewidths=(0.42, 0.55), alpha=0.78, zorder=1.4)
    elif target == 'two_moons':
        radius = 1.15
        band = 1.15 * 0.055 * np.sqrt(-2.0 * np.log(0.1))
        for centre, start, stop in (((-0.575, -0.2875), 0.0, 180.0), ((0.575, 0.2875), 180.0, 360.0)):
            axis.add_patch(Wedge(centre, radius + band, start, stop, width=2.0 * band, facecolor=face,
                                 edgecolor=edge, linewidth=0.50, alpha=0.25, zorder=1))
            for angle in np.deg2rad((start, stop)):
                axis.add_patch(Circle((centre[0] + radius * np.cos(angle), centre[1] + radius * np.sin(angle)),
                                      band, facecolor=face, edgecolor='none', alpha=0.25, zorder=1))
            angles = np.deg2rad(np.linspace(start, stop, 320))
            axis.plot(centre[0] + radius * np.cos(angles), centre[1] + radius * np.sin(angles),
                      color=edge, linewidth=0.62, alpha=0.88, zorder=1.4)
    else:
        z90 = 1.6448536269514722
        inner, outer = 1.35 - z90 * 0.08, 1.35 + z90 * 0.08
        axis.add_patch(Wedge((0.0, 0.0), outer, 0.0, 360.0, width=outer - inner, facecolor=face,
                             edgecolor=edge, linewidth=0.55, alpha=0.25, zorder=1))
        axis.add_patch(Circle((0.0, 0.0), 1.35, fill=False, edgecolor=edge, linewidth=0.62,
                              alpha=0.88, zorder=1.4))


def figure(out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import ConnectionPatch, Rectangle

    plt.rcParams.update({'font.family': 'serif', 'font.size': 8.0, 'pdf.fonttype': 42, 'ps.fonttype': 42,
                         'axes.titlesize': 9.0, 'axes.labelsize': 8.0, 'xtick.labelsize': 7.0,
                         'ytick.labelsize': 7.0, 'legend.fontsize': 7.0, 'lines.linewidth': 1.4})
    rows = json.loads((out / 'results.json').read_text())['rows']
    particles = np.load(out / 'particles.npz')
    fig, axes = plt.subplots(3, 3, figsize=(7.5, 6.5), constrained_layout=True,
                             gridspec_kw={'hspace': 0.06, 'wspace': 0.08})
    for column, target in enumerate(TARGETS):
        group = {v: sorted((r for r in rows if r['target'] == target and r['variant'] == v),
                           key=lambda r: r['seed']) for v in VARIANTS}

        axis = axes[0, column]
        final, initial = particles[f'{target}-s0-full'], particles[f'{target}-s0-initial']
        limit = max(1.7, 1.04 * float(np.max(np.abs(np.vstack((samples(target)[1], initial, final))))))
        draw_target(axis, target)
        axis.scatter(final[:, 0], final[:, 1], s=1.0, color=COLORS['full'], alpha=0.86, linewidths=0, zorder=3)
        axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal', xlabel=r'$x_1$',
                 title=LABELS[target])
        if column == 0:
            axis.set_ylabel(r'$x_2$')
        else:
            axis.set_yticklabels([])

        axis = axes[1, column]
        axis.set_box_aspect(0.77)
        curves = []
        for variant in VARIANTS:
            color = COLORS[variant]
            trials, (lower, median, upper), stop = ragged_summary(
                [r['diagnostic_history'] for r in group[variant]], [r['iterations'] for r in group[variant]])
            curves.append((trials, median, color))
            axis.semilogy(trials, median, color=color)
            axis.fill_between(trials, lower, upper, color=color, alpha=0.13, linewidth=0)
            capped = all(r['termination'] == 'iteration_cap' for r in group[variant])
            offset = (4, 6) if variant == 'no_growth' else (-4, 6)
            axis.plot(stop, median[-1], marker='o', markersize=3.6, markeredgecolor='white',
                      markeredgewidth=0.55, markerfacecolor=color, linestyle='none', zorder=6)
            axis.annotate('cap' if capped else f'{stop:g}', xy=(stop, median[-1]), xytext=offset,
                          textcoords='offset points', color=color, fontsize=5.0, fontweight='semibold',
                          ha='left' if offset[0] > 0 else 'right', va='bottom',
                          bbox=dict(boxstyle='round,pad=0.12', facecolor='white', edgecolor='none', alpha=0.82),
                          zorder=7)
        axis.set_xlabel('trust-region iteration')
        if column == 0:
            axis.set_ylabel(r'holdout $\frac{1}{2}\mathrm{MMD}^2$')
        inset = axis.inset_axes([0.40, 0.52, 0.56, 0.43], zorder=7)
        for trials, median, color in curves:
            inset.semilogy(trials, median, color=color, linewidth=0.95)
        (x0, x1), (y0, y1) = ZOOM[target]
        inset.set(xlim=(x0, x1), ylim=(y0, y1), facecolor='white')
        inset.set_xticks((5, 10, 15, 20))
        inset.grid(alpha=0.20, linewidth=0.30)
        inset.tick_params(axis='both', which='both', labelsize=4.2, length=1.8, pad=1.0)
        inset.set_title('zoom', fontsize=4.8, pad=1.0)
        for spine in inset.spines.values():
            spine.set(color='#555555', linewidth=0.55)
        axis.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor='#555555',
                                 alpha=0.76, linewidth=0.55, zorder=5))
        for point, corner in (((x1, y1), (0, 0)), ((x1, y0), (1, 0))):
            fig.add_artist(ConnectionPatch(xyA=point, coordsA=axis.transData, xyB=corner,
                                           coordsB=inset.transAxes, axesA=axis, axesB=inset, color='#555555',
                                           linewidth=0.48, alpha=0.68, clip_on=False, zorder=6))

        axis = axes[2, column]
        axis.set_box_aspect(0.77)
        for position, variant in enumerate(VARIANTS):
            values = [r['diagnostic_history'][-1] for r in group[variant]]
            axis.scatter(position + np.linspace(-0.09, 0.09, len(values)), values, s=16,
                         color=COLORS[variant], alpha=0.72, linewidths=0)
            axis.plot([position - 0.18, position + 0.18], [np.median(values)] * 2, color='#222222',
                      linewidth=1.25)
        axis.set_yscale('log')
        axis.set_xticks(np.arange(3), ['full', 'Cauchy-only', 'no radius growth'], rotation=24, fontsize=5.9)
        if column == 0:
            axis.set_ylabel(r'final holdout $\frac{1}{2}\mathrm{MMD}^2$')

    for label, axis in zip('abcdefghi', axes.ravel()):
        axis.text(-0.12, 1.12, f'({label})', transform=axis.transAxes, fontsize=9, fontweight='bold',
                  va='bottom')
        axis.grid(alpha=0.18, linewidth=0.4)
    fig.savefig(out / 'fig-mmd-targets.pdf', bbox_inches='tight')
    fig.savefig(out / 'fig-mmd-targets.png', dpi=300, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out / 'fig-mmd-targets.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('command', choices=('run', 'figure'))
    parser.add_argument('--out', type=Path, default=Path('results/mmd-ablation'))
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    if args.command == 'run':
        run(args.out, args.workers)
    else:
        figure(args.out)


if __name__ == '__main__':
    main()
