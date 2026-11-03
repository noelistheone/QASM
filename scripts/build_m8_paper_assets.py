"""Reproduce M8 explanatory figures and tables from frozen M3/M7 predictions.

No fitting, checkpoint selection, or object selection is performed here.
"""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'analysis/m8_presentation_v1'
PAPER = ROOT / 'paper'
BLUE, ORANGE, GREEN = '#2166ac', '#c56d27', '#168570'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})


def save(fig, name):
    fig.savefig(PAPER / 'assets' / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(PAPER / 'assets' / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def rank_one():
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.55))
    ax = axes[0]
    ax.add_patch(Ellipse((0, 0), 6, 2, facecolor='#eef1f5', edgecolor='#8493a1', lw=1.4))
    ax.axhline(0, color='#bbc4ce', lw=.8); ax.axvline(0, color='#bbc4ce', lw=.8)
    ax.annotate('', xy=(3, 0), xytext=(0, 0), arrowprops={'arrowstyle': '-|>', 'color': ORANGE, 'lw': 2})
    ax.annotate('', xy=(0, 1), xytext=(0, 0), arrowprops={'arrowstyle': '-|>', 'color': BLUE, 'lw': 2})
    ax.text(1.65, -.58, 'Image variance', color=ORANGE, ha='center', fontsize=8)
    ax.text(.18, 1.07, 'Query direction', color=BLUE, fontsize=8)
    ax.set(xlim=(-3.5, 3.5), ylim=(-1.6, 1.65), xlabel=r'$x_1$',
           ylabel=r'$x_2$', title=r'(a) Spatial covariance $\Sigma=\mathrm{diag}(9,1)$')
    ax.set_aspect('equal'); ax.set_xticks([-3, 0, 3]); ax.set_yticks([-1, 0, 1])
    ax.tick_params(labelsize=8); ax.title.set_fontsize(9)
    ax = axes[1]
    ax.bar([0, 1, 2], [0, 0, 1], color=['#8b97a3', ORANGE, BLUE], width=.58)
    ax.axhline(1, color=GREEN, lw=1.5, ls='--', label='True expected response')
    ax.scatter([0, 1], [0, 0], color=['#8b97a3', ORANGE], marker='_', s=220, linewidths=3, zorder=3)
    ax.set(xticks=[0, 1, 2], xticklabels=['Mean\nimage', 'Image-variance\nrank one', 'Query-aligned\nrank one'],
           ylabel='Response', ylim=(-.04, 1.35),
           title=r'(b) Query observes $x_2^2$')
    ax.tick_params(labelsize=8); ax.title.set_fontsize(9)
    ax.legend(frameon=False, loc='upper left', fontsize=7.5)
    fig.tight_layout(h_pad=1.0); save(fig, 'rank_one_query_geometry')


def overview():
    fig, ax = plt.subplots(figsize=(6.9, 2.2))
    ax.set(xlim=(0, 6.9), ylim=(0, 2.2)); ax.axis('off')
    def box(x, y, w, h, label, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=.025,rounding_size=.04',
                                   facecolor=color, edgecolor='#b5c4d0', lw=.8))
        ax.text(x+w/2, y+h/2, label, fontsize=8, ha='center', va='center', linespacing=1.5)
    def arrow(a, b, color='#526779'):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=8, lw=1.1, color=color))
    maps = np.load(ROOT / 'analysis/m7_mechanism_v1/fixed_index_maps.npz')
    theta = np.linspace(0, 2*np.pi, 30, endpoint=False)
    ax.plot(.48+.38*np.cos(theta), 1.12+.38*np.sin(theta), 'o', ms=2, color='#c6d2db')
    ax.plot(.48+.38*np.cos(theta[::6]), 1.12+.38*np.sin(theta[::6]), 'o', ms=3, color=BLUE)
    ax.imshow(maps['object'], cmap='gray', extent=(.25, .71, .89, 1.35))
    ax.text(.48, 1.72, 'Acquisition', fontsize=8.5, weight='bold', ha='center')
    ax.text(.48, .56, r'$y_\Omega,\;\Omega$', fontsize=10, ha='center')
    arrow((.9, 1.12), (1.07, 1.12))
    box(1.10, .83, 1.04, .60, 'Physical\nfeature', '#eef3f7')
    arrow((2.19, 1.27), (2.41, 1.62)); arrow((2.19, .99), (2.41, .59))
    box(2.45, 1.40, 1.03, .46, 'Frozen image\n'+r'$m$', '#eef3f7')
    box(2.45, .34, 1.03, .49, 'Factor network\n'+r'$L_\theta$', '#dbeaf7')
    arrow((2.97, 1.36), (2.97, .87))
    arrow((3.53, 1.62), (3.79, 1.33)); arrow((3.53, .59), (3.79, .98))
    box(3.84, .78, 1.22, .75, 'Shared state\n'+r'$[m,L_\theta]$', '#e5f1ec')
    arrow((5.11, 1.34), (5.34, 1.68)); arrow((5.11, .98), (5.34, .57))
    box(5.39, 1.37, 1.38, .57, 'Training waves\n4, 6, 8', '#e6eef9')
    box(5.39, .26, 1.38, .63, 'New waves\n5, 7, 10, 12', '#fff1df')
    ax.plot([6.08, 6.08, 2.30, 2.30], [1.99, 2.10, 2.10, .59], lw=1.0, color=BLUE)
    arrow((2.30, .59), (2.42, .59), BLUE)
    ax.text(4.24, 2.13, 'Response loss and physical gradients', color=BLUE, fontsize=8, ha='center')
    ax.text(4.45, .36, r'$\widehat q_B=\|Bm\|^2+\|BL_\theta\|_F^2$', fontsize=9, ha='center')
    ax.text(.12, .10, 'Infer the spatial state once', color='#344c60', fontsize=8)
    save(fig, 'query_moment_overview')


def mechanism():
    data = np.load(ROOT / 'analysis/m7_mechanism_v1/fixed_index_maps.npz')
    waves = json.loads((ROOT / 'analysis/m7_mechanism_v1/wavelength_errors.json').read_text())
    fig = plt.figure(figsize=(6.9, 2.03))
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 1.55], wspace=.40)
    for i, (key, title) in enumerate([('mean', 'Frozen image'), ('nppc_factor_norm', 'Image objective'),
                                     ('response_factor_norm', 'Response objective')]):
        ax = fig.add_subplot(gs[i]); im = ax.imshow(data[key], cmap='gray' if i == 0 else 'magma')
        ax.set_title(f'({chr(97+i)}) {title}', fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
        cb = fig.colorbar(im, ax=ax, orientation='horizontal', fraction=.08, pad=.09)
        cb.ax.tick_params(labelsize=7, length=2)
        cb.set_label('Contrast' if i == 0 else r'$\sqrt{\sum_j|\ell_j|^2}$', fontsize=8)
    ax = fig.add_subplot(gs[3]); order = np.argsort([4, 6, 8, 5, 7, 10, 12])
    xs = np.array([4, 6, 8, 5, 7, 10, 12])[order]
    for name, label, color in [('Mean image', 'Mean image', '#75808a'), ('NPPC', 'NPPC-style', ORANGE),
                               ('Response-trained', 'Response moments', BLUE)]:
        ax.plot(xs, np.array(waves[name])[order]*100, 'o-', label=label, color=color, ms=3, lw=1.2)
    for x in [4, 6, 8]: ax.axvline(x, color='#d2dbe4', lw=2, alpha=.6, zorder=0)
    ax.set_xticks([4, 6, 8, 10, 12]); ax.set_xlabel(r'$\lambda/\Delta x$', fontsize=8)
    ax.set_ylabel('Error (%)', fontsize=8, labelpad=1); ax.set_title('(d) All 100 objects', fontsize=8)
    ax.tick_params(labelsize=7, length=2); ax.grid(alpha=.12); ax.legend(frameon=False, fontsize=6.5)
    fig.subplots_adjust(bottom=.24, top=.85); save(fig, 'objective_mechanism')


def heatmap():
    target_path = ROOT / 'data/official_scattering_v1/scoring/targets35.npy'
    if target_path.exists():
        truth = np.load(target_path).astype(np.float64)
        assert truth.shape == (100, 35) and np.all(truth > 0)
        models = [
            ('NPPC-style', 'nppc', [20261120, 20261121, 20261122]),
            ('Joint image + response', 'joint', [20261117, 20261118, 20261119]),
            ('Response moments', 'primary_moment8', [20261126, 20261127, 20261128]),
        ]
        records, hashes, errors = [], [], []
        for label, family, seeds in models:
            replicates = []
            for noise in [20270111, 20270112, 20270113]:
                for seed in seeds:
                    if family == 'nppc':
                        path = ROOT / f'runs/m7_nppc_v1/noise{noise}/sample_{seed}_raw_budget0.npz'
                    else:
                        path = ROOT / f'runs/m3_official_assessment_v1/noise{noise}_x1/{family}_seed{seed}/mask0_budget0.npz'
                    p = np.load(path)['prediction'].astype(np.float64)
                    assert p.shape == truth.shape
                    replicates.append(np.abs(p / truth - 1))
                    hashes.append({'file': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            # All objects, all three checkpoints, and all three observation-noise draws.
            result = np.mean(replicates, axis=(0, 1)).reshape(7, 5)
            errors.append(result)
            for row, wave in enumerate([4, 6, 8, 5, 7, 10, 12]):
                for band in range(5):
                    records.append(dict(model=label, wavelength=wave, band=band, mre=float(result[row, band])))
    else:
        # The compact release ships exact aggregates and provenance rather than
        # duplicating all 27 prediction arrays. Plotting never selects a model.
        cached = json.loads((OUT / 'wavelength_band_errors.json').read_text())
        assert cached['objects'] == 100 and cached['receivers'] == 30
        records, hashes = cached['rows'], cached['source_hashes']
        models = [(name, None, None) for name in
                  ['NPPC-style', 'Joint image + response', 'Response moments']]
        lookup = {(r['model'], r['wavelength'], r['band']): r['mre'] for r in records}
        errors = [np.array([[lookup[name, wave, band] for band in range(5)]
                            for wave in [4, 6, 8, 5, 7, 10, 12]]) for name, _, _ in models]
    order = np.argsort([4, 6, 8, 5, 7, 10, 12])
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.05), sharey=True)
    vmax = max(float(e.max()) * 100 for e in errors)
    for i, (ax, e, (label, _, _)) in enumerate(zip(axes, errors, models)):
        im = ax.imshow(e[order].T * 100, cmap='YlGnBu', vmin=0, vmax=vmax, aspect='auto')
        for row in range(5):
            for col in range(7):
                value = e[order[col], row] * 100
                ax.text(col, row, f'{value:.1f}', ha='center', va='center', fontsize=7,
                        color='white' if value > .53 * vmax else '#132c40')
        ax.set(xticks=range(7), xticklabels=['4', '5', '6', '7', '8', '10', '12'],
               yticks=range(5), yticklabels=['0–10', '10–30', '30–60', '60–90', '90–180'],
               xlabel=r'Wavelength $\lambda/\Delta x$', title=f'({chr(97+i)}) {label}')
        for ix in [0, 2, 4]: ax.get_xticklabels()[ix].set_weight('bold')
        ax.tick_params(length=0, labelsize=7)
        ax.title.set_fontsize(8); ax.xaxis.label.set_fontsize(8)
    axes[0].set_ylabel('Angle (degrees)', fontsize=8)
    fig.subplots_adjust(left=.075, right=.91, bottom=.2, top=.86, wspace=.13)
    cb = fig.colorbar(im, cax=fig.add_axes([.93, .20, .015, .66]))
    cb.set_label('Relative error (%)', fontsize=8); cb.ax.tick_params(labelsize=7)
    save(fig, 'wavelength_band_errors')
    (OUT / 'wavelength_band_errors.json').write_text(json.dumps({'objects': 100, 'training_seeds_per_method': 3,
        'noise_draws': 3, 'receivers': 30, 'rows': records, 'source_hashes': hashes}, indent=2))
    with (OUT / 'wavelength_band_errors.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(records[0])); w.writeheader(); w.writerows(records)


def tables():
    source = json.loads((ROOT / 'analysis/m3_official_assessment_v1/summary.json').read_text())
    rows = [r for r in source['comparisons'] if r['multiplier'] == 1 and r['mask_order'] == 0
            and r['queries'] == 'new_combined' and r['right'] in ['ridge_budget', 'joint', 'direct_additive_original']]
    look = {(r['right'], r['budget']): r for r in rows}
    lines = [r'\begin{tabular}{lrrr}', r'\toprule', r'Receivers & vs. Ridge & vs. Joint & vs. Direct \\', r'\midrule']
    for b, n in zip([0, 2, 5, 10], [30, 42, 60, 90]):
        cells = []
        for model in ['ridge_budget', 'joint', 'direct_additive_original']:
            r = look[model, b]; lo, hi = [-100 * v for v in r['object_bootstrap95'][::-1]]
            cells.append(f'${lo:.2f},\,{hi:.2f}$')
        lines.append(str(n) + ' & ' + ' & '.join(cells) + r' \\')
    (PAPER / 'tables/paired_intervals.tex').write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular}', '']))
    (OUT / 'paired_intervals.json').write_text(json.dumps(rows, indent=2))
    costs = json.loads((ROOT / 'audit/m3_prediction_review_20261101/cost_review.json').read_text())
    lookup = {(r['model'], r['batch'], r['query_count']): r for r in costs['metrics'] if r['budget'] == 0}
    lines = [r'\begin{tabular}{lrrrr}', r'\toprule', r' & \multicolumn{2}{c}{Batch 1} & \multicolumn{2}{c}{Batch 16} \\',
             r'Method & 15 & 35 & 15 & 35 \\', r'\midrule']
    for model, name in [('ridge_budget', 'Ridge'), ('joint_seed20261117', 'Joint image + response'),
                        ('random_current_seed20261126', 'Response moments')]:
        values = [lookup[model, b, q]['end_to_end_device_input']['median_ms'] for b in [1, 16] for q in [15, 35]]
        lines.append(name + ' & ' + ' & '.join(f'{v:.2f}' for v in values) + r' \\')
    (PAPER / 'tables/query_runtime.tex').write_text('\n'.join(lines + [r'\bottomrule', r'\end{tabular}', '']))


def transfer():
    groups = json.loads((ROOT / 'analysis/m3_official_assessment_v1/summary.json').read_text())['groups']
    lookup = {(g['family'], g['budget'], g['queries']): g for g in groups
              if g['multiplier'] == 1 and g['mask_order'] == 0}
    models = [('primary_moment8', 'Response moments', '#1864a0'),
              ('joint', 'Joint', '#de8642'), ('direct_additive_original', 'Direct CNN', '#418e70'),
              ('ridge_budget', 'Ridge', '#8867a9')]
    fig, axes = plt.subplots(1, 3, figsize=(6.9, 2.02), constrained_layout=True)
    for ax, query, title in zip(axes, ['new_combined', 'interpolation', 'extrapolation'],
                                ['All 20 new queries', 'Interpolation (10)', 'Extrapolation (10)']):
        for family, label, color in models:
            values = [lookup[family, b, query] for b in [0, 2, 5, 10]]
            ys = np.array([v['mean'] for v in values])*100
            ax.plot([30, 42, 60, 90], ys, 'o-', color=color, label=label, ms=3,
                    lw=1.5 if family == 'primary_moment8' else 1.0)
            if family == 'primary_moment8':
                ci = np.array([v['object_bootstrap95'] for v in values])*100
                ax.fill_between([30, 42, 60, 90], ci[:, 0], ci[:, 1], color=color, alpha=.12)
        ax.set_xticks([30, 42, 60, 90]); ax.tick_params(labelsize=7, length=2)
        ax.set_xlabel('Observed receivers', fontsize=8); ax.set_title(title, fontsize=8)
        ax.grid(alpha=.14)
    axes[0].set_ylabel('Relative error (%)', fontsize=8)
    axes[0].legend(frameon=False, fontsize=6.5)
    save(fig, 'born_query_transfer')


if __name__ == '__main__':
    OUT.mkdir(exist_ok=True, parents=True)
    rank_one(); overview(); mechanism(); heatmap(); tables(); transfer()
