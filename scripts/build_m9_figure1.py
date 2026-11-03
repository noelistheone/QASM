"""Editable Figure 1: measured inputs, learned fields, and physical query readout.

All raster inserts are the fixed first object and its archived model outputs.
Network blocks and the acquisition geometry are schematic. No model is fitted.
"""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle, Polygon, Circle, Arc
from matplotlib import colors

ROOT = Path(__file__).resolve().parents[1]
BLUE = '#2B73B5'
INK = '#22384B'
GRAY = '#7E8C99'
GREEN = '#268475'
ORANGE = '#D87D27'
PINK = '#B04B79'
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'svg.fonttype': 'none'})


def draw():
    data = np.load(ROOT / 'analysis/m9_figure1_v1/example.npz')
    fig = plt.figure(figsize=(7.2, 3.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, 104), ylim=(2.0, 44.5)); ax.axis('off')

    def text(x, y, label, size=7.5, color=INK, weight='normal', ha='center', va='center', **kw):
        return ax.text(x, y, label, fontsize=size, color=color, weight=weight,
                       ha=ha, va=va, linespacing=1.25, zorder=12, **kw)

    def box(x, y, w, h, fill='white', edge='#B9C8D4', lw=.65, radius=.6, **kw):
        p = FancyBboxPatch((x, y), w, h, boxstyle=f'round,pad=0,rounding_size={radius}',
                          facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1, **kw)
        ax.add_patch(p); return p

    def arrow(points, color=INK, lw=.85, dashed=False, head=True, zorder=8):
        pts = np.asarray(points)
        style = (0, (3, 2)) if dashed else '-'
        if len(pts) > 2:
            ax.plot(pts[:-1, 0], pts[:-1, 1], color=color, lw=lw, ls=style, zorder=zorder)
        if head:
            ax.add_patch(FancyArrowPatch(pts[-2], pts[-1], arrowstyle='-|>', mutation_scale=7,
                                        lw=lw, color=color, linestyle=style, zorder=zorder,
                                        shrinkA=0, shrinkB=0))
        else:
            ax.plot(pts[-2:, 0], pts[-2:, 1], color=color, lw=lw, ls=style, zorder=zorder)

    def tile(array, x, y, w, h=None, cmap='gray', vmin=None, vmax=None, edge='#AAB9C5'):
        if h is None: h = w
        ax.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax, origin='upper',
                  extent=(x, x+w, y, y+h), interpolation='nearest', aspect='auto', zorder=5)
        ax.add_patch(Rectangle((x, y), w, h, fill=False, edgecolor=edge, lw=.65, zorder=6))

    def panel(x, w, number, title, color, fill):
        box(x, 10.9, w, 33.1, fill, '#D5E0E7', lw=.7, radius=1)
        box(x+.8, 40.2, 3.5, 2.8, color, color, radius=.55)
        text(x+2.55, 41.6, number, 8, 'white', 'bold')
        text(x+5.2, 41.6, title, 8.3, color, 'bold', ha='left')

    def lock(x, y):
        ax.add_patch(Arc((x, y+1.25), 1.6, 2.2, theta1=0, theta2=180, color=GRAY, lw=.9, zorder=9))
        ax.add_patch(Rectangle((x-.95, y-.25), 1.9, 1.7, facecolor=GRAY, edgecolor=GRAY, lw=.4, zorder=9))
        ax.add_patch(Circle((x, y+.65), .18, facecolor='white', zorder=10))

    def unet(x, y, color, fill):
        """Four encoder scales and three decoder scales, matching the U-Net."""
        hs = [5.0, 4.0, 3.0, 2.0, 3.0, 4.0, 5.0]
        centers = [y+3.2, y+2.7, y+2.2, y+1.7, y+2.2, y+2.7, y+3.2]
        xs = [x+2.1*i for i in range(7)]
        for i in range(6):
            arrow([(xs[i]+1.35, centers[i]), (xs[i+1]-.1, centers[i+1])], color, .55)
        for i, (xx, cy, h) in enumerate(zip(xs, centers, hs)):
            bottom = cy-h/2
            ax.add_patch(Polygon([(xx, bottom+h), (xx+.38, bottom+h+.4),
                                  (xx+1.7, bottom+h+.4), (xx+1.32, bottom+h)],
                                 facecolor=colors.to_rgba(color, .32), edgecolor=color, lw=.45, zorder=5))
            ax.add_patch(Polygon([(xx+1.32, bottom), (xx+1.7, bottom+.4),
                                  (xx+1.7, bottom+h+.4), (xx+1.32, bottom+h)],
                                 facecolor=colors.to_rgba(color, .4), edgecolor=color, lw=.45, zorder=5))
            ax.add_patch(Rectangle((xx, bottom), 1.32, h, facecolor=fill, edgecolor=color, lw=.6, zorder=6))
        for i in [0, 1, 2]:
            level = y+7.0-.5*i
            a = (xs[i]+.65, centers[i]+hs[i]/2+.4)
            b = (xs[6-i]+.65, centers[6-i]+hs[6-i]/2+.4)
            arrow([a, (a[0], level), (b[0], level), b], color, .45, head=False, zorder=4)

    panel(.3, 19.7, 'a', 'Sparse input', BLUE, '#F7FAFC')
    panel(21.4, 40.0, 'b', 'Learn a shared spatial state', BLUE, '#F3F7FC')
    panel(62.8, 40.5, 'c', 'Query through physics', GREEN, '#F3F9F6')

    # (a) Sensor sketch and genuinely observed field coordinates.
    center = (6.7, 32.3)
    angle = np.linspace(0, 2*np.pi, 180, endpoint=False)
    visible = np.repeat(data['mask'], 6).astype(bool)
    ax.scatter(center[0]+4.3*np.cos(angle), center[1]+4.3*np.sin(angle), s=1.8,
               color='#CEDAE3', zorder=5, linewidths=0)
    ax.scatter(center[0]+4.3*np.cos(angle[visible]), center[1]+4.3*np.sin(angle[visible]),
               s=3.3, color=BLUE, zorder=6, linewidths=0)
    tile(data['object'], center[0]-2.25, center[1]-2.25, 4.5, edge='#A8B6C2')
    for a in [.45, 2.65, 4.25]:
        ax.plot([center[0], center[0]+4.0*np.cos(a)], [center[1], center[1]+4*np.sin(a)],
                color=BLUE, alpha=.22, lw=.5, zorder=4)
    text(6.7, 26.2, 'Measured subset', 6.7)
    obs = np.log10(np.maximum(data['observed_magnitude'], 1e-12))
    cmap = plt.colormaps['viridis'].copy(); cmap.set_bad('#E6EDF2')
    tile(obs, 12.9, 28.0, 5.1, 8.0, cmap=cmap)
    arrow([(11.0, 32.3), (12.5, 32.3)], BLUE, .7)
    text(15.45, 37.5, 'Field + mask', 6.4)
    arrow([(15.45, 27.3), (15.45, 24.0), (10.0, 24.0), (10.0, 22.3)], BLUE, .7)
    tile(data['feature'], 6.1, 14.1, 7.8, cmap='gray')
    text(10, 12.3, 'Iterative image feature', 6.7)

    # (b) One numerical feature feeds both networks. The frozen image also
    # conditions the factor network, exactly as in MaskPosteriorComponents.
    arrow([(14.25, 18.0), (20.7, 18.0), (20.7, 24.7), (22.0, 24.7)], BLUE)
    arrow([(22.0, 24.7), (22.0, 31.5), (24.1, 31.5)], GRAY)
    arrow([(22.0, 24.7), (22.0, 18.7), (24.1, 18.7)], BLUE)
    text(31.5, 37.7, 'Image estimator', 7.4, GRAY, 'bold')
    lock(40.0, 37.0)
    unet(24.3, 28.3, GRAY, '#DEE5EB')
    text(31.5, 24.0, 'Factor network', 7.4, BLUE, 'bold')
    unet(24.3, 15.5, BLUE, '#B9D5ED')
    text(31.4, 12.4, 'Trainable encoder–decoder', 6.5, BLUE)
    arrow([(38.9, 31.5), (43.3, 31.5)], GRAY)
    tile(data['mean'], 43.7, 27.5, 8.0)
    text(54.8, 31.5, 'Image', 7.0)
    arrow([(47.7, 27.0), (47.7, 25.8), (23.0, 25.8), (23.0, 18.7), (24.1, 18.7)], GRAY, .65)
    arrow([(38.9, 18.7), (42.3, 18.7)], BLUE)
    limit = float(np.max(np.abs(data['factors'][:3])))
    for j in range(3):
        tile(data['factors'][j], 42.8+5.15*j, 16.2, 4.7,
             cmap='RdBu_r', vmin=-limit, vmax=limit, edge='#84A9C6')
    text(57.8, 18.6, '…', 10, BLUE)
    text(50.0, 14.4, '8 spatial factors', 7.0, BLUE)
    ax.plot([58.7, 59.8, 59.8, 58.7], [35.6, 35.6, 14.3, 14.3], lw=.9, color=BLUE, zorder=8)
    text(50.1, 23.1, 'Shared across queries', 6.8, BLUE)
    arrow([(59.9, 25.0), (64.5, 25.0)], GREEN, 1.0)

    # (c) Query specification is explicit and separate from the learned state.
    text(83, 38.0, 'Set wavelength and angular band', 7.1, GREEN)
    box(65.0, 34.5, 16.4, 2.4, '#E6EFF8', '#A8C1D9', radius=.4)
    text(73.2, 35.7, 'Train: 4, 6, 8', 7.0, BLUE)
    box(83.0, 34.5, 18.0, 2.4, '#FFF0DF', '#E5BC90', radius=.4)
    text(92.0, 35.7, 'New: 5, 7, 10, 12', 6.7, ORANGE)
    # Both familiar and new queries select the same physical readout.
    arrow([(73.2, 34.4), (73.2, 34.0), (69.7, 34.0)], GREEN, .55, head=False)
    arrow([(92.0, 34.4), (92.0, 34.0), (69.7, 34.0)], GREEN, .55, head=False)
    arrow([(69.7, 34.0), (69.7, 32.9)], GREEN, .65)
    box(64.8, 14.2, 10.0, 18.3, '#E2F0E9', '#6BA597', lw=.85, radius=.7)
    text(69.8, 30.4, 'Known', 7.3, GREEN, 'bold')
    text(69.8, 28.5, 'propagation', 6.5, GREEN)
    t = np.linspace(0, 1, 100)
    for j, n in enumerate([1.5, 2.2, 3.0]):
        ax.plot(66+7.5*t, 25.2-2.7*j+.85*np.sin(2*np.pi*n*t), color=GREEN, lw=.9, alpha=.85, zorder=6)
    text(69.8, 16.1, 'Each field', 6.5, GREEN)

    e0 = np.log10(np.maximum(data['image_energy'], 1e-16))
    e1 = np.log10(np.maximum(data['factor_energy'], 1e-16))
    lo, hi = min(e0.min(), e1.min()), max(e0.max(), e1.max())
    tile(e0, 77.0, 27.4, 8.0, 4.8, 'magma', lo, hi, '#89ADA0')
    tile(e1, 77.0, 17.8, 8.0, 4.8, 'magma', lo, hi, '#89ADA0')
    text(81.0, 33.1, 'Image energy', 6.2, GREEN)
    text(81.0, 23.7, 'Factor energy', 6.2, GREEN)
    arrow([(75.0, 29.5), (76.7, 29.5)], GREEN, .65)
    arrow([(75.0, 20.2), (76.7, 20.2)], GREEN, .65)
    ax.add_patch(Circle((87.0, 25.0), 1.15, facecolor='white', edgecolor=GREEN, lw=.85, zorder=8))
    text(87, 25.1, '+', 10, GREEN)
    arrow([(85.2, 29.8), (87.0, 29.8), (87.0, 26.3)], GREEN, .7)
    arrow([(85.2, 20.2), (87.0, 20.2), (87.0, 23.7)], GREEN, .7)
    text(84.4, 13.0, 'Sum energies · average over angle', 6.5, GREEN)

    q = np.log10(np.maximum(data['prediction'], 1e-16))
    tile(q[:3].T, 90.5, 27.4, 9.8, 4.8, 'magma', q.min(), q.max(), BLUE)
    tile(q[3:].T, 90.5, 17.8, 9.8, 4.8, 'magma', q.min(), q.max(), ORANGE)
    text(95.4, 33.1, 'Supervised', 6.7, BLUE, 'bold')
    text(95.4, 23.7, 'Unseen queries', 6.5, ORANGE, 'bold')
    for j, v in enumerate([4, 6, 8]): text(90.5+(j+.5)*9.8/3, 26.2, str(v), 6.3, BLUE)
    for j, v in enumerate([5, 7, 10, 12]): text(90.5+(j+.5)*9.8/4, 16.4, str(v), 6.3, ORANGE)
    arrow([(88.3, 25), (89.2, 25), (89.2, 29.8), (90.2, 29.8)], GREEN, .7)
    arrow([(88.3, 25), (89.2, 25), (89.2, 20.2), (90.2, 20.2)], GREEN, .7)

    # A separate training lane makes it clear that the frozen image branch and
    # unseen-query outputs are not optimized at deployment.
    box(73.0, 3.8, 14.8, 4.6, '#F9EBF1', '#D2A0B6', radius=.6)
    text(80.4, 6.1, 'Response loss', 7.3, PINK, 'bold')
    arrow([(100.7, 29.8), (102.2, 29.8), (102.2, 9.5), (84.4, 9.5), (84.4, 8.6)], PINK, .8, True)
    text(95.5, 4.8, 'Training targets', 6.8, BLUE)
    arrow([(91.0, 6.1), (88.2, 6.1)], BLUE, .7)
    arrow([(72.6, 6.1), (31.2, 6.1), (31.2, 14.9)], PINK, .85, True)
    text(53.0, 7.9, 'Backpropagate through known physics', 7.1, PINK)
    text(3.0, 7.1, 'Frozen', 6.8, GRAY, ha='left'); lock(1.4, 6.35)
    box(10.9, 6.2, 1.8, 1.8, '#B9D5ED', BLUE, radius=.2)
    text(13.3, 7.1, 'Trainable', 6.8, BLUE, ha='left')
    text(1.0, 3.6, 'Solid: predictions   Dashed: training', 6.5, ha='left')

    path = ROOT / 'paper/assets/query_moment_overview'
    fig.savefig(path.with_suffix('.pdf'), bbox_inches='tight', pad_inches=.025)
    fig.savefig(path.with_suffix('.svg'), bbox_inches='tight', pad_inches=.025)
    fig.savefig(path.with_suffix('.png'), dpi=350, bbox_inches='tight', pad_inches=.025)
    plt.close(fig)
    print(json.dumps({'figure': str(path.relative_to(ROOT)), 'formats': ['pdf', 'svg', 'png'],
                      'data': 'analysis/m9_figure1_v1/example.npz'}))


if __name__ == '__main__':
    draw()
