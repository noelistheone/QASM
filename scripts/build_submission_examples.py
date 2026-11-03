"""Render fixed first-index examples saved by the core inference replay."""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42})
born = np.load(ROOT/'reproduction/replay/born/born_state_example.npz')
real = np.load(ROOT/'reproduction/replay/broadband/broadband_query_example.npz')
fig, axes = plt.subplots(1,5,figsize=(11.8,2.6),constrained_layout=True)
mean = born['mean']
energy = np.sqrt(np.sum(born['factors']**2,axis=0))
for ax, value, title, cmap in zip(axes[:2],[mean,energy],
    ['Image estimate $m$','Spatial factor norm'],['gray','magma']):
    im=ax.imshow(value,cmap=cmap)
    ax.set_title(title);ax.set_axis_off()
    fig.colorbar(im,ax=ax,fraction=.046,pad=.03,location='bottom')
values=[np.log10(1+real[name]) for name in ['observed','native','learned']]
vmax=max(float(v.max()) for v in values)
for ax, value, title in zip(axes[2:],values,['Measured query','Native prediction','Learned prediction']):
    im=ax.imshow(value,cmap='magma',vmin=0,vmax=vmax)
    ax.set_title(title);ax.set_axis_off()
fig.colorbar(im,ax=list(axes[2:]),fraction=.046,pad=.03,location='bottom',label=r'$\log_{10}(1+\mathrm{counts})$')
for suffix in ['pdf','png']:
    fig.savefig(ROOT/'paper/assets'/('state_and_query_examples.'+suffix),bbox_inches='tight',dpi=180)
plt.close(fig)
