"""Build manuscript figures and tables directly from completed result files."""
from pathlib import Path
import argparse
import csv
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / 'paper'
COLORS = ['#1864a0', '#de8642', '#418e70', '#8867a9', '#707780', '#be5770']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.spines.top': False,
    'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42,
    'axes.labelsize': 9, 'axes.titlesize': 10, 'legend.fontsize': 8})


def savefig(fig, name):
    fig.savefig(PAPER / 'assets' / (name + '.pdf'), bbox_inches='tight')
    fig.savefig(PAPER / 'assets' / (name + '.png'), dpi=180, bbox_inches='tight')
    plt.close(fig)


def overview():
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.set(xlim=(0, 12), ylim=(0, 3.5)); ax.axis('off')
    def box(x, y, w, h, title, body, color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.08,rounding_size=.10',
            facecolor=color,edgecolor='#b7c4ce',linewidth=1.1))
        ax.text(x+w/2,y+h-.28,title,ha='center',va='center',weight='bold',fontsize=10)
        ax.text(x+w/2,y+h/2-.13,body,ha='center',va='center',fontsize=9,linespacing=1.55)
    def arrow(x1,y1,x2,y2,color='#43596c',style='-'):
        ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=12,
            linewidth=1.4,color=color,linestyle=style))
    box(.15,1.05,2.15,1.45,'Finite measurements',r'$y_{\Omega}$, geometry'+'\nSparse and noisy input','#edf2f7')
    box(2.85,1.05,2.5,1.45,'Learn inside the state',r'$Z_{\theta}(y_{\Omega})$'+'\nShared spatial representation','#e6f1f7')
    box(6.0,1.05,2.6,1.45,'Differentiable Sandbox',r'$q_{B}=\|BZ_{\theta}\|_F^2$'+'\nKnown propagation + readout','#eaf3ed')
    box(9.3,1.05,2.5,1.45,'New physical queries','New wavelengths\nNew scan positions','#faf0e4')
    for a,b in [(2.4,2.75),(5.45,5.9),(8.7,9.2)]:arrow(a,1.75,b,1.75)
    ax.text(4.1,.58,'Moment factors  |  Coherent state updates',ha='center',fontsize=9,color='#43596c')
    ax.text(7.3,3.10,'Response loss on training queries',ha='center',fontsize=10,color='#1864a0')
    arrow(7.3,2.6,7.3,2.92,'#1864a0');arrow(7.05,2.92,4.1,2.92,'#1864a0');arrow(4.1,2.92,4.1,2.60,'#1864a0')
    ax.text(9.63,.35,'Change the operator, reuse the learned state',ha='center',fontsize=10,color='#8c602b')
    savefig(fig,'overview')


def born():
    report = json.loads((ROOT/'analysis/m3_official_assessment_v1/summary.json').read_text())
    groups = report['groups']
    lookup = {(g['family'],g['multiplier'],g['mask_order'],g['budget'],g['queries']):g for g in groups}
    budgets=[0,2,5,10];receivers=[30,42,60,90]
    names={'primary_moment8':'Response state (ours)','joint':'Joint image + response',
        'direct_additive_original':'Direct CNN, additive','direct_multiplicative_original':'Direct CNN, multiplicative',
        'ridge_budget':'Budget-specific Ridge','ridge_pooled':'Shared-budget Ridge',
        'image_only':'Image-only readout','matched_coherent1':'Coherent correction',
        'matched_moment1':'One factor','matched_moment2':'Two factors',
        'matched_moment4':'Four factors','matched_moment8':'Eight factors (matched)',
        'direct_additive_headrate':'Direct additive, tuned LR','direct_multiplicative_headrate':'Direct multiplicative, tuned LR'}
    methods=['primary_moment8','joint','direct_additive_original','ridge_budget']
    fig,axes=plt.subplots(1,3,figsize=(11,3.15),constrained_layout=True)
    for ax,query,title in zip(axes,['new_combined','interpolation','extrapolation'],
                              ['All 20 new queries','Interpolation (10 queries)','Extrapolation (10 queries)']):
        for method,color in zip(methods,COLORS):
            gs=[lookup[method,1.,0,b,query] for b in budgets]
            values=np.array([g['mean'] for g in gs])*100
            ci=np.array([g['object_bootstrap95'] for g in gs])*100
            ax.plot(receivers,values,'o-',color=color,label=names[method],lw=2 if method==methods[0] else 1.3,ms=4)
            if method==methods[0]:ax.fill_between(receivers,ci[:,0],ci[:,1],color=color,alpha=.12)
        ax.set(title=title,xlabel='Observed receivers (20 transmitters)',xticks=receivers)
        ax.grid(alpha=.14)
    axes[0].set_ylabel('Mean relative error (%)');axes[0].legend(frameon=False)
    savefig(fig,'born_query_transfer')
    main_methods=['image_only','ridge_budget','ridge_pooled','direct_multiplicative_original',
                  'direct_additive_original','joint','matched_coherent1','primary_moment8']
    def table(methods,query):
        lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & 30 & 42 & 60 & 90 \\',r'\midrule']
        for method in methods:
            nums=[100*lookup[method,1.,0,b,query]['mean'] for b in budgets]
            label=names[method].replace('_',r'\_')
            if method=='primary_moment8':lines.append(r'\midrule')
            lines.append(label+' & '+' & '.join(f'{v:.2f}' for v in nums)+r' \\')
        return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',''])
    (PAPER/'tables/born_primary.tex').write_text(table(main_methods,'new_combined'))
    for query in ['original','interpolation','extrapolation','new_combined']:
        (PAPER/'tables'/f'born_all_{query}.tex').write_text(table(list(names),query))
    fig,axes=plt.subplots(1,2,figsize=(7.2,2.75),constrained_layout=True)
    for method,color in zip(['matched_moment1','matched_moment2','matched_moment4','matched_moment8','matched_coherent1'],COLORS):
        axes[0].plot(receivers,[100*lookup[method,1.,0,b,'new_combined']['mean'] for b in budgets],
                     'o-',label=names[method],color=color,ms=3)
    axes[0].set(xlabel='Observed receivers',ylabel='New-query error (%)',title='Matched representation ablation',xticks=receivers)
    axes[0].legend(fontsize=6,frameon=False)
    for method,color in zip(methods,COLORS):
        noise=[.5,1.,2.]
        vals=[np.mean([lookup[method,s,m,b,'new_combined']['mean'] for m in [0,1] for b in budgets])*100 for s in noise]
        axes[1].plot(noise,vals,'o-',label=names[method],color=color,ms=4)
    axes[1].set(xlabel='Noise standard-deviation multiplier',ylabel='New-query error (%)',title='Noise and mask variation',xticks=[.5,1,2])
    for ax in axes:ax.grid(alpha=.14)
    savefig(fig,'born_ablation_robustness')
    (PAPER/'tables/born_source.json').write_text(json.dumps(dict(source='analysis/m3_official_assessment_v1/summary.json',
        objects=100,training_seeds=3,noise_seeds_nominal=3,primary='new_combined,multiplier1,mask0',
        groups=[g for g in groups if g['multiplier']==1 and g['mask_order']==0]),indent=2)+'\n')


def real():
    rows=list(csv.DictReader((ROOT/'analysis/m4_broadband_strengthened_v1/all_metrics.csv').open()))
    assert len(rows)==1080
    lookup={(r['probe'],r['method'],int(r['energy']),int(r['visible_count']),int(r['pool'])):r for r in rows}
    conditions=[(e,n) for e in [8,12] for n in [60,150,300]]
    initial=list(csv.DictReader((ROOT/'analysis/m4_broadband_transfer_v1/all_metrics.csv').open()))
    interpolation={(r['method'],int(r['energy']),int(r['visible_count']),int(r['pool'])):r for r in initial if r['method'] in ['mean_pattern','idw16']}
    def mean(probe,method,k):return np.mean([float(lookup[probe,method,e,n,k]['relative_intensity_rmse']) for e,n in conditions])
    methods=['native_1000','native_5000','uniform_unbounded3_1000','smooth_0.5_1000','smooth_1_1000',
             'smooth_2_1000','smooth_4_1000','poisson_gain_1000','l2_gain_1000',
             'complex_unbounded_2026110207_1000','complex_unbounded_2026110208_1000',
             'complex_unbounded_2026110207_5000','complex_unbounded_2026110208_5000']
    labels=['Native, 1000','Native, 5000','Uniform, +3','Smooth 0.5, +3','Smooth 1, +3',
            'Smooth 2, +3','Smooth 4, +3','Poisson gain','L2 gain',
            'Learned A, 1000+3','Learned B, 1000+3','Learned A, 5000+3','Learned B, 5000+3']
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Fixed probe} & \multicolumn{2}{c}{Joint probe} \\',
           r'Method & Pixels & $16\!\times\!16$ & Pixels & $16\!\times\!16$ \\',r'\midrule']
    for method,label in [('mean_pattern','Mean input pattern'),('idw16','16-neighbor interpolation')]:
        values=[np.mean([float(interpolation[method,e,n,k]['relative_intensity_rmse']) for e,n in conditions]) for k in [1,16]]
        lines.append(label+' & '+' & '.join(f'{v:.4f}' for v in values*2)+r' \\')
    for method,label in zip(methods,labels):
        values=[mean(p,method,k) for p in ['fixed','free'] for k in [1,16]]
        lines.append(label+' & '+' & '.join(f'{v:.4f}' for v in values)+r' \\')
    (PAPER/'tables/real_controls.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}','']))
    primary_labels=['Mean input pattern','16-neighbor interpolation','Native, 1000','Native, 5000',
                    'Uniform, +3','Smooth 4, +3','Poisson gain','Learned A, 1000+3','Learned B, 1000+3']
    primary=lines[:5]+[line for line in lines[5:] if line.split(' & ')[0] in primary_labels]
    (PAPER/'tables/real_primary.tex').write_text('\n'.join(primary+[r'\bottomrule',r'\end{tabular}','']))
    fig,axes=plt.subplots(1,2,figsize=(10.8,3.15),constrained_layout=True)
    selected=['native_1000','native_5000','mean_pattern','idw16','complex_unbounded_2026110207_1000','complex_unbounded_2026110208_1000']
    display=['Native 1000','Native 5000','Mean input pattern','16-neighbor interpolation','Learned A +3','Learned B +3']
    for ax,probe in zip(axes,['fixed','free']):
        for method,label,color in zip(selected,display,COLORS):
            vals=[float((interpolation[method,e,n,1] if method in ['mean_pattern','idw16'] else lookup[probe,method,e,n,1])['relative_intensity_rmse']) for e,n in conditions]
            for indices in [[0,1,2],[3,4,5]]:
                ax.plot(indices,[vals[i] for i in indices],'o-',label=label if indices[0]==0 else None,color=color,ms=4)
        ax.set(xticks=range(6),xticklabels=[f'E{e}/{n}' for e,n in conditions],ylabel='Relative intensity RMSE',
               title='Fixed calibrated probe' if probe=='fixed' else 'Joint probe with power normalization')
        ax.grid(alpha=.14)
    axes[0].legend(ncol=2,fontsize=7,frameon=False)
    savefig(fig,'real_control_comparison')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--part',choices=['born','real','all'],default='all');args=parser.parse_args()
    for folder in ['assets','tables']:(PAPER/folder).mkdir(parents=True,exist_ok=True)
    if args.part in ['born','all']:overview();born()
    if args.part in ['real','all']:real()
