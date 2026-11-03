"""Figures and tables for the focused spatial-moment manuscript."""
from pathlib import Path
import csv, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

ROOT=Path(__file__).resolve().parents[1];PAPER=ROOT/'paper'
BLUE='#2166ac';ORANGE='#c56d27';GREEN='#168570';PURPLE='#8560a9';GRAY='#75808a'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
    'axes.spines.right':False,'pdf.fonttype':42,'ps.fonttype':42,'legend.fontsize':8})

def save(fig,name):
    fig.savefig(PAPER/'assets'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(PAPER/'assets'/f'{name}.png',dpi=180,bbox_inches='tight');plt.close(fig)

def overview():
    fig,ax=plt.subplots(figsize=(12.2,3.45));ax.set(xlim=(0,12.2),ylim=(0,3.45));ax.axis('off')
    def box(x,y,w,h,text,color):
        ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle='round,pad=.05,rounding_size=.09',
            facecolor=color,edgecolor='#c3cdd5',lw=1))
        ax.text(x+w/2,y+h/2,text,ha='center',va='center',linespacing=1.55,fontsize=10)
    def arrow(a,b,color='#536777',dashed=False):
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=12,lw=1.5,
            color=color,linestyle='--' if dashed else '-'))
    ax.text(.2,3.13,'Sparse acquisition',weight='bold',fontsize=11)
    center=np.array([1.05,1.92]);t=np.linspace(0,2*np.pi,30,endpoint=False)
    ax.plot(center[0]+.67*np.cos(t),center[1]+.67*np.sin(t),'o',ms=3,color='#d3dbe2')
    ax.plot(center[0]+.67*np.cos(t[::6]),center[1]+.67*np.sin(t[::6]),'o',ms=5,color=BLUE)
    maps=np.load(ROOT/'analysis/m7_mechanism_v1/fixed_index_maps.npz')
    image=maps['object']
    ax.imshow(image,cmap='gray',extent=(.68,1.42,1.55,2.29),zorder=0)
    ax.text(1.05,.95,r'$y_\Omega,\;\Omega$',ha='center',fontsize=12)
    arrow((1.92,1.96),(2.3,1.96))
    box(2.38,1.48,1.6,1.0,'Physical feature\nCG + mask','#eef3f7')
    arrow((4.05,2.21),(4.48,2.53));arrow((4.05,1.75),(4.48,1.34))
    box(4.55,2.18,1.65,.72,'Frozen mean  $m$','#edf1f5')
    box(4.55,.91,1.65,.84,r'Factor CNN  $L_\theta$','#dbeaf7')
    arrow((6.27,2.53),(6.73,2.19));arrow((6.27,1.33),(6.73,1.69))
    box(6.8,1.25,1.75,1.37,r'$M_\theta=mm^*+LL^*$'+'\nShared spatial state','#e5f1ec')
    arrow((8.62,2.15),(9.11,2.55));arrow((8.62,1.72),(9.11,1.09))
    box(9.18,2.15,2.73,.77,'Training queries  '+r'$B_{4,6,8}$'+'\nResponse loss','#e6eef9')
    box(9.18,.68,2.73,.88,'New queries  '+r'$B_{5,7,10,12}$'+'\nSame state, new propagation','#fff1df')
    ax.text(7.69,.69,r'$\widehat q_B=\|Bm\|^2+\|BL\|_F^2$',ha='center',fontsize=11)
    ax.plot([10.54,10.54,4.39],[3.01,3.31,3.31],color=BLUE,lw=1.4)
    ax.text(7.5,3.36,'Gradients through the physical readout',ha='center',fontsize=9,color=BLUE)
    # Route training gradient around, rather than through, the frozen mean.
    ax.plot([4.39,4.39],[3.31,1.34],color=BLUE,lw=1.25)
    arrow((4.39,1.34),(4.5,1.34),BLUE)
    ax.text(.2,.22,'Image-space energy selects large variance; response-space learning selects useful propagated energy.',
            fontsize=10,color='#344c60')
    save(fig,'query_moment_overview')

def mechanism():
    data=np.load(ROOT/'analysis/m7_mechanism_v1/fixed_index_maps.npz')
    waves=json.loads((ROOT/'analysis/m7_mechanism_v1/wavelength_errors.json').read_text())
    fig=plt.figure(figsize=(12,3.05));gs=fig.add_gridspec(1,4,width_ratios=[1,1,1,1.48],wspace=.32)
    for i,(key,title) in enumerate([('mean','Frozen mean image'),('nppc_factor_norm','Image-trained factors'),
                                   ('response_factor_norm','Response-trained factors')]):
        ax=fig.add_subplot(gs[i]);im=ax.imshow(data[key],cmap='gray' if i==0 else 'magma')
        ax.set_title(f'({chr(97+i)}) {title}',fontsize=10);ax.set_xticks([]);ax.set_yticks([])
        cb=fig.colorbar(im,ax=ax,orientation='horizontal',fraction=.06,pad=.06)
        cb.ax.tick_params(labelsize=7);cb.set_label('Contrast' if i==0 else r'$\sqrt{\sum_j |\ell_j|^2}$',fontsize=8)
    ax=fig.add_subplot(gs[3]);order=np.argsort([4,6,8,5,7,10,12]);xs=np.array([4,6,8,5,7,10,12])[order]
    for name,color in [('Mean image',GRAY),('NPPC',ORANGE),('Response-trained',BLUE)]:
        ax.plot(xs,np.array(waves[name])[order]*100,'o-',label='NPPC-style' if name=='NPPC' else name,color=color,ms=4,lw=1.6)
    for x in [4,6,8]:ax.axvline(x,color='#d2dbe4',lw=3,alpha=.5,zorder=0)
    ax.set(title='(d) Query prediction, all 100 objects',xlabel=r'Wavelength $\lambda/\Delta x$',
        ylabel='Mean relative error (%)',xticks=[4,5,6,7,8,10,12]);ax.grid(alpha=.12)
    ax.legend(frameon=False,fontsize=8);fig.subplots_adjust(bottom=.24,top=.87)
    save(fig,'objective_mechanism')

def tables():
    old=json.loads((ROOT/'analysis/m3_official_assessment_v1/summary.json').read_text())['groups']
    nppc=json.loads((ROOT/'analysis/m7_nppc_v1/summary.json').read_text())['groups']
    budgets=[0,2,5,10]
    lookup={(r['family'],r['budget']):r['mean'] for r in old
            if r['multiplier']==1 and r['mask_order']==0 and r['queries']=='new_combined'}
    lookup.update({(r['family'],r['budget']):r['mre'] for r in nppc if r['queries']=='new'})
    methods=[('image_only','Image-only readout'),('nppc_sample_raw','NPPC-style, sample scale'),
        ('nppc_fixed_raw','NPPC-style, fixed scale'),('ridge_budget','Budget-specific Ridge'),
        ('ridge_pooled','Shared-budget Ridge'),('direct_multiplicative_original','Direct CNN, multiplicative'),
        ('direct_additive_original','Direct CNN, additive'),('joint','Joint image + response'),
        ('matched_coherent1','Coherent correction'),('primary_moment8','Response moments (ours)')]
    def table(methods,look,scale=100):
        lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r'Method & 30 & 42 & 60 & 90 \\',r'\midrule']
        for family,name in methods:
            if family in ['primary_moment8','response_state']:lines.append(r'\midrule')
            values=[scale*look[family,b] for b in budgets]
            lines.append(name+' & '+' & '.join(f'{v:.2f}' for v in values)+r' \\')
        return '\n'.join(lines+[r'\bottomrule',r'\end{tabular}',''])
    (PAPER/'tables/born_primary.tex').write_text(table(methods,lookup))
    methods=[]
    for mode in ['sample','fixed']:
        for variant,label in [('raw','Native'),('rescaled','Rescaled'),('constant','Constant variance'),
                              ('rescaled_tail','Rescaled + tail'),('constant_tail','Constant + tail')]:
            methods.append((f'nppc_{mode}_{variant}',f'{label}, {mode}'))
    methods.append(('response_state','Response moments (ours)'))
    (PAPER/'tables/nppc_controls.tex').write_text(table(methods,lookup))
    real=list(csv.DictReader((ROOT/'analysis/m7_real_moments_v1/metrics.csv').open()))
    lines=[r'\begin{tabular}{lrrrr}',r'\toprule',r' & \multicolumn{2}{c}{Fixed probe} & \multicolumn{2}{c}{Joint probe} \\',
        r'Spatial predictor & Pixel & Pool 16 & Pixel & Pool 16 \\',r'\midrule']
    for kind,name in [('coherent','One coherent field'),('moment','Mean + four factors'),
        ('moment_no_factors','Same model, factors removed'),('direct','Direct intensity head')]:
        values=[np.mean([float(r['rmse']) for r in real if r['probe']==probe and r['kind']==kind and int(r['pool'])==pool])
            for probe in ['fixed','free'] for pool in [1,16]]
        lines.append(name+' & '+' & '.join(f'{v:.5f}' for v in values)+r' \\')
    (PAPER/'tables/real_moment_transfer.tex').write_text('\n'.join(lines+[r'\bottomrule',r'\end{tabular}','']))

def morphology():
    p=ROOT/'analysis/m7_morphology_v1/summary.json'
    if not p.exists():return
    rows=json.loads(p.read_text())['groups'];data=np.load(p.parent/'objects_and_targets.npz')
    look={(r['cohort'],r['family'],r['budget']):r for r in rows if r['queries']=='new'}
    fig=plt.figure(figsize=(11.8,3.0));gs=fig.add_gridspec(2,4,width_ratios=[.65,1.45,.65,1.45],wspace=.44,hspace=.10)
    for col,cohort in enumerate(['inclusions','perforated']):
        examples=np.flatnonzero(data['labels']==cohort)[:2]
        for row,index in enumerate(examples):
            ax=fig.add_subplot(gs[row,2*col]);ax.imshow(data['images'][index,0],vmin=0,vmax=1,cmap='gray')
            ax.set_xticks([]);ax.set_yticks([])
        ax=fig.add_subplot(gs[:,2*col+1])
        for family,name,color in [('primary_moment8','Response moments',BLUE),('joint','Joint model',GREEN),
            ('nppc_sample','NPPC-style',ORANGE),('ridge_budget','Ridge',PURPLE)]:
            rs=[look[cohort,family,b] for b in [0,2,5,10]]
            ax.plot([30,42,60,90],[100*r['mre'] for r in rs],'o-',label=name,color=color,ms=4)
        ax.set(title=cohort.capitalize()+' (100 objects)',xlabel='Observed receivers',ylabel='New-query error (%)',xticks=[30,42,60,90])
        ax.grid(alpha=.12)
        if col==0:ax.legend(frameon=False,fontsize=7)
    fig.subplots_adjust(bottom=.20,top=.89);save(fig,'morphology_transfer')
    lines=[r'\begin{tabular}{llrrrr}',r'\toprule',r'Cohort & Model & 30 & 42 & 60 & 90 \\',r'\midrule']
    for cohort in ['inclusions','perforated','combined']:
        for family in ['image_only','nppc_sample','nppc_fixed','ridge_budget','ridge_pooled',
            'direct_additive_original','direct_multiplicative_original','joint','matched_coherent1','primary_moment8']:
            label={'direct_additive_original':'Direct additive','direct_multiplicative_original':'Direct multiplicative',
                'matched_coherent1':'Coherent correction','primary_moment8':'Response moments'}.get(family,family.replace('_',' '))
            lines.append(cohort.capitalize()+' & '+label+' & '+' & '.join(f'{100*look[cohort,family,b]["mre"]:.2f}' for b in [0,2,5,10])+r' \\')
        lines.append(r'\midrule')
    (PAPER/'tables/morphology_all.tex').write_text('\n'.join(lines[:-1]+[r'\bottomrule',r'\end{tabular}','']))

if __name__=='__main__':overview();mechanism();tables();morphology()
