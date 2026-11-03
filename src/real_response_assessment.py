"""Prospective real-measurement metrics and artifact checks, without label I/O.

Only the separate final scorer may open held-out intensities. These functions
can be checked with generated arrays while training/baselines are in flight.
"""
from pathlib import Path
import json
import numpy as np

from src.born_scattering import sha256


def detector_partitions(size=64):
    """Fixed disjoint annuli/sectors of the centered detector frequency grid."""
    frequency=np.fft.fftshift(np.fft.fftfreq(size))
    fy,fx=np.meshgrid(frequency,frequency,indexing='ij')
    radius=np.sqrt(fx**2+fy**2)
    edges=np.array([0.,.125,.25,.5,np.sqrt(.5)])
    annulus=np.searchsorted(edges[1:-1],radius,side='right')
    angle=np.mod(np.arctan2(fy,fx)+np.pi,2*np.pi)
    sector=np.minimum(np.floor(8*angle/(2*np.pi)).astype(np.int64),7)
    joint=annulus*8+sector
    # One central detector pixel has no physical azimuth. Its conventional
    # atan2(0,0) assignment is fixed, rather than changed after looking at data.
    weights=np.stack([(joint==i).astype(np.float64) for i in range(32)])
    if not np.array_equal(weights.sum(0),np.ones((size,size))):
        raise AssertionError('Detector cells do not partition every pixel')
    return dict(weights=weights,annulus=annulus,sector=sector,radial_edges=edges,
                counts=weights.sum((1,2)).astype(np.int64),coordinate_unit='cycles per sampled object-grid pixel')


def safe_ratio_root(numerator,denominator):
    if denominator==0:
        return None
    return float(np.sqrt(numerator/denominator))


class MeasurementMetrics:
    """Streaming, raw-unit sums; no target-dependent exclusion or rescaling."""
    def __init__(self,size=64):
        self.partitions=detector_partitions(size)
        self.size=size
        self.squared_error=0.;self.reference_square=0.
        self.amplitude_squared_error=0.;self.reference_sum=0.
        self.flux_squared_error=0.;self.flux_reference_square=0.
        self.predicted_total=0.;self.target_total=0.
        self.cell_squared_error=np.zeros(32)
        self.cell_reference_square=np.zeros(32)
        self.cell_target_sum=np.zeros(32)
        self.cell_prediction_sum=np.zeros(32)
        self.frames=0;self.zero_target_frames=0;self.zero_prediction_frames=0

    def add(self,prediction,target):
        p=np.asarray(prediction,dtype=np.float64);y=np.asarray(target,dtype=np.float64)
        if p.shape!=y.shape or p.ndim!=3 or p.shape[-2:]!=(self.size,self.size) or len(p)==0:
            raise ValueError('Mismatched or empty detector arrays')
        if not np.isfinite(p).all() or not np.isfinite(y).all() or (p<0).any() or (y<0).any():
            raise ValueError('Every prediction and target must be finite/nonnegative')
        self.squared_error+=float(np.square(p-y).sum())
        self.reference_square+=float(np.square(y).sum())
        self.amplitude_squared_error+=float(np.square(np.sqrt(p)-np.sqrt(y)).sum())
        self.reference_sum+=float(y.sum())
        pf=p.sum((1,2));yf=y.sum((1,2))
        self.flux_squared_error+=float(np.square(pf-yf).sum())
        self.flux_reference_square+=float(np.square(yf).sum())
        self.predicted_total+=float(pf.sum());self.target_total+=float(yf.sum())
        # Detector cell sums are correlated endpoints derived from the same I.
        cp=p.reshape(len(p),-1)@self.partitions['weights'].reshape(32,-1).T
        cy=y.reshape(len(y),-1)@self.partitions['weights'].reshape(32,-1).T
        self.cell_squared_error+=np.square(cp-cy).sum(0)
        self.cell_reference_square+=np.square(cy).sum(0)
        self.cell_target_sum+=cy.sum(0);self.cell_prediction_sum+=cp.sum(0)
        self.frames+=len(p);self.zero_target_frames+=int(np.count_nonzero(yf==0))
        self.zero_prediction_frames+=int(np.count_nonzero(pf==0))

    def result(self):
        if self.frames==0:raise ValueError('No measurements were scored')
        cells=[]
        for i in range(32):
            cells.append(dict(cell=i,annulus=i//8,sector=i%8,pixels=int(self.partitions['counts'][i]),
                relative_rmse=safe_ratio_root(self.cell_squared_error[i],self.cell_reference_square[i]),
                absolute_rmse=float(np.sqrt(self.cell_squared_error[i]/self.frames)),
                target_sum=float(self.cell_target_sum[i]),predicted_sum=float(self.cell_prediction_sum[i]),
                squared_error=float(self.cell_squared_error[i]),target_squared_norm=float(self.cell_reference_square[i]),
                zero_reference=bool(self.cell_reference_square[i]==0)))
        return dict(frames=self.frames,pixels_per_frame=self.size**2,
            intensity_relative_rmse=safe_ratio_root(self.squared_error,self.reference_square),
            intensity_absolute_rmse=float(np.sqrt(self.squared_error/(self.frames*self.size**2))),
            amplitude_relative_rmse=safe_ratio_root(self.amplitude_squared_error,self.reference_sum),
            fluence_relative_rmse=safe_ratio_root(self.flux_squared_error,self.flux_reference_square),
            intensity_squared_error=self.squared_error,intensity_target_squared_norm=self.reference_square,
            predicted_total=self.predicted_total,target_total=self.target_total,
            zero_target_frames=self.zero_target_frames,zero_prediction_frames=self.zero_prediction_frames,
            detector_cells=cells,
            intensity_unit='Original archived intensity; summed pixels are not asserted absolute photon flux.',
            coordinate_unit=self.partitions['coordinate_unit'])


def checked_json(path,status='completed'):
    value=json.loads(Path(path).read_text())
    if value.get('status')!=status:raise RuntimeError(f'Artifact not complete: {path}')
    return value


def checked_prediction(root,relative,digest,frames,size=64,chunk=256):
    """Validate an already generated prediction without loading any target."""
    path=Path(root)/relative
    if sha256(path)!=digest:raise RuntimeError(f'Changed prediction: {relative}')
    values=np.load(path,mmap_mode='r',allow_pickle=False)
    if values.shape!=(frames,size,size) or values.dtype!=np.float32:
        raise ValueError(f'Unexpected prediction shape/dtype: {relative}')
    for first in range(0,frames,chunk):
        part=values[first:first+chunk]
        if not np.isfinite(part).all() or (part<0).any():
            raise FloatingPointError(f'Invalid prediction: {relative}')
    return values


def aggregate_replicates(rows):
    """Average errors across training seeds, never ensemble their predictions."""
    grouped={}
    for row in rows:
        key=(row['family'],row['scan'],row['fraction'],row['role'])
        grouped.setdefault(key,[]).append(row)
    output=[]
    for (family,scan,fraction,role),values in sorted(grouped.items()):
        seeds=[v['seed'] for v in values]
        if len(seeds)!=len(set(seeds)):raise ValueError('Duplicate seed within a condition')
        for name in ['intensity_relative_rmse','amplitude_relative_rmse','fluence_relative_rmse']:
            numbers=np.array([v['metrics'][name] for v in values],dtype=np.float64)
            if not np.isfinite(numbers).all():raise ValueError('Undefined whole-scan metric')
        output.append(dict(family=family,scan=scan,fraction=fraction,role=role,seeds=seeds,
            seed_count=len(seeds),**{name:float(np.mean([v['metrics'][name] for v in values]))
                for name in ['intensity_relative_rmse','amplitude_relative_rmse','fluence_relative_rmse']},
            intensity_seed_sd=float(np.std([v['metrics']['intensity_relative_rmse'] for v in values],ddof=1)) if len(values)>1 else None))
    return output
