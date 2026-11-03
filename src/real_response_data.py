"""Role-restricted data access for the frozen real-response study.

Input preparation never reads held-out measurements. Training and validation
labels have separate, explicit entry points. This module has no test scorer.
"""
from pathlib import Path
import json
import numpy as np
import torch

from src.born_scattering import sha256
from src.visible_ptycho_measurements import VisiblePtychoMeasurements
from src.global_ptycho_response import GlobalProbeOperator, observation_features

LABELS={.1:'0p1',.25:'0p25',.5:'0p5'}


def save_json(path,value):
    tmp=path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(value,indent=2,allow_nan=False))
    tmp.replace(path)


def read_contract(root):
    path=Path(root)/'protocols/m4_real_response_learning_v1.json'
    return json.loads(path.read_text())


def role_scans(contract,role):
    return [scan for values in contract['roles'][role].values() for scan in values]


def key(scan,fraction):
    return f'{scan}_{LABELS[fraction]}'


def training_target_ids(view):
    available=np.arange(len(view.coordinates),dtype=np.int64)
    return np.setdiff1d(available,np.union1d(view.visible_ids,view.query_ids))


@torch.no_grad()
def prepare_input(root,scan,fraction,device):
    """One completed LSQML input -> dimensionless, invariant network features."""
    root=Path(root);contract=read_contract(root)
    allowed=sum([role_scans(contract,r) for r in ['train','validation','validation_stress_not_for_checkpoint_selection','test']],[])
    if scan not in allowed or fraction not in contract['fractions']:
        raise ValueError('Task outside the frozen contract')
    source=root/'runs/m4_visible_lsqml_v1'/key(scan,fraction)
    summary=json.loads((source/'summary.json').read_text())
    if summary['status']!='completed' or summary['query_scoring_performed']:
        raise ValueError('Expected completed unscored baseline')
    checkpoint=next(c for c in summary['checkpoints'] if c['epoch']==1000)
    obj_path=source/'epoch1000_object.npy'
    if sha256(obj_path)!=checkpoint['files'][obj_path.name]:
        raise RuntimeError('Input object hash mismatch')
    view=VisiblePtychoMeasurements(root,scan,fraction)
    view.verify_storage()
    scales=view.scales();alpha=scales['rms_scale']
    probe_power=float(np.abs(view.probe.astype(np.complex128)).__pow__(2).sum())
    amplitude=float(np.sqrt(1/(scales['inverse_mean_fluence']*probe_power)))
    obj=torch.tensor(np.load(obj_path,allow_pickle=False)/amplitude,dtype=torch.complex128,device=device)
    probe=torch.tensor(view.probe.astype(np.complex128)*amplitude*np.sqrt(alpha),dtype=torch.complex128,device=device)
    measured=torch.tensor(view.frames().astype(np.float64)*alpha,dtype=torch.float64,device=device)
    operator=GlobalProbeOperator(probe,view.centers(),obj.shape)
    feature,coverage,gradient=observation_features(obj,operator,measured,fraction)
    # Same physical field in native units after scaling; check permitted frames.
    selected=torch.arange(min(17,len(view.visible_ids)),device=device)
    predicted=operator.fields(obj[None],selected)[:,0].abs().square()/alpha
    from src.ptychi_baseline_adapter import fourier_probe_intensity
    raw=fourier_probe_intensity(obj*amplitude,probe/(amplitude*np.sqrt(alpha)),view.centers()[:len(selected)],len(selected))
    normalization_error=float((predicted-raw).norm()/raw.norm())
    if normalization_error>1e-12:raise AssertionError('Physical normalization mismatch')
    directory=root/'data/m4_real_response_inputs_v1'/key(scan,fraction)
    directory.mkdir(parents=True,exist_ok=False)
    arrays=dict(features=feature.cpu().float().numpy(),base_object=obj.cpu().to(torch.complex64).numpy(),
        probe=probe.cpu().to(torch.complex64).numpy(),coverage=coverage.cpu().numpy(),
        visible_ids=view.visible_ids,query_ids=view.query_ids,
        centers=view.coordinates+np.asarray(view.metadata['parity_shift_xy'])[None],
        training_ids=training_target_ids(view))
    files={}
    for name,value in arrays.items():
        path=directory/f'{name}.npy';np.save(path,value,allow_pickle=False)
        files[path.name]=dict(sha256=sha256(path),shape=list(value.shape),dtype=str(value.dtype))
    record=dict(status='completed',scan=scan,fraction=fraction,alpha=alpha,object_amplitude_scale=amplitude,
        files=files,baseline_summary_sha256=sha256(source/'summary.json'),baseline_object_sha256=sha256(obj_path),
        contract_sha256=sha256(root/'protocols/m4_real_response_learning_v1.json'),
        native_normalization_relative_error=normalization_error,query_values_read=False,
        reference_pixels_read=False,visible_count=len(view.visible_ids),training_target_count=len(arrays['training_ids']),
        query_count=len(view.query_ids),feature_definition='nine invariant channels; float64 preparation, float32 storage',
        code_hashes={str(p.relative_to(root)):sha256(p) for p in [root/'src/real_response_data.py',root/'src/global_ptycho_response.py']})
    save_json(directory/'manifest.json',record)
    return record


class ResponseCase:
    def __init__(self,root,scan,fraction,device,role=None):
        self.root=Path(root);self.scan=scan;self.fraction=fraction
        contract=read_contract(root)
        if role is not None and (role not in ['train','validation'] or scan not in role_scans(contract,role)):
            raise PermissionError('Only declared training/validation label access is supported')
        directory=self.root/'data/m4_real_response_inputs_v1'/key(scan,fraction)
        self.manifest=json.loads((directory/'manifest.json').read_text())
        if self.manifest['status']!='completed':raise ValueError('Incomplete input cache')
        arrays={}
        for name,record in self.manifest['files'].items():
            path=directory/name
            if sha256(path)!=record['sha256']:raise RuntimeError(f'Changed input cache:{path}')
            arrays[path.stem]=np.load(path,allow_pickle=False)
        self.alpha=self.manifest['alpha'];self.device=torch.device(device)
        self.object=torch.tensor(arrays['base_object'],device=device)
        self.features=torch.tensor(arrays['features'],device=device)
        self.probe=torch.tensor(arrays['probe'],device=device)
        self.operator=GlobalProbeOperator(self.probe,arrays['centers'],self.object.shape)
        self.visible_ids=arrays['visible_ids'];self.query_ids=arrays['query_ids']
        self.training_ids=arrays['training_ids'];self.role=role
        self._raw=None
        self.label_ids=(self.training_ids if role=='train' else self.query_ids if role=='validation' else None)
        if role is not None:
            self._raw=np.load(self.root/'data/multibeamline_measurements_v1'/scan/'intensity.npy',mmap_mode='r',allow_pickle=False)

    def targets(self,ids):
        ids=np.asarray(ids,dtype=np.int64)
        if self.role is None or not np.isin(ids,self.label_ids).all():
            raise PermissionError('Target IDs are not allowed for this case role')
        value=np.asarray(self._raw[ids],dtype=np.float32)*self.alpha
        return torch.tensor(value,device=self.device)
