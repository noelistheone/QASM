"""Pinned public PtychoPINN patch inference from explicit visible frames.

Only the reviewed network dependency closure is compiled from exact source AST
nodes. This avoids the source entry point's MLflow setup and automatic use of
all GPUs. Group selection and full-object assembly are separate operations.
"""
from pathlib import Path
from types import SimpleNamespace
import ast
import copy
import hashlib
import json
import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.ptycho_measurement_interface import intensity_scales

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'sources/resources/real_ptycho_expansion_20261101'
SOURCE = BASE/'multibeamline'
RUN = 'f6ce8d9583164c84955fc6209c340e04'
CHECKPOINT_SHA256 = '87070e64c2bc1c9d9bfec74a535fe560eada592454835fdc282100fd5ce2b5f5'


def reviewed_classes():
    namespace = dict(torch=torch, nn=nn, F=F, np=np, math=math, copy=copy,
        ModelConfig=SimpleNamespace, DataConfig=SimpleNamespace, TrainingConfig=SimpleNamespace)
    tree = {v['path']: v['sha'] for v in json.loads((SOURCE/'tree.json').read_text())['tree'] if v['type'] == 'blob'}
    hashes = {}
    for filename in ['model_attention.py', 'model.py']:
        relative = 'ptychopinn_torch/'+filename
        path = SOURCE/relative
        body = path.read_bytes()
        if hashlib.sha1(b'blob '+str(len(body)).encode()+b'\0'+body).hexdigest() != tree[relative]:
            raise RuntimeError('Published network source hash changed')
        nodes, after_encoder = [], False
        for node in ast.parse(body, filename=str(path)).body:
            if not isinstance(node, ast.ClassDef):
                continue
            if filename == 'model_attention.py' or not after_encoder or node.name in ['CombineComplex', 'IntensityScalerModule', 'PtychoPINN']:
                nodes.append(node)
            if node.name == 'Autoencoder':
                after_encoder = True
            if node.name == 'PtychoPINN':
                break
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
        hashes[relative] = hashlib.sha256(body).hexdigest()
    return namespace, hashes


class VisiblePtychoPINN:
    def __init__(self, device='cpu'):
        self.device = torch.device(device)
        namespace, hashes = reviewed_classes()
        folder = BASE/'all_model_metadata'/RUN/'params'
        self.model_config = SimpleNamespace(**json.loads((folder/'ModelConfig_params').read_text()))
        self.data_config = SimpleNamespace(**json.loads((folder/'DataConfig_params').read_text()))
        if self.model_config.intensity_scale_trainable:
            raise RuntimeError('This artifact adapter expects a fixed external intensity scale')
        self.network = namespace['Autoencoder'](self.model_config, self.data_config).to(self.device)
        checkpoint = BASE/'baseline_artifacts'/RUN/'latest_checkpoint.pth'
        digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
        if digest != CHECKPOINT_SHA256:
            raise RuntimeError('Unexpected public weight artifact')
        weights = torch.load(checkpoint, map_location='cpu', weights_only=True)['state_dict']
        prefix = 'model.autoencoder.'
        if not all(k.startswith(prefix) for k in weights):
            raise RuntimeError('Checkpoint has unhandled parameters')
        self.network.load_state_dict({k[len(prefix):]: v for k, v in weights.items()}, strict=True)
        self.network.eval().requires_grad_(False)
        self.combine = namespace['CombineComplex']().to(self.device)
        # The source scaler is a plain helper, not nn.Module; this artifact's
        # disabled trainable scale leaves it with no device-resident state.
        self.source_scaler = namespace['IntensityScalerModule'](self.model_config)
        self.source_forward = namespace['PtychoPINN'].forward_predict
        self.metadata = dict(run_id=RUN, checkpoint_sha256=digest, source_hashes=hashes,
            trainable_parameters=0, model_parameters=sum(p.numel() for p in self.network.parameters()),
            input='Rounded measured intensity times visible-only RMS scale; no square root',
            output='Four complex64 object patches per supplied four-frame group',
            common_input_adapter=True,
            scope='Public weights under explicit visible groups. Not exact source grouping/assembly or a new-source training claim.')

    @torch.no_grad()
    def predict_patches(self, visible_intensity, groups, batch_size=32):
        values = np.asarray(visible_intensity)
        if values.ndim != 3 or tuple(values.shape[1:]) != (self.data_config.N, self.data_config.N):
            raise ValueError('Visible frames must have the artifact detector shape')
        groups = np.asarray(groups)
        if groups.ndim != 2 or groups.shape[1] != self.data_config.C or not np.issubdtype(groups.dtype, np.integer):
            raise ValueError('Expected integer four-channel local visible-frame IDs')
        if len(groups) == 0 or groups.min() < 0 or groups.max() >= len(values) or batch_size < 1:
            raise ValueError('Invalid group indices or inference batch size')
        if any(len(set(row)) != self.data_config.C for row in groups.tolist()):
            raise ValueError('Each group must contain four distinct visible measurements')
        scales = intensity_scales(values, source_rounding=True)
        rounded = np.rint(values).astype(np.float32)
        result = []
        for start in range(0, len(groups), batch_size):
            inputs = torch.tensor(rounded[groups[start:start+batch_size]], device=self.device)
            amplitude, phase = self.network(inputs*scales['rms_scale'])
            result.append(self.combine(amplitude, phase).cpu())
        patches = torch.cat(result)
        if not bool(torch.isfinite(patches).all()):
            raise FloatingPointError('Nonfinite public network output')
        return patches, dict(scales=scales, groups=len(groups), distinct_channels=self.data_config.C,
                             checkpoint_sha256=self.metadata['checkpoint_sha256'])
