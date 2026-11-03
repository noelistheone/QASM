"""Explicit visible-only view of the prepared real measurement store.

This is an access contract, not a security sandbox. Predictor code receives this
view, query coordinates and supplied calibration; scoring loads targets apart.
"""
from pathlib import Path
import json

import numpy as np

from src.born_scattering import sha256
from src.ptycho_measurement_interface import validated_ids


class VisiblePtychoMeasurements:
    def __init__(self, root, scan, fraction):
        self.root = Path(root)
        self.directory = self.root/'data/multibeamline_measurements_v1'/scan
        self.metadata = json.loads((self.directory/'manifest.json').read_text())
        if self.metadata['status'] != 'completed' or self.metadata['dataset'] != scan:
            raise ValueError('Incomplete or mismatched measurement store')
        label = {0.1:'0p1',0.25:'0p25',0.5:'0p5'}.get(fraction)
        if label is None:
            raise ValueError('Undeclared visible-frame fraction')
        geometry_path = self.root/self.metadata['geometry_path']
        if sha256(geometry_path) != self.metadata['geometry_sha256']:
            raise RuntimeError('Fixed scan geometry changed')
        with np.load(geometry_path,allow_pickle=False) as geometry:
            self.visible_ids = geometry[f'visible_{label}'].copy()
            self.query_ids = geometry['query_ids'].copy()
            self.group_ids = geometry[f'groups_{label}'].copy()
            self.group_center_ids = geometry[f'group_centers_{label}'].copy()
            self.coordinates = geometry['coordinates'].copy()
        if (np.intersect1d(self.visible_ids,self.query_ids).size
                or not np.all(np.diff(self.visible_ids)>0)
                or not np.isin(self.group_ids,self.visible_ids).all()):
            raise ValueError('Visible/query identity contract is inconsistent')
        self._intensity = np.load(self.directory/'intensity.npy',mmap_mode='r',allow_pickle=False)
        self.probe = np.load(self.directory/'probe.npy',allow_pickle=False)
        self.fraction = fraction
        self.local_group_indices = np.searchsorted(self.visible_ids,self.group_ids)
        self.local_group_center_indices = np.searchsorted(self.visible_ids,self.group_center_ids)
        if not np.array_equal(self.visible_ids[self.local_group_indices],self.group_ids):
            raise ValueError('Local and original group IDs disagree')

    def verify_storage(self):
        """Optional one-time integrity check, separate from input preparation."""
        for name, record in self.metadata['files'].items():
            if sha256(self.directory/name) != record['sha256']:
                raise RuntimeError(f'Measurement store changed:{name}')

    def frames(self, original_ids=None):
        ids = self.visible_ids if original_ids is None else validated_ids(original_ids,len(self.coordinates))
        if not np.isin(ids,self.visible_ids).all():
            raise ValueError('A nonvisible frame was requested by an input consumer')
        return np.asarray(self._intensity[ids]).copy()

    def centers(self, original_ids=None, calibrated=True):
        ids = self.visible_ids if original_ids is None else validated_ids(original_ids,len(self.coordinates))
        # Query positions are allowed, query intensities are not.
        if calibrated and not self.metadata['physical_origin_confirmed']:
            raise ValueError('LCLS physical patch origin remains unconfirmed')
        shift = self.metadata['parity_shift_xy'] if calibrated else np.zeros(2)
        return self.coordinates[ids]+shift

    def scales(self, source_rounding=False, batch=256):
        if batch < 1:
            raise ValueError('Positive streaming batch size required')
        total, squared, pixels = 0., 0., 0
        for first in range(0,len(self.visible_ids),batch):
            value = self.frames(self.visible_ids[first:first+batch]).astype(np.float64)
            if not np.isfinite(value).all() or (value < 0).any():
                raise ValueError('Visible intensity must be finite and nonnegative')
            if source_rounding:
                value=np.rint(value)
            total += float(value.sum())
            squared += float(np.square(value).sum())
            pixels += value.size
        if total <= 0 or squared <= 0:
            raise ValueError('All-dark input cannot determine a multiplicative scale')
        return dict(rms_scale=float(np.sqrt(pixels/squared)),
            inverse_mean_fluence=float(len(self.visible_ids)/total),
            frames=len(self.visible_ids),source_rounding=bool(source_rounding))
