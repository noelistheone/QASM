"""Matched standard LSQML calibration controls with a fixed or learned probe.

Author-fitted fields initialize this diagnostic. It cannot yield an independent
prediction score. No change is made to the frozen known-probe benchmark adapter.
"""
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

from src.ptychi_baseline_adapter import SOURCE, PINNED_COMMIT, checked_geometry, fourier_probe_intensity


def computational_centers(physical_xy):
    """Keep the integer object crop but reverse the fractional probe shift."""
    physical_xy=np.asarray(physical_xy,dtype=np.float64)
    return 2*np.rint(physical_xy)-physical_xy


class CalibrationSolver:
    def __init__(self, measured, physical_xy, initial_object, initial_probe,
                 noise, optimize_probe, device, seed=2026110217):
        measured=np.asarray(measured,dtype=np.float64)
        if noise not in ('amplitude_gaussian','poisson'):raise ValueError(noise)
        if measured.ndim!=3 or not np.isfinite(measured).all() or (measured<0).any():
            raise ValueError('Finite nonnegative counts required')
        centers=computational_centers(physical_xy)
        checked_geometry(centers,initial_object.shape,initial_probe.shape)
        assert len(centers)==len(measured) and tuple(measured.shape[-2:])==tuple(initial_probe.shape)
        assert np.iscomplexobj(initial_object) and np.iscomplexobj(initial_probe)
        assert np.isfinite(initial_object).all() and np.isfinite(initial_probe).all()
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=SOURCE,text=True).strip()
        assert commit==PINNED_COMMIT
        sys.path.insert(0,str(SOURCE/'src'))
        import ptychi.api as api
        from ptychi.api.task import PtychographyTask
        torch.cuda.set_device(torch.device(device))
        options=api.LSQMLOptions();rec=options.reconstructor_options
        rec.default_device=api.Devices.GPU;rec.default_dtype=api.Dtypes.FLOAT32
        rec.batch_size=32;rec.num_epochs=300;rec.random_seed=seed
        rec.noise_model=api.NoiseModels.GAUSSIAN if noise=='amplitude_gaussian' else api.NoiseModels.POISSON
        rec.allow_nondeterministic_algorithms=False
        rec.rescale_probe_intensity_in_first_epoch=False;rec.forward_model_options.pad_for_shift=0
        options.data_options.fft_shift=True
        options.object_options.pixel_size_m=1.;options.probe_options.pixel_size_m=1.
        options.object_options.optimizable=True;options.object_options.step_size=1.
        options.object_options.remove_object_probe_ambiguity.enabled=False
        options.object_options.determine_position_origin_coords_by=api.ObjectPosOriginCoordsMethods.SPECIFIED
        options.object_options.position_origin_coords=[-.5,-.5]
        options.probe_options.optimizable=bool(optimize_probe);options.probe_options.step_size=1.
        options.probe_options.power_constraint.enabled=False
        options.probe_options.center_constraint.enabled=False
        options.probe_position_options.optimizable=False
        options.opr_mode_weight_options.optimizable=False
        self.scale=1.0  # Native numerical floors now remain in photon-count units.
        self.n=measured.shape[-1];self.device=device;self.centers=centers
        self.task=PtychographyTask(options,
            diffraction_data=torch.tensor(measured*self.scale,dtype=torch.float32,device=device),
            object_data=torch.tensor(initial_object[None].copy(),dtype=torch.complex64,device=device),
            probe_data=torch.tensor(initial_probe[None,None].copy(),dtype=torch.complex64,device=device)*np.sqrt(self.scale/self.n**2),
            probe_position_x_px=torch.tensor(centers[:,0],dtype=torch.float32,device=device),
            probe_position_y_px=torch.tensor(centers[:,1],dtype=torch.float32,device=device))

    def object(self):return self.task.get_data('object')[0]

    def probe(self):return self.task.get_data('probe')[0,0]*np.sqrt(self.n**2/self.scale)

    @torch.no_grad()
    def predict(self):
        return fourier_probe_intensity(self.object(),self.probe(),self.centers,batch_size=64)

    @torch.no_grad()
    def parity(self):
        manual=self.predict()
        native=torch.cat([self.task.reconstructor.forward_model(
            torch.arange(i,min(i+64,len(self.centers)),device=self.device))
            for i in range(0,len(self.centers),64)])
        native=torch.fft.fftshift(native,dim=(-2,-1))/self.scale
        return float(torch.linalg.vector_norm(manual-native)/torch.linalg.vector_norm(native))

    def run_to(self,epoch):
        current=self.task.reconstructor.current_epoch
        if epoch<current:raise ValueError('Cannot rewind a calibration trajectory')
        if epoch>current:self.task.run(n_epochs=epoch-current)
