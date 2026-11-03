"""Native probe-power gauge fixing with inverse object scaling.

Uses the existing public Pty-Chi power-constraint API. No vendor algorithm is
modified, and the gauge transformation preserves every single-slice exit wave.
"""
import torch
from src.broadband_calibration_solver_v2 import CalibrationSolver


class GaugeCalibrationSolver(CalibrationSolver):
    def __init__(self, measured, physical_xy, initial_object, initial_probe,
                 noise, optimize_probe, device, seed=2026110217):
        super().__init__(measured,physical_xy,initial_object,initial_probe,noise,optimize_probe,device,seed)
        if not optimize_probe:
            raise ValueError('This control is specifically for joint object/probe fitting')
        group=self.task.reconstructor.parameter_group
        power=float(group.probe.data.detach().abs().square().sum())
        assert power>0
        options=group.probe.options.power_constraint
        options.enabled=True;options.scale_object=True;options.probe_power=power
        group.probe.probe_power=power
        self.native_reference_probe_power=power

    @torch.no_grad()
    def gauge_statistics(self):
        group=self.task.reconstructor.parameter_group
        return dict(object_max_abs=float(self.object().abs().max()),
                    object_rms=float(self.object().abs().square().mean().sqrt()),
                    native_probe_power_ratio=float(group.probe.data.abs().square().sum()/self.native_reference_probe_power))
