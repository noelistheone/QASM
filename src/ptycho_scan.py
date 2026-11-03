"""Differentiable, scan-level coherent diffraction with explicit pixel geometry.

Coordinates are patch centers in object pixels, ordered (x,y). FFT is unitary
and shifted on the detector. No physical photon calibration is inferred.
The extraction convention is independently checked against the public Run1084
workflow. PyTorch bilinear interpolation gives a piecewise smooth coordinate
map; derivatives at integer interpolation knots are not unique.
"""
import torch
import torch.nn.functional as F


def extract_patches(obj, centers, patch_shape):
    """obj: complex (B,H,W); centers: (F,2) or (B,F,2)."""
    if obj.ndim!=3 or not obj.is_complex():
        raise ValueError('Expected complex (batch,height,width) object')
    if centers.ndim==2:
        centers=centers[None].expand(len(obj),-1,-1)
    if centers.ndim!=3 or centers.shape[0]!=len(obj) or centers.shape[-1]!=2:
        raise ValueError('Expected per-batch (x,y) frame coordinates')
    ph,pw=patch_shape
    _,h,w=obj.shape
    if min(h,w)<2:
        raise ValueError('Object dimensions must exceed one pixel')
    yy,xx=torch.meshgrid(torch.arange(ph,device=obj.device,dtype=obj.real.dtype)-ph//2,
                        torch.arange(pw,device=obj.device,dtype=obj.real.dtype)-pw//2,indexing='ij')
    centers=centers.to(device=obj.device,dtype=obj.real.dtype)
    gx=xx[None,None]+centers[:,:,0,None,None]
    gy=yy[None,None]+centers[:,:,1,None,None]
    grid=torch.stack([2*gx/(w-1)-1,2*gy/(h-1)-1],dim=-1)
    grid=grid.reshape(len(obj),-1,pw,2)
    channels=torch.stack([obj.real,obj.imag],dim=1)
    patches=F.grid_sample(channels,grid,mode='bilinear',padding_mode='zeros',align_corners=True)
    result=torch.complex(patches[:,0],patches[:,1])
    return result.reshape(len(obj),centers.shape[1],ph,pw)


def field(obj,probe,centers):
    if probe.ndim==2:
        probe=probe[None,None]
    elif probe.ndim==3:
        probe=probe[:,None]
    else:
        raise ValueError('Probe must have (H,W) or (B,H,W) shape')
    patches=extract_patches(obj,centers,probe.shape[-2:])
    return torch.fft.fftshift(torch.fft.fft2(patches*probe,norm='ortho'),dim=(-2,-1))


def intensity(obj,probe,centers,gain=1.0):
    z=field(obj,probe,centers)
    if torch.is_tensor(gain):
        if bool((gain<=0).any()):raise ValueError('Gain must be positive')
    elif gain<=0:
        raise ValueError('Gain must be positive')
    return gain*z.abs().square()


def amplitude(obj,probe,centers,gain=1.0,epsilon=0.0):
    return (intensity(obj,probe,centers,gain)+epsilon**2).sqrt()
