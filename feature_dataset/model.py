"""Load only an official local DINOv3 ViT backbone + trusted native .pth weights.
No network calls, training heads, model downloads, or fallback random weights.
"""
from __future__ import annotations
from contextlib import nullcontext
import importlib
from pathlib import Path
import subprocess
import sys
import numpy as np
from .common import sha, source_hash

ARCHS = {'dinov3_vits16':384, 'dinov3_vits16plus':384, 'dinov3_vitb16':768,
         'dinov3_vitl16':1024}


class DINOEncoder:
    def __init__(self, repo, weights, arch='dinov3_vitb16', device='cuda:0', precision='bf16'):
        import torch
        self.torch=torch
        self.repo=Path(repo).expanduser().resolve(); self.weights=Path(weights).expanduser().resolve()
        if arch not in ARCHS: raise ValueError('This stage supports the listed DINOv3 ViT /16 backbones only.')
        if not (self.repo/'dinov3/hub/backbones.py').is_file():
            raise FileNotFoundError('Clone the official facebookresearch/dinov3 repository and pass --repo.')
        if not self.weights.is_file():
            raise FileNotFoundError(f'Local official .pth weights missing: {self.weights}. Request/download them through Meta first.')
        if self.weights.suffix.lower() not in ('.pth','.pt'):
            raise ValueError('Use the native official PyTorch .pth checkpoint, not Hugging Face model.safetensors.')
        if device.isdigit(): device='cuda:'+device
        self.device=torch.device(device)
        if self.device.type=='cuda':
            torch.cuda.init()
            torch.ones(1,device=self.device).sum().item()
            if precision=='bf16' and not torch.cuda.is_bf16_supported():
                raise ValueError('GPU lacks bf16 support. Use --precision fp16 or fp32 in a new feature run.')
            print('GPU:',torch.cuda.get_device_name(self.device),flush=True)
        elif self.device.type!='cpu': raise ValueError('Use cpu or cuda:N')
        if self.device.type=='cpu' and precision!='fp32': raise ValueError('CPU inference requires --precision fp32.')
        if precision not in ('fp32','fp16','bf16'): raise ValueError('Unsupported precision')
        torch.set_num_threads(2)
        if self.device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=False
        sys.path.insert(0,str(self.repo))
        module=importlib.import_module('dinov3.hub.backbones')
        if self.repo not in Path(module.__file__).resolve().parents:
            raise RuntimeError('Another dinov3 module is shadowing --repo. Start a fresh Python process.')
        # Build official architecture, then REQUIRE strict loading of official local weights.
        # Direct import avoids unrelated optional depth/text/training dependencies.
        model=getattr(module,arch)(pretrained=False)
        print('Loading official local checkpoint:',self.weights,flush=True)
        state=torch.load(self.weights,map_location='cpu',weights_only=True)
        if not isinstance(state,dict) or 'cls_token' not in state:
            raise ValueError('Expected a native DINOv3 backbone state_dict. No implicit checkpoint conversion is performed.')
        model.load_state_dict(state,strict=True); del state
        self.model=model.eval().requires_grad_(False).to(self.device)
        self.dim=ARCHS[arch];self.precision=precision
        if int(self.model.patch_size)!=16 or int(self.model.embed_dim)!=self.dim: raise ValueError('Backbone dimensions mismatch')
        try: commit=subprocess.check_output(['git','-C',str(self.repo),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
        except (OSError,subprocess.CalledProcessError): commit=None
        self.identity={'backbone':arch,'dimension':self.dim,'patch_size':16,
                       'checkpoint_sha256':sha(self.weights),'official_source_sha256':source_hash(self.repo/'dinov3'),
                       'git_commit':commit,'torch':str(torch.__version__),'precision':precision,
                       'checkpoint_format':'native_torch_state_dict_strict', 'training':False}

    def infer(self, batch):
        torch=self.torch
        x=torch.from_numpy(np.ascontiguousarray(batch,np.float32)).to(self.device)
        if x.ndim!=4 or x.shape[1]!=3 or x.shape[2]%16 or x.shape[3]%16:
            raise ValueError('Encoder input must be Bx3xHxW; H,W multiples of 16')
        autocast=(torch.autocast(device_type='cuda',dtype=torch.bfloat16 if self.precision=='bf16' else torch.float16)
                  if self.device.type=='cuda' and self.precision!='fp32' else nullcontext())
        with torch.inference_mode(),autocast:
            result=self.model.forward_features(x)
            patches=result['x_norm_patchtokens'];cls=result['x_norm_clstoken']
        expected=(len(x),(x.shape[2]//16)*(x.shape[3]//16),self.dim)
        if tuple(patches.shape)!=expected or tuple(cls.shape)!=(len(x),self.dim):
            raise RuntimeError(f'DINOv3 output contract mismatch: {tuple(patches.shape)} != {expected}')
        patches=patches.float().cpu().numpy();cls=cls.float().cpu().numpy()
        if not np.isfinite(patches).all() or not np.isfinite(cls).all():
            raise RuntimeError('Nonfinite DINO features. Retry with fp32 in a NEW output directory; do not keep corrupt features.')
        return patches,cls
