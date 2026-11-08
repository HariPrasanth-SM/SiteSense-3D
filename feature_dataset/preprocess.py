"""All transforms preserve aspect ratio and explicitly record pixel-edge affines."""
from __future__ import annotations
import math
import numpy as np
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)
PATCH = 16


def normalize_rows(a):
    a = np.asarray(a, np.float32)
    norms = np.linalg.norm(a, axis=-1, keepdims=True)
    ok = np.isfinite(a).all(axis=-1) & np.isfinite(norms[..., 0]) & (norms[..., 0] > 1e-8)
    result = np.zeros_like(a)
    np.divide(a, np.maximum(norms, 1e-8), out=result, where=np.expand_dims(ok, -1))
    return result, ok


def make_layout(h, w, max_side, square):
    if h <= 0 or w <= 0 or max_side < 16 or max_side % 16:
        raise ValueError('Image sizes positive; encoder side must be a positive multiple of 16.')
    scale = max_side / max(h, w)
    # Full frames are not enlarged; object crops deliberately are.
    if not square: scale = min(1.0, scale)
    rh = max(1, round(h*scale)); rw = max(1, round(w*scale))
    ch = max_side if square else int(math.ceil(rh/16)*16)
    cw = max_side if square else int(math.ceil(rw/16)*16)
    left = (cw-rw)//2; top = (ch-rh)//2
    return dict(source_hw=[h,w], resized_hw=[rh,rw], canvas_hw=[ch,cw],
                pad_left=left, pad_top=top, scale_xy=[rw/w,rh/h])


def rgb_tensor(rgb, layout):
    h,w = layout['source_hw']
    if rgb.shape != (h,w,3) or rgb.dtype != np.uint8: raise ValueError('Expected original uint8 RGB.')
    rh,rw = layout['resized_hw']; ch,cw = layout['canvas_hw']
    im = np.asarray(Image.fromarray(rgb).resize((rw,rh), Image.Resampling.BICUBIC), np.float32)/255.0
    canvas = np.broadcast_to(MEAN, (ch,cw,3)).copy()
    y,x = layout['pad_top'], layout['pad_left']; canvas[y:y+rh,x:x+rw] = im
    return np.ascontiguousarray(((canvas-MEAN)/STD).transpose(2,0,1))


def patch_weights(mask, layout):
    if mask.shape != tuple(layout['source_hw']): raise ValueError('Mask/layout mismatch')
    rh,rw = layout['resized_hw']; ch,cw = layout['canvas_hw']
    # BOX resampling preserves area coverage rather than nearest-neighbor labels.
    coverage = np.asarray(Image.fromarray(mask.astype(np.float32)).resize((rw,rh), Image.Resampling.BOX),np.float32)
    canvas = np.zeros((ch,cw),np.float32)
    y,x=layout['pad_top'],layout['pad_left'];canvas[y:y+rh,x:x+rw]=coverage
    return canvas.reshape(ch//16,16,cw//16,16).mean(axis=(1,3)).reshape(-1)


def full_frame(rgb, side=640):
    spec = make_layout(*rgb.shape[:2], side, False)
    spec['original_to_encoder_pixel_edges'] = [[spec['scale_xy'][0],0,spec['pad_left']],
        [0,spec['scale_xy'][1],spec['pad_top']],[0,0,1]]
    spec['patch_size'] = 16
    spec['token_order'] = 'row_major; no CLS or storage/register tokens'
    return rgb_tensor(rgb,spec), spec


def mask_crop(rgb, mask, size=256, pad_fraction=0.08):
    if mask.shape != rgb.shape[:2]: raise ValueError('RGB/mask alignment mismatch')
    y,x=np.nonzero(mask)
    if not len(x): return None
    h,w=mask.shape; x0,x1=int(x.min()),int(x.max())+1; y0,y1=int(y.min()),int(y.max())+1
    px=max(2,int(math.ceil((x1-x0)*pad_fraction)));py=max(2,int(math.ceil((y1-y0)*pad_fraction)))
    x0=max(0,x0-px);x1=min(w,x1+px);y0=max(0,y0-py);y1=min(h,y1+py)
    m=mask[y0:y1,x0:x1]; patch=rgb[y0:y1,x0:x1].copy()
    patch[~m] = np.rint(MEAN*255).astype(np.uint8)
    spec=make_layout(y1-y0,x1-x0,size,True)
    spec.update(crop_xyxy=[x0,y0,x1,y1], source_mask_pixels=int(mask.sum()), padding_fraction=pad_fraction,
                background='ImageNet mean RGB outside mask; original image unchanged')
    sx,sy=spec['scale_xy']
    spec['original_to_encoder_pixel_edges']=[[sx,0,spec['pad_left']-sx*x0],
        [0,sy,spec['pad_top']-sy*y0],[0,0,1]]
    return rgb_tensor(patch,spec), patch_weights(m,spec), spec


def weighted_pool(tokens, weights):
    """Tokens (P,D), weights (N,P); normalize tokens AND output descriptors."""
    tokens, tok_ok=normalize_rows(tokens)
    weights=np.asarray(weights,np.float32).copy()
    if weights.ndim==1: weights=weights[None]
    if weights.shape[1] != tokens.shape[0]: raise ValueError('Patch count mismatch; no silent token truncation.')
    weights[:,~tok_ok]=0
    mass=weights.sum(axis=1)
    effective=mass**2/np.maximum((weights**2).sum(axis=1),1e-12)
    out,ok=normalize_rows(weights @ tokens)
    ok &= mass > 1e-8; out[~ok]=0
    return out,ok,mass,effective
