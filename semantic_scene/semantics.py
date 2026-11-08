"""Optional text-image category suggestions. Never rewrite geometry or reviewed labels."""
from __future__ import annotations
from pathlib import Path
import importlib.metadata
import hashlib
import json
import time
import numpy as np
from PIL import Image
from .common import (load_scene,read_json,write_json,write_npz,sha,digest,unit,checked,code_hash)


def decide_label(scores,categories,min_score=.22,min_margin=.025,min_votes=.67):
    scores=np.asarray(scores,float)
    if scores.ndim!=2 or scores.shape[0]==0 or scores.shape[1]!=len(categories) or not np.isfinite(scores).all():
        raise ValueError('Invalid score matrix')
    mean=scores.mean(axis=0);order=np.argsort(-mean,kind='stable');top=int(order[0])
    margin=float(mean[top]-mean[order[1]]) if len(order)>1 else 0.
    votes=float(np.mean(np.argmax(scores,axis=1)==top))
    reasons=[]
    if scores.shape[0]<2:reasons.append('fewer_than_two_views')
    if categories[top]['kind']=='unknown':reasons.append('unknown_category_ranked_first')
    if mean[top]<min_score:reasons.append('low_text_image_similarity')
    if margin<min_margin:reasons.append('ambiguous_category_margin')
    if votes<min_votes:reasons.append('inconsistent_view_votes')
    return {'suggested_label':None if reasons else categories[top]['label'],
            'suggested_kind':None if reasons else categories[top]['kind'],
            'state':'abstained' if reasons else 'unverified_suggestion','reasons':reasons,
            'top_score':float(mean[top]),'top_margin':margin,'view_vote_fraction':votes,
            'view_count':int(scores.shape[0]),'top_candidates':[{'label':categories[int(i)]['label'],
              'kind':categories[int(i)]['kind'],'cosine':float(mean[i])} for i in order[:5]],
            'score_is_probability':False,'semantic_label':None,'instance_identity_changed':False}


def validate_categories(cats):
    if not isinstance(cats,list) or len(cats)<2:raise ValueError('Use at least two categories')
    labels=[]
    for c in cats:
        if not isinstance(c,dict) or not all(isinstance(c.get(k),str) and c[k].strip() for k in ('label','kind','description')):
            raise ValueError('Each category needs label, kind and description strings')
        if c['kind'] not in ('structure','equipment','part','other','unknown'):raise ValueError('Invalid category kind')
        labels.append(c['label'])
    if len(set(labels))!=len(labels):raise ValueError('Category labels must be unique')
    return cats


def suggest(args):
    out,scene=load_scene(args.scene)
    cats=validate_categories(read_json(args.categories or Path(__file__).with_name('categories.json')))
    if args.views<2:raise ValueError('Use at least two source views per region')
    if not 0<=args.mask_weight<=1:raise ValueError('mask-weight must be 0..1')
    recipe={'scene_id':scene['scene_id'],'semantic_code_sha256':code_hash(),'model':args.model,'pretrained':str(args.pretrained),
            'categories':cats,'views':args.views,'mask_weight':args.mask_weight,'device':args.device,
            'min_score':args.min_score,'min_margin':args.min_margin,'min_votes':args.min_votes,
            'prompt_templates':['a photo of {description}.','a close-up photo of {description}.']}
    if Path(args.pretrained).is_file():recipe['weights_sha256']=sha(args.pretrained)
    recipe['recipe_hash']=digest(recipe)
    destination=out/'label_runs'/recipe['recipe_hash'][:16]
    destination.mkdir(parents=True,exist_ok=True)
    final=destination/'suggestions.json'
    if final.exists():
        old=read_json(final)
        if old['recipe']!=recipe:raise ValueError('Label recipe collision')
        checked(destination/'clip_vectors.npz',old['files']['clip_vectors.npz'])
        write_json(out/'semantic_suggestions.json',old)
        print('Reused category suggestions:',final);return old
    try:
        import torch
        import open_clip
    except ImportError as e:
        raise RuntimeError('Optional labeling needs open_clip_torch; build/view work without it. See README for constrained installation.') from e
    if args.device.startswith('cuda') and not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; fix driver or use --device cpu. Do not change the renderer.')
    started=time.perf_counter()
    model,_,preprocess=open_clip.create_model_and_transforms(args.model,pretrained=args.pretrained)
    model.eval();model=model.to(args.device);tokenizer=open_clip.get_tokenizer(args.model)
    # Hash loaded parameters for a reproducible identity even when the library downloads a tagged checkpoint.
    h=hashlib.sha256()
    for name,t in sorted(model.state_dict().items()):
        h.update(name.encode());a=t.detach().cpu().contiguous().numpy();h.update(str(a.dtype).encode());h.update(a.tobytes())
    model_sha=h.hexdigest()
    prompts=[template.format(description=c['description']) for c in cats for template in recipe['prompt_templates']]
    with torch.inference_mode():
        text=model.encode_text(tokenizer(prompts).to(args.device)).float()
        text=text/text.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        text=text.reshape(len(cats),len(recipe['prompt_templates']),-1).mean(dim=1)
        text=text/text.norm(dim=-1,keepdim=True).clamp_min(1e-8)
    records={};all_features=[];ids=[];offsets=[0]
    for j,brief in enumerate(scene['regions'],1):
        rid=brief['region_id'];folder=out/'regions'/rid
        stamp=read_json(folder/'complete.json')
        checked(folder/'region.json',stamp['files']['region.json'])
        reg=read_json(folder/'region.json');views=reg['views']
        vi=np.unique(np.linspace(0,len(views)-1,min(args.views,len(views))).round().astype(int))
        images=[];observations=[]
        for k in vi:
            view=views[int(k)];observations.append(view['observation_id'])
            for key in ('masked_crop','context_crop'):
                fn=view[key];checked(folder/fn,stamp['files'][fn])
                with Image.open(folder/fn) as im:images.append(preprocess(im.convert('RGB')))
        with torch.inference_mode():
            encoded=model.encode_image(torch.stack(images).to(args.device)).float()
            encoded=encoded/encoded.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            encoded=encoded.reshape(len(vi),2,-1)
            mix=args.mask_weight*encoded[:,0]+(1-args.mask_weight)*encoded[:,1]
            mix=mix/mix.norm(dim=-1,keepdim=True).clamp_min(1e-8)
            score=(mix@text.T).cpu().numpy()
            feature=mix.cpu().numpy()
        rec=decide_label(score,cats,args.min_score,args.min_margin,args.min_votes)
        rec.update(region_id=rid,observation_ids=observations,per_view_scores=score.tolist())
        records[rid]=rec;all_features.append(feature.astype(np.float32));ids.append(rid);offsets.append(offsets[-1]+len(feature))
        print(f'[{j}/{len(scene["regions"])}] {rid}: {rec["suggested_label"] or "UNKNOWN"} | top={rec["top_candidates"][0]["label"]} | margin={rec["top_margin"]:.3f}',flush=True)
    write_npz(destination/'clip_vectors.npz',vectors=np.concatenate(all_features),region_ids=np.array(ids),
              offsets=np.asarray(offsets,np.int64),text_vectors=text.cpu().numpy(),category_labels=np.array([c['label'] for c in cats]))
    result={'schema':'valmet.semantic_suggestions.v1','scene_id':scene['scene_id'],'recipe':recipe,
            'loaded_model_sha256':model_sha,'packages':{'torch':torch.__version__,'open_clip_torch':importlib.metadata.version('open_clip_torch')},
            'seconds':time.perf_counter()-started,'regions':records,
            'files':{'clip_vectors.npz':{'bytes':(destination/'clip_vectors.npz').stat().st_size,'sha256':sha(destination/'clip_vectors.npz')}},
            'notes':['Category ranking is vocabulary-dependent, not calibrated probabilities.',
                     'Unknown and ambiguous labels remain null. Suggestions never assign physical instance identity.',
                     'review.json is separate; no reviewed labels or geometry were changed.']}
    write_json(final,result);write_json(out/'semantic_suggestions.json',result)
    print('Saved:',final);return result
