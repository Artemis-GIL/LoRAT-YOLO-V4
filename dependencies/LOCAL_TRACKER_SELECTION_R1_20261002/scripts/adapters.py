"""Local-only adapters. No GT, dataset path, or future visibility accepted."""
import os, sys, json, math
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]

def write_json(p,x):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n')

class MVLMLocal:
    def __init__(self,audit):
        repo=Path('/home/xu/zyx/_external/MVLM');sys.path.insert(0,str(repo))
        os.environ['TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD']='1'
        import clip
        from lib.test.parameter.mvlm import parameters
        from lib.test.tracker.mvlm import MVLM
        original_clip=clip.load; original_load=torch.nn.Module.load_state_dict
        records=[]
        def clip_load(name,*a,**kw):
            kw['download_root']=str(repo/'pretrained/clip');return original_clip(name,*a,**kw)
        def checked(module,state,*a,**kw):
            r=original_load(module,state,*a,**kw)
            records.append(dict(module=type(module).__name__,missing=list(r.missing_keys),unexpected=list(r.unexpected_keys)))
            if type(module).__name__=='MVLM':
                assert not r.unexpected_keys
                assert all(k.startswith('text_encoder.clip.') for k in r.missing_keys)
            return r
        clip.load=clip_load;torch.nn.Module.load_state_dict=checked
        prev=os.getcwd();os.chdir(repo)
        try:
            p=parameters('mvlm_TF',str(repo/'models/MVLM_TF.pth.tar'));p.debug=0;p.device='cuda';p.search_factor=3.
            self.t=MVLM(p,'uavantiuav')
        finally:
            clip.load=original_clip;torch.nn.Module.load_state_dict=original_load;os.chdir(prev)
        self.t.identity_scorer=None
        assert getattr(self.t.network,'temporal_shadow',None) is None
        self.meta=dict(model='MVLM_TF',search_size=224,search_factor=3.,global_enabled=False,
                       extra_dino=False,mamba=False,language=True,checkpoint_load=records)
        write_json(audit,self.meta)
    def initialize(self,image,box,language):
        self.t.initialize(image,dict(init_bbox=list(box),init_nlp=language))
        self.t.flag=False;self.t.state=list(box);self.t.kmem=None;self.t.kmvlm=None
    def track(self,image):
        old=list(self.t.state);x,y,w,h=old;side=math.ceil(math.sqrt(w*h)*3.)
        roi=[round(x+w/2-side/2),round(y+h/2-side/2),side,side]
        self.t.flag=False
        out=self.t.track(image)
        box=[float(x) for x in out['target_bbox']]
        return box,dict(roi=roi,score=float(self.t.prev_conf),requested_global=bool(self.t.flag),flag='LOCAL',search_pixels=224)

class GOTLocal:
    def __init__(self,audit):
        sys.path.insert(0,'/home/xu/zyx/GOT_REMOTE/GOT/pytracking')
        os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',GOTEDIT_USE_COMPILE='0')
        os.environ.pop('GOTEDIT_NET_DTYPE',None)
        from pytracking.features.net_wrappers import NetWithBackbone
        from pytracking.parameter.tomp.got_edit_378_dino_da3 import parameters
        from pytracking.tracker.tomp.tomp import ToMP
        import ltr.models.backbone as backbones
        da3_path='/home/xu/.cache/huggingface/hub/models--depth-anything--da3-large/snapshots/c54c26b16ec04d218e8d584ecf4bce082a9fcc20'
        original_da3=backbones.dinov2_DA3
        original_load=torch.nn.Module.load_state_dict;loads=[]
        def checked(module,state,*args,**kwargs):
            result=original_load(module,state,*args,**kwargs)
            loads.append(dict(module=type(module).__name__,keys=len(state),missing=list(result.missing_keys),unexpected=list(result.unexpected_keys)))
            return result
        backbones.dinov2_DA3=lambda checkpoint_path,device='cuda':original_da3(da3_path,device)
        torch.nn.Module.load_state_dict=checked
        self.cls=ToMP;self.parameters=parameters
        self.wrapper=NetWithBackbone('/home/xu/zyx/GOT_REMOTE/weights/GOT_Edit_378_Vast_DA3.tar',use_gpu=True)
        try:self.wrapper.initialize()
        finally:backbones.dinov2_DA3=original_da3;torch.nn.Module.load_state_dict=original_load
        assert loads and not loads[-1]['missing'] and not loads[-1]['unexpected']
        self.wrapper.net.eval();self.wrapper.net.show=0
        self.audit=audit
        self.meta=dict(model='GOT_Edit_378_Vast_DA3',sample_size=432,dino_input=378,feature_grid=27,
                       search_factor=5.,conf_threshold=.9,judge_action=False,temporal_edit=False,
                       native_local_scale_rescaling=True,extra_recovery=False,checkpoint_load=loads,da3_local_path=da3_path)
        write_json(audit,self.meta)
    def initialize(self,image,box,language):
        p=self.parameters();p.net=self.wrapper
        p.judge_action_enable=False;p.temporal_edit_enable=False;p.use_gt_box=False;p.plot_iou=False
        p.debug=0;p.visualization=False;p.device='cuda';p.output_not_found_box=False
        self.t=self.cls(p);self.t.features_initialized=True;self.t.visdom=None
        self.update_count=0;self.lastflag=None
        native_update=self.t.update_memory;native_localize=self.t.localize_target
        def update(*a,**kw):
            r=native_update(*a,**kw);self.update_count+=1;return r
        def localize(*a,**kw):
            r=native_localize(*a,**kw);self.lastflag=r[3];return r
        self.t.update_memory=update;self.t.localize_target=localize
        self.t.initialize(image,dict(init_bbox=list(box),object_ids=[None]))
        assert not getattr(self.t,'judge_gt',[])
    def track(self,image):
        prev=self.update_count;out=self.t.track(image)
        roi=self.t.search_area_box.detach().cpu().tolist()
        return [float(x) for x in out['target_bbox']],dict(roi=roi,score=float(out['object_presence_score']),
            flag=self.lastflag,reference_updated=self.update_count>prev,search_pixels=378)

class LoRATLocal:
    def __init__(self,audit):
        sys.path.insert(0,str(ROOT/'external/LoRAT'))
        from trackit.models.backbone.dinov2.builder import build_dino_v2_backbone
        from trackit.models.methods.LoRAT.lorat import LoRAT_DINOv2
        from trackit.models.methods.LoRAT.funcs.vit_backbone_freeze import freeze_vit_backbone_
        from trackit.models.methods.LoRAT.funcs.vit_lora_utils import enable_lora_
        from safetensors.torch import load_file
        vit=build_dino_v2_backbone('ViT-L/14',False,acc='default')
        base=torch.load('/home/xu/.cache/torch/hub/checkpoints/dinov2_vitl14_pretrain.pth',map_location='cpu',weights_only=True)
        vit.load_state_dict(base,strict=True);del base
        model=LoRAT_DINOv2(vit,(14,14),(27,27));freeze_vit_backbone_(model)
        enable_lora_(model,64,64,0.,False)
        state=load_file(str(ROOT/'weights/LoRAT_large-378.bin'))
        trainable={k for k,p in model.named_parameters() if p.requires_grad}
        assert trainable==set(state),(sorted(trainable-set(state)), sorted(set(state)-trainable))
        result=model.load_state_dict(state,strict=False)
        assert not result.unexpected_keys
        assert not (set(result.missing_keys)&trainable)
        self.model=model.cuda().eval()
        from trackit.runners.evaluation.common.siamfc_search_region_cropping_params_provider.simple import SiamFCCroppingParameterSimpleProvider
        from trackit.runners.evaluation.distributed.tracker_evaluator.components.post_process.box_with_score_map import PostProcessing_BoxWithScoreMap
        from trackit.core.utils.siamfc_cropping import get_siamfc_cropping_params,apply_siamfc_cropping,apply_siamfc_cropping_to_boxes,reverse_siamfc_cropping_params
        from trackit.runners.evaluation.distributed.tracker_evaluator.default.pipelines.utils.bbox_mask_gen import get_foreground_bounding_box
        from trackit.core.operator.numpy.bbox.utility.image import bbox_clip_to_image_boundary_
        self.Provider=SiamFCCroppingParameterSimpleProvider
        self.get_params=get_siamfc_cropping_params;self.crop=apply_siamfc_cropping
        self.map_boxes=apply_siamfc_cropping_to_boxes;self.reverse=reverse_siamfc_cropping_params
        self.maskbox=get_foreground_bounding_box;self.clip=bbox_clip_to_image_boundary_
        self.post=PostProcessing_BoxWithScoreMap(torch.device('cuda'),(27,27),(378,378),.45);self.post.start()
        self.mean=torch.tensor([.485,.456,.406],device='cuda')[:,None,None]
        self.std=torch.tensor([.229,.224,.225],device='cuda')[:,None,None]
        self.meta=dict(model='LoRAT_L378',template_size=196,search_size=378,search_factor=5.,template_factor=2.,
            window_penalty=.45,min_search_object_size=10,amp='bfloat16',lora_rank=64,lora_alpha=64,
            checkpoint_trainable_keys=len(trainable),missing_frozen_keys=len(result.missing_keys),unexpected=[],
            backbone_strict_loaded=True,extra_recovery=False,template_update=False)
        write_json(audit,self.meta)
    def tensor(self,image):return torch.from_numpy(np.ascontiguousarray(image)).permute(2,0,1).cuda().float()
    def normalize(self,x):return ((x/255.-self.mean)/self.std).unsqueeze(0)
    def initialize(self,image,box,language):
        b=np.asarray(box,dtype=float).copy();b[2:]+=b[:2]
        self.provider=self.Provider(5.,10);self.provider.initialize(b)
        params=self.get_params(b,2.,np.array([196,196]))
        z,self.image_mean,params=self.crop(self.tensor(image),np.array([196,196]),params,'bilinear',False)
        self.z=self.normalize(z)
        bb=self.maskbox(b,params,(14.,14.));self.clip(bb,np.array([14,14]))
        assert bb[2]>bb[0] and bb[3]>bb[1]
        self.mask=torch.zeros((1,14,14),dtype=torch.long,device='cuda');self.mask[:,bb[1]:bb[3],bb[0]:bb[2]]=1
    def track(self,image):
        params=self.provider.get(np.array([378,378]))
        x,_,params=self.crop(self.tensor(image),np.array([378,378]),params,'bilinear',False,self.image_mean)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model(self.z,self.normalize(x),self.mask)
        out=self.post(output)
        pred=self.map_boxes(out['box'].double().cpu().numpy(),self.reverse(params))[0]
        self.clip(pred,np.array([image.shape[1],image.shape[0]]))
        score=float(out['confidence'][0]);self.provider.update(score,pred,np.array([image.shape[1],image.shape[0]]))
        roi=self.map_boxes(np.array([0.,0.,378.,378.]),self.reverse(params));roi[2:]-=roi[:2]
        pred[2:]-=pred[:2]
        return pred.tolist(),dict(roi=roi.tolist(),score=score,flag='LOCAL',search_pixels=378,reference_updated=False)

ADAPTERS={'mvlm':MVLMLocal,'gotedit':GOTLocal,'lorat':LoRATLocal}
