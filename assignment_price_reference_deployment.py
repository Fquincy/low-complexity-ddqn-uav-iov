"""Equal CPU graph compilation for the four frozen engineering references."""
import copy
from pathlib import Path
import warnings
import numpy as np
import torch
import onnx
import compiled_controller_inference as deployment
import deadline_screen_research_models as original
from controller_v3_fused_runtime import native


class NeuralCore(torch.nn.Module):
    def __init__(self,model):
        super().__init__()
        core=model.model.model if model.arm in original.original.existing.CORRECTED else model.model
        self.core=copy.deepcopy(core.net if model.base_arm in original.original.NEW_ARMS else core).eval()
    def forward(self,state):return self.core(state)


@native
def finish(raw,excluded,h,mode,scaled):
    scores=raw.reshape(-1).copy()
    if mode==0:
        # Wide additive values: maximize each unexcluded branch.
        a,q=original.fast.branch_values(scores,excluded,0)
    else:
        for i in range(30):
            if excluded[i]:scores[5*i]=np.float32(np.inf)
        if mode==1:a,q=original.fast.decoder.flat_decode(scores,1)
        else:
            raise ValueError('price needs observed state')
    return a[None].copy(),np.array([np.float32(q)*np.float32(h if scaled else 1.)],np.float32)


@native
def price_finish(raw,x,excluded,h,scaled):
    scores=raw.reshape(-1).copy()
    for i in range(30):
        if excluded[i]:scores[5*i]=np.float32(np.inf)
    a,q=original.fast.decoder.v3_decode(scores,x,True)
    return a[None].copy(),np.array([np.float32(q)*np.float32(h if scaled else 1.)],np.float32)


class Runtime:
    def __init__(self,model,path):
        if model.base_arm not in ('amortized_workload_price24','additive64_nonpositive','additive128_unbounded'):
            raise ValueError('unsupported reference')
        self.arm=model.base_arm;self.scaled=model.scaled;self.screened=model.screened
        self.path=Path(path)
        if self.path.exists():raise FileExistsError(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with warnings.catch_warnings(record=True) as caught,torch.inference_mode():
            warnings.simplefilter('always')
            torch.onnx.export(NeuralCore(model).eval(),torch.zeros(1,533),str(path),
                input_names=['state'],output_names=['raw'],opset_version=17,
                dynamo=False,do_constant_folding=True)
        self.warnings=sorted({str(w.message) for w in caught})
        onnx.checker.check_model(onnx.load(str(path)))
        self.session=deployment.session(path)

    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        excluded,h=original.fast.validate_and_screen(x) if self.screened else original.validate_unmasked(x)
        raw=self.session.run(None,{'state':x[None]})[0]
        if self.arm=='amortized_workload_price24':return price_finish(raw,x,excluded,h,self.scaled)
        return finish(raw,excluded,h,1 if self.arm=='additive64_nonpositive' else 0,self.scaled)
