"""Matched CPU graph/native adapters for all eleven frozen original controls."""
import copy
from pathlib import Path
import warnings
import numpy as np
import torch
import onnx
import assignment_price_reference_deployment as base
from controller_v3_fused_runtime import native

class Core(torch.nn.Module):
    def __init__(self,model):
        super().__init__();self.bdq=model.base_arm=='bdq64_uniform_adapted'
        core=model.model.model if model.arm in base.original.original.existing.CORRECTED else model.model
        self.core=copy.deepcopy(core.net if model.base_arm in base.original.original.NEW_ARMS else core).eval()
    def forward(self,state):
        if self.bdq:
            h=self.core.trunk(state);v=self.core.value(h);a=self.core.advantage(h).reshape(-1,32,5)
            return v[:,:,None]+a-a.mean(2,keepdim=True)
        return self.core(state)

@native
def finish(raw,x,excluded,h,mode,scaled):
    scores=raw.reshape(-1).copy()
    if mode in (0,4):a,q=base.original.fast.branch_values(scores,excluded,1 if mode==4 else 0)
    else:
        for i in range(30):
            if excluded[i]:scores[5*i]=np.float32(np.inf)
        if mode in (2,3):a,q=base.original.fast.decoder.v3_decode(scores,x,mode==2)
        else:a,q=base.original.fast.decoder.flat_decode(scores,1 if mode==1 else 2)
    return a[None].copy(),np.array([np.float32(q)*np.float32(h if scaled else 1.)],np.float32)

class Runtime:
    def __init__(self,model,path):
        modes={'amortized_workload_price24':2,'amortized_no_price24':3,'plain24_nonpositive':5,
               'additive64_nonpositive':1,'additive128_unbounded':0,'bdq64_uniform_adapted':4}
        self.mode=modes[model.base_arm];self.scaled=model.scaled;self.screened=model.screened
        self.path=Path(path)
        if self.path.exists():raise FileExistsError(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with warnings.catch_warnings(record=True) as caught,torch.inference_mode():
            warnings.simplefilter('always')
            torch.onnx.export(Core(model).eval(),torch.zeros(1,533),str(path),input_names=['state'],output_names=['raw'],
                opset_version=17,dynamo=False,do_constant_folding=True)
        self.warnings=sorted({str(w.message) for w in caught})
        onnx.checker.check_model(onnx.load(str(path)));self.session=base.deployment.session(path)
    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        excluded,h=base.original.fast.validate_and_screen(x) if self.screened else base.original.validate_unmasked(x)
        raw=self.session.run(None,{'state':x[None]})[0]
        return finish(raw,x,excluded,h,self.mode,self.scaled)
