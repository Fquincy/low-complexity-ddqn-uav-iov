"""Versioned raw neural graph and fused FP32 heads/FP64 ADP decoder."""
import copy
from pathlib import Path
import warnings
import numpy as np
import torch
import onnx
import compiled_controller_inference as deployment
from controller_v3_fused_runtime import native,softplus
from assignment_price_deployment_runtime import finish
from assignment_price_fast_runtime import validate_mask


@native
def heads(raw,fixed):
    g=np.empty((30,5),np.float64);v=np.empty((2,5),np.float64)
    for i in range(30):
        for d in range(5):g[i,d]=np.float32(softplus(raw[5*i+d])/np.float32(30.))
    for i in range(2):
        for d in range(5):v[i,d]=np.float32(softplus(raw[150+5*i+d])/np.float32(2.))
    b=float(softplus(raw[160]));beta=np.zeros(5,np.float64);eta=np.empty(5,np.float64)
    if fixed:
        for d in range(1,5):beta[d]=1.
        for d in range(5):eta[d]=np.float32(.2)
    else:
        for d in range(1,5):beta[d]=softplus(raw[160+d])
        top=np.max(raw[165:170]);total=np.float32(0.);exp=np.empty(5,np.float32)
        for d in range(5):
            exp[d]=np.exp(np.float32(raw[165+d]-top));total=np.float32(total+exp[d])
        for d in range(5):eta[d]=np.float32(exp[d]/total)
    return g,v,b,beta,eta


@native
def fused_finish(raw,x,scale,mask,h,sweep,separable,fixed):
    g,v,b,beta,eta=heads(raw,fixed)
    return finish(x,scale,g,v,b,beta,eta,mask,h,sweep,separable)


class Runtime:
    def __init__(self,critic,path):
        self.arm=critic.arm;self.path=Path(path)
        if self.path.exists():raise FileExistsError(path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with warnings.catch_warnings(record=True) as caught,torch.inference_mode():
            warnings.simplefilter('always')
            torch.onnx.export(copy.deepcopy(critic.core.net).eval(),torch.zeros(1,533),str(path),
                input_names=['state'],output_names=['raw'],opset_version=17,dynamo=False,do_constant_folding=True)
        self.warnings=sorted({str(w.message) for w in caught})
        onnx.checker.check_model(onnx.load(str(path)));self.session=deployment.session(path)
        self.scale=critic.core.cycle_to_work.detach().numpy().copy()
        self.sweep=self.arm not in ('coupled_no_sweep','separable64')
        self.separable=self.arm=='separable64';self.fixed=self.arm=='fixed_price_plain_sweep'

    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        mask,h=validate_mask(x)
        raw=self.session.run(None,{'state':x[None]})[0][0]
        return fused_finish(raw,x,self.scale,mask,h,self.sweep,self.separable,self.fixed)
