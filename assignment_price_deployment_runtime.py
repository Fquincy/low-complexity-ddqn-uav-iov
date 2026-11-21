"""CPU graph heads plus native ADP workload/decoder; no training launcher."""
import copy
from pathlib import Path
import warnings
import numpy as np
import torch
from torch.nn import functional as F
import onnx
import compiled_controller_inference as deployment
from assignment_price_fast_runtime import validate_mask,evaluate
from controller_v3_fused_runtime import native


class NeuralHeads(torch.nn.Module):
    def __init__(self,critic):
        super().__init__();self.net=copy.deepcopy(critic.core.net).eval()
        self.fixed=critic.arm=='fixed_price_plain_sweep'
    def forward(self,state):
        raw=self.net(state);p=F.softplus(raw[:,:161 if self.fixed else 165])
        g=p[:,:150].reshape(30,5)/30;v=p[:,150:160].reshape(2,5)/2;b=p[:,160]
        if self.fixed:
            beta=state.new_ones((5,));beta[0]=0;eta=state.new_full((5,),.2)
        else:
            beta=F.pad(p[:,161:165],(1,0)).reshape(5);eta=F.softmax(raw[:,165:170],dim=1).reshape(5)
        return g.double(),v.double(),b.double(),beta.double(),eta.double()


@native
def finish(x,scale,g,v,b,beta,eta,mask,h,sweep,separable):
    w=np.empty((30,5),np.float64);q=np.zeros(5,np.float64);cycles=np.empty(30,np.float32)
    for i in range(30):
        cycles[i]=x[i*8+6]
        for d in range(5):w[i,d]=np.float32(cycles[i]*scale[d])
    for d in range(1,5):q[d]=np.float32(np.expm1(x[269+d]))
    a,cost=evaluate(g,v,b,w,q,beta,eta,mask,cycles,sweep,separable)
    return a[None].copy(),np.array([-h*cost],np.float64)


class Runtime:
    """Immutable exported snapshot. Precision matches FP32-head/FP64-cost design."""
    def __init__(self,critic,path):
        self.arm=critic.arm;self.path=Path(path)
        if self.path.exists():raise FileExistsError(self.path)
        self.path.parent.mkdir(parents=True,exist_ok=True)
        with warnings.catch_warnings(record=True) as caught,torch.inference_mode():
            warnings.simplefilter('always')
            torch.onnx.export(NeuralHeads(critic).eval(),torch.zeros(1,533),str(path),
                input_names=['state'],output_names=['g','v','b','beta','eta'],opset_version=17,
                dynamo=False,do_constant_folding=True)
        self.warnings=sorted({str(w.message) for w in caught})
        onnx.checker.check_model(onnx.load(str(path)))
        self.session=deployment.session(path)
        self.scale=critic.core.cycle_to_work.detach().numpy().copy()
        self.sweep=self.arm not in ('coupled_no_sweep','separable64')
        self.separable=self.arm=='separable64'

    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        mask,h=validate_mask(x)
        g,v,b,beta,eta=self.session.run(None,{'state':x[None]})
        return finish(x,self.scale,g,v,b[0],beta,eta,mask,h,self.sweep,self.separable)
