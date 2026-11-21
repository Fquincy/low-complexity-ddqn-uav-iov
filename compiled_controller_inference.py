"""Inference-only ONNX adapter for the unchanged eight engineering models.

Training and target evaluation stay on the original PyTorch implementations.
Fixed batch-one FP32 deployment graphs require re-export after weight changes.
"""
import time,warnings
from pathlib import Path
import numpy as np
import onnx
import onnxruntime as ort
import torch
from torch.nn import functional as F
import run_controller_redesign_engineering as r

class CompletePolicy(torch.nn.Module):
    def __init__(self,model,arm):super().__init__();self.model=model;self.arm=arm
    def forward(self,states):
        if self.arm=='bdq64_uniform_adapted':
            # _BDQScale has an identity forward and a training-only backward.
            # Its identity forward is written directly for portable inference.
            h=self.model.trunk(states)
            a=self.model.advantage(h).reshape(-1,32,5)
            q=self.model.value(h)[:,:,None]+a-a.mean(2,keepdim=True)
            actions=q.argmax(2)
            return actions,q.gather(2,actions[:,:,None]).squeeze(2).mean(1)
        actions,q,_=r.greedy(self.model,states,self.arm)
        return actions,q

def options():
    x=ort.SessionOptions();x.intra_op_num_threads=1;x.inter_op_num_threads=1
    x.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    x.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return x

def session(path):
    return ort.InferenceSession(str(path),sess_options=options(),providers=['CPUExecutionProvider'])

def export(model,arm,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    original=r.old.digest(model.state_dict());wrapper=CompletePolicy(model.eval(),arm)
    # ONNX has Exp and Sub but legacy export lacks an aten::expm1 symbolic.
    # This is mathematically expm1(x), not bitwise identical for tiny FP32 x.
    # The amended protocol requires near-zero numerical and action checks.
    torch.onnx.register_custom_op_symbolic('aten::expm1',lambda graph,x:graph.op(
        'Sub',graph.op('Exp',x),graph.op('Constant',value_t=torch.tensor(1.,dtype=torch.float32))),17)
    start=time.perf_counter()
    with warnings.catch_warnings(record=True) as caught,torch.inference_mode():
        warnings.simplefilter('always')
        torch.onnx.export(wrapper,torch.zeros(1,533,dtype=torch.float32),str(path),
            input_names=['states'],output_names=['actions','chosen_q'],opset_version=17,
            dynamo=False,do_constant_folding=True,keep_initializers_as_inputs=False)
    export_seconds=time.perf_counter()-start
    onnx.checker.check_model(onnx.load(str(path)))
    start=time.perf_counter();compiled=session(path);startup=time.perf_counter()-start
    assert r.old.digest(model.state_dict())==original
    return compiled,dict(arm=arm,source_parameter_hash=original,onnx_sha256=r.old.sha(path),
        serialized_onnx_bytes=path.stat().st_size,export_seconds=export_seconds,session_startup_seconds=startup,
        export_warnings=sorted({str(w.message) for w in caught}),runtime=ort.__version__,onnx=onnx.__version__,
        providers=compiled.get_providers(),parameters=sum(p.numel() for p in model.parameters()),parameter_bytes=r.parameter_bytes(model))

def compiled_values(compiled,state):
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,):raise ValueError('fixed batch-one controller requires one 533-element observation')
    return compiled.run(['actions','chosen_q'],{'states':x[None]})

def compiled_action(compiled,state):return compiled_values(compiled,state)[0][0].copy()

def eager_action(model,arm,state):
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,):raise ValueError('fixed batch-one controller requires one 533-element observation')
    return r.select_complete(model,arm,x)
