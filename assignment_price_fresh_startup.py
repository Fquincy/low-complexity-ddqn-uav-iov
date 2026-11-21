"""Fresh-process deployment startup helper; loads only frozen checkpoint/graph."""
import time
PROCESS_MODULE_START=time.perf_counter_ns()
import argparse
from pathlib import Path
import numpy as np
import torch
import run_assignment_price_research_bound as w

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True);parser.add_argument('--arm',required=True);parser.add_argument('--fit',type=int,required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=Path(args.run);w.r.configure();pl=w.r.read(out/'runtime_plan.json');start=time.perf_counter_ns()
    a,ck=w.load_agent(out,args.arm,args.fit,pl);loaded=time.perf_counter_ns()
    runtime=w.from_graph(a,out/args.arm/f'fit_{args.fit:02d}'/'final.onnx',args.arm);session=time.perf_counter_ns()
    state=np.load(out/'startup_state.npy');actual,q=runtime.values(state);finished=time.perf_counter_ns()
    with torch.inference_mode():actions,values=a['online'].greedy(torch.from_numpy(state)[None])
    np.testing.assert_array_equal(actions.numpy(),actual);np.testing.assert_allclose(values.numpy(),q,rtol=1e-5,atol=1e-6)
    w.r.write(args.output,dict(process_id=__import__('os').getpid(),model_load_ns=loaded-start,session_load_ns=session-loaded,
        first_action_ns=finished-session,module_start_to_first_action_ns=finished-PROCESS_MODULE_START,
        checkpoint_sha256=w.r.sha(out/args.arm/f'fit_{args.fit:02d}'/'final.pt'),graph_sha256=w.r.sha(out/args.arm/f'fit_{args.fit:02d}'/'final.onnx'),
        equivalent=True,includes_native_first_compilation=True,scope='Fresh Python process; shared OS file caches not flushed; parent wall includes interpreter startup/exit and verifier'))

if __name__=='__main__':main()
