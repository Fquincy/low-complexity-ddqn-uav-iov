"""Final deployment/storage stage from frozen checkpoints; no fitting or timing."""
import io,json
from pathlib import Path
import torch
import run_horizon_loss_iov_fixture as r

def tensor_bytes(value):
    if isinstance(value,torch.Tensor):return value.numel()*value.element_size()
    if isinstance(value,dict):return sum(tensor_bytes(x) for x in value.values())
    if isinstance(value,(list,tuple)):return sum(map(tensor_bytes,value))
    return 0

def log_totals(path):
    elapsed=samples=0
    with path.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);elapsed+=row['optimizer_ns'];samples+=len(row['anchor_indices'])
    return elapsed,samples

def export(saved_run,output):
    saved_run=Path(saved_run);output=Path(output);output.mkdir(parents=True,exist_ok=False)
    p=r.protocol();pl=r.read(saved_run/'runtime_plan.json');frozen=r.read(saved_run/'ALL_FINAL_CHECKPOINTS_FREEZE.json');rows=[]
    for definition in p['learned_arms']:
        arm=definition['id']
        for fit,seed in enumerate(pl['model_seeds']):
            path=saved_run/arm/f'fit_{fit:02d}'/'final.pt';key=f'{arm}/fit_{fit:02d}/final.pt'
            r.require(r.sha(path)==frozen[key],'checkpoint mutated before export')
            ck=torch.load(path,weights_only=False,map_location='cpu')
            bundle=dict(kind='DEPLOYMENT_EXPORT_NOT_TRAINING_CHECKPOINT',protocol_sha256=r.sha(r.SPEC),definition=definition,
                online=ck['online'],source_checkpoint_sha256=frozen[key],runtime_sources={n:r.sha(r.ROOT/n) for n in ('deadline_screen_research_models.py','deadline_screen_fast_runtime.py','deadline_screen_prototype.py','controller_v3_fused_runtime.py')})
            name=f'{arm}_fit_{fit:02d}.deployment.bin';torch.save(bundle,output/name)
            elapsed,samples=log_totals(path.parent/'optimizer.jsonl')
            rows.append(dict(arm=arm,fit=fit,source_checkpoint=key,source_checkpoint_sha256=frozen[key],
                deployment_file=name,deployment_file_bytes=(output/name).stat().st_size,deployment_sha256=r.sha(output/name),
                online_parameter_bytes=definition['expected_fp32_parameter_bytes'],online_buffer_bytes=definition['expected_buffer_bytes'],
                online_state_tensor_bytes=tensor_bytes(ck['online']),target_state_tensor_bytes=tensor_bytes(ck['target']),
                optimizer_state_tensor_bytes=tensor_bytes(ck['opt']),replay_preallocated_bytes=ck['replay_storage_bytes'],
                checkpoint_file_bytes=path.stat().st_size,training_wall_ns=ck['training_ns'],optimizer_wall_ns=elapsed,
                anchor_samples=samples))
    r.write(output/'storage_and_training_cost.json',dict(kind=pl['kind'],models=rows,export_only=True,
        extra_optimizer_steps=0,extra_simulator_steps=0,timing_repeated=False,
        scope='Deployment bundle contains online state and reconstruction metadata; Python/framework/runtime dependencies remain external. Not total device/process memory or standalone executable.'))
    r.write(output/'manifest.json',dict(files={x.name:r.sha(x) for x in output.iterdir() if x.is_file()}))
    return rows

def audit(saved_run,output):
    saved_run=Path(saved_run);output=Path(output);p=r.protocol();pl=r.read(saved_run/'runtime_plan.json');rows=r.read(output/'storage_and_training_cost.json')['models']
    manifest=r.read(output/'manifest.json')['files']
    r.require({x.name for x in output.iterdir() if x.is_file()}==set(manifest)|{'manifest.json'},'storage manifest files')
    for n,h in manifest.items():r.require(r.sha(output/n)==h,'storage artifact hash')
    expected={(d['id'],i) for d in p['learned_arms'] for i in range(len(pl['model_seeds']))}
    r.require(len(rows)==len(expected) and {(x['arm'],x['fit']) for x in rows}==expected,'storage model coverage')
    for x in rows:
        ck=torch.load(saved_run/x['source_checkpoint'],weights_only=False,map_location='cpu')
        bundle=torch.load(output/x['deployment_file'],weights_only=False,map_location='cpu')
        r.require(bundle['kind']=='DEPLOYMENT_EXPORT_NOT_TRAINING_CHECKPOINT','deployment kind')
        r.require(bundle['source_checkpoint_sha256']==x['source_checkpoint_sha256']==r.sha(saved_run/x['source_checkpoint']),'deployment source')
        r.require(bundle['definition']==ck['definition'] and bundle['protocol_sha256']==r.sha(r.SPEC),'deployment definition')
        r.require(set(bundle['online'])==set(ck['online']),'deployment state keys')
        for key,value in ck['online'].items():r.require(torch.equal(value,bundle['online'][key]),'deployment changed weight')
        # Independent sums of serialized state tensors, not file-size estimates.
        online=sum(t.numel()*t.element_size() for t in ck['online'].values())
        target=sum(t.numel()*t.element_size() for t in ck['target'].values())
        opt=sum(t.numel()*t.element_size() for state in ck['opt']['state'].values() for t in state.values() if isinstance(t,torch.Tensor))
        r.require((online,target,opt)==(x['online_state_tensor_bytes'],x['target_state_tensor_bytes'],x['optimizer_state_tensor_bytes']),'storage tensor counts')
        d=ck['definition'];r.require(online==d['expected_fp32_parameter_bytes']+d['expected_buffer_bytes'],'online bytes')
        r.require(x['online_parameter_bytes']==d['expected_fp32_parameter_bytes'] and x['online_buffer_bytes']==d['expected_buffer_bytes'],'parameter/buffer split')
        capacity=pl['cfg']['replay_capacity'];r.require(x['replay_preallocated_bytes']==capacity*(2*533*4+32+2*4),'replay capacity bytes')
        r.require(x['deployment_file_bytes']==(output/x['deployment_file']).stat().st_size and x['deployment_sha256']==r.sha(output/x['deployment_file']),'deployment file')
        r.require(x['checkpoint_file_bytes']==(saved_run/x['source_checkpoint']).stat().st_size and x['training_wall_ns']==ck['training_ns']>0,'training wall/checkpoint size')
        elapsed,samples=log_totals((saved_run/x['source_checkpoint']).parent/'optimizer.jsonl')
        r.require(x['optimizer_wall_ns']==elapsed and x['anchor_samples']==samples,'training cost sums')
    return dict(passed=True,models=len(rows),online_weights_exact=True,storage_and_training_cost_recomputed=True,extra_optimizer_steps=0,extra_simulator_steps=0,timing_repeated=False)

def complete_workflow(saved_run,storage_output):
    """Finish the same pipeline after r.run; independent auditor remains a separate stage."""
    export(saved_run,storage_output)
    result=audit(saved_run,storage_output)
    return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--saved-run',type=Path,required=True);parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    print(json.dumps(complete_workflow(args.saved_run,args.output)))
