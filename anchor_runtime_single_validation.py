"""Equivalent corrected-controller route with a single complete input check."""
import numpy as np
import controller_v3_fused_runtime as native
from terminal_anchor_prototype import ARMS

def values(runtime,arm,state):
    """Validate once, execute unchanged core/decoder, scale the FP32 value."""
    if arm not in ARMS:raise ValueError('corrected arm required')
    x=np.asarray(state,dtype=np.float32)
    if x.shape!=(533,) or not np.isfinite(x).all():raise ValueError('finite533 observation required')
    t=float(x[-1])*40;slot=round(t)
    if abs(t-slot)>1e-4 or not 0<=slot<40:raise ValueError('nonterminal grid time required')
    session,extra=runtime;raw=session.run(None,{'states':x[None]})[0][0]
    base_arm,scaled,_=ARMS[arm]
    if base_arm=='plain24_nonpositive':action,q=native.flat_decode(raw,2)
    elif base_arm=='additive64_nonpositive':action,q=native.flat_decode(raw.reshape(-1),1)
    else:action,q=native.v3_decode(raw,x,base_arm=='amortized_workload_price24')
    # Preserve the original two FP32 operations/rounding, not a fused float64 product.
    value=np.float32(q)*np.float32(40-slot if scaled else 1.)
    return action[None],np.array([value],np.float32)
