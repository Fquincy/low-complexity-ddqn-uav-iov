"""Frozen observed-state heuristic. No env object, private task data or fitting."""
import numpy as np
from controller_v3_fused_runtime import native
from assignment_price_fast_runtime import validate_mask
import run_horizon_loss_research as r

def constants():
    c=r.Chapter3Config()
    return np.array([c.area_m,c.task_size_bytes[1]*8,c.vehicle_cpu_hz,*c.uav_cpu_hz,*c.rsu_cpu_hz,
        c.g2a_bandwidth_hz,c.vehicle_tx_power_w,10**((c.noise_dbm-30)/10),c.carrier_hz,c.uav_altitude_m,
        c.los_a,c.los_b,c.los_excess_loss_db,c.nlos_excess_loss_db,c.rsu_link_bps,
        c.delay_reward_weight,c.deadline_miss_penalty,c.uav_energy_reward_weight,c.effective_switched_capacitance,c.uav_energy_normalizer_j],np.float64)

@native
def prepare(x,c):
    tasks=x[:240].astype(np.float64).reshape(30,8);uavs=x[526:532].astype(np.float64).reshape(2,3)
    bits=tasks[:,5]*c[1];cycles=tasks[:,6]*500e6;deadline=tasks[:,7]*.300
    local=np.expm1(x[240:270].astype(np.float64))*.300
    remote=np.zeros(5,np.float64);remote[1:]=np.expm1(x[270:274].astype(np.float64))*.300
    backlog=np.expm1(x[274:514].astype(np.float64).reshape(30,8)[:,:4])*c[1]
    gain=np.empty((30,2),np.float64)
    for i in range(30):
        for j in range(2):
            delta=(tasks[i,:2]-uavs[j,:2])*c[0]
            distance=np.sqrt(np.sum(delta*delta)+c[11]**2)
            elevation=np.degrees(np.arcsin(c[11]/distance))
            los=1/(1+c[12]*np.exp(-c[13]*(elevation-c[12])))
            loss=20*np.log10(4*np.pi*c[10]/299792458.)+20*np.log10(distance)+los*c[14]+(1-los)*c[15]
            gain[i,j]=10**(-loss/10)
    return bits,cycles,deadline,local,remote,backlog,gain,uavs[:,2].copy()

@native
def rates(a,gain,c):
    n=len(a);out=np.full((n,4),c[16],np.float64);sums=np.zeros(2,np.float64)
    for i in range(n):
        if a[i]==1 or a[i]==2:sums[a[i]-1]+=gain[i,a[i]-1]
    for i in range(n):
        for j in range(2):
            interference=c[8]*max(0.,sums[j]-(gain[i,j] if a[i]==j+1 else 0.))
            out[i,j]=max(1.,c[7]*np.log2(1+c[8]*gain[i,j]/(c[9]+interference)))
    return out

@native
def objective(a,bits,cycles,deadline,local,remote,backlog,gain,order,c):
    rate=rates(a,gain,c);prefix=np.zeros(5,np.float64);total=0.;misses=0;energy=0.
    for index in range(len(a)):
        i=order[index];d=a[i]
        if d==0:finish=local[i]+cycles[i]/c[2]
        else:
            prefix[d]+=cycles[i]/c[2+d]
            wait=0.
            for k in range(4):wait+=backlog[i,k]/rate[i,k]
            finish=wait+bits[i]/rate[i,d-1]+remote[d]+prefix[d]
            if d<=2:energy+=c[20]*cycles[i]*c[2+d]**2
        total+=finish/.300;misses+=int(finish>deadline[i]+1e-10)
    return c[17]*total/len(a)+c[18]*misses/len(a)+c[19]*energy/c[21]

@native
def decode(bits,cycles,deadline,local,remote,backlog,gain,battery,mask,c):
    n=len(bits);order=np.argsort(-cycles,kind='mergesort');a=np.zeros(n,np.int64);checks=0
    for j in range(2):
        if battery[j]<=0:mask[:,j+1]=False
    for stage in range(2):
        for index in range(n):
            i=order[index];old=a[i];best=old;value=np.inf
            if stage==1:value=objective(a,bits,cycles,deadline,local,remote,backlog,gain,order,c)
            for d in range(5):
                if mask[i,d]:
                    a[i]=d;candidate=objective(a,bits,cycles,deadline,local,remote,backlog,gain,order,c);checks+=1
                    if candidate<value or (candidate==value and (stage==0 or best!=old) and d<best):best=d;value=candidate
            a[i]=best
    return a,checks

class Runtime:
    def __init__(self):self.c=constants()
    def values(self,state):
        x=np.asarray(state)
        if x.dtype!=np.float32 or x.shape!=(533,):raise ValueError('one FP32 state required')
        mask,h=validate_mask(x);parts=prepare(x,self.c)
        if not all(np.isfinite(v).all() for v in parts):raise ValueError('nonfinite decoded feature')
        if any(np.any(v<0) for v in parts[:6]):raise ValueError('negative load')
        a,count=decode(*parts,mask.copy(),self.c)
        return np.r_[a,np.zeros(2,np.int64)][None],np.array([count],np.int64)
