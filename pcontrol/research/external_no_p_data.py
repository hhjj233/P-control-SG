"""Natural observations only: no risk/percentile/CDF fields in external packs."""
import hashlib, json, sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
OUT=ROOT/'outputs/natural_percentile/external_no_p_v1_20260922'
SOURCE=ROOT/'outputs/natural_percentile/complete_scene_expanded_v1_20260910/manifest.json'
SOURCE_SHA='e05bd02e52a81ece015d86be8e106d7c353db42cc29377df364d9306d88c9012'

def bind(path):return dict(path=str(path),sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest())
def read(binding):
    assert bind(binding['path'])['sha256']==binding['sha256']
    return json.loads(Path(binding['path']).read_text())
def write(path,obj):
    with Path(path).open('x') as f:json.dump(obj,f,indent=2)
    return bind(path)
def load(role):
    assert role in ('FIT','STOP');m=json.loads((OUT/'data/manifest.json').read_text());b=m['packs'][role]
    assert bind(b['path'])['sha256']==b['sha256']
    with np.load(b['path'],allow_pickle=False) as f:a={k:f[k] for k in f.files}
    assert set(a['role'])=={role};return a

def prepare():
    from pcontrol.generation.data import NaturalTrajectorySource
    dest=OUT/'data';dest.mkdir(parents=True,exist_ok=True);bindings={}
    assert not (dest/'manifest.json').exists(), 'Do not overwrite a completed data preparation'
    for role in ('FIT','STOP'):
        s=NaturalTrajectorySource(dict(path=str(SOURCE),sha256=SOURCE_SHA),role=role)
        n=len(s)
        shapes=[]
        for i in range(n):
            f=s.view[i]['features'];shapes.append((len(f['ego_mask']),len(f['road_boundaries'])))
        maximum=max(v[0] for v in shapes);roads=max(v[1] for v in shapes)
        path=dest/(role+'.npz')
        if path.exists():
            with np.load(path,allow_pickle=False) as saved:
                assert set(saved['role'])=={role} and len(saved['role'])==n
                assert saved['history'].shape==(n,13,maximum,4)
                assert saved['future'].shape==(n,175,maximum,4)
                assert saved['scene_id'].tolist()==[row[2] for row in s.rows]
            bindings[role]=bind(path);continue
        a=dict(history=np.zeros((n,13,maximum,4),np.float32),future=np.zeros((n,175,maximum,4),np.float32),
            dimensions=np.zeros((n,maximum,2),np.float32),road_boundaries=np.zeros((n,roads),np.float32),
            road_count=np.zeros(n,np.int32),num_agents=np.zeros(n,np.int32),ego_index=np.zeros(n,np.int32),
            anchors=np.zeros((n,maximum,4),np.float64))
        sid=[];recs=[]
        for i in range(n):
            e=s[i];f=e['features'];count=len(f['ego_mask']);nr=len(f['road_boundaries'])
            assert count<=maximum and nr<=roads
            a['history'][i,:,:count]=f['history'];a['future'][i,:,:count]=e['future_observed']
            a['dimensions'][i,:count]=f['dimensions'];a['anchors'][i,:count]=e['anchors']
            a['num_agents'][i]=count;a['ego_index'][i]=int(np.flatnonzero(f['ego_mask'])[0]);a['road_count'][i]=nr
            a['road_boundaries'][i,:nr]=f['road_boundaries'];sid.append(e['metadata']['scene_id']);recs.append(e['metadata']['recording_id'])
            if (i+1)%2000==0:print(json.dumps(dict(stage='prepare',role=role,scenes=i+1)),flush=True)
        a.update(scene_id=np.array(sid),recording_id=np.array(recs),role=np.full(n,role))
        np.savez_compressed(path,**a);bindings[role]=bind(path)
        del a
    write(dest/'manifest.json',dict(status='complete',source=dict(path=str(SOURCE),sha256=SOURCE_SHA),packs=bindings,
        actual_observed_futures=True,coefficient_reconstruction=False,P_or_PET_or_CDF_fields=False,
        code=bind(__file__),protocol=bind(ROOT/'docs/EXTERNAL_NO_P_BASELINES_PROTOCOL_20260922.md')))

def groups(a,rng=None,batch=16):
    result=[]
    for n in sorted(set(a['num_agents'])):
        rows=np.flatnonzero(a['num_agents']==n)
        if rng is not None:rows=rng.permutation(rows)
        result.extend(rows[i:i+batch] for i in range(0,len(rows),batch))
    if rng is not None:rng.shuffle(result)
    return result

if __name__=='__main__':prepare()
