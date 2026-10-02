#!/usr/bin/env python3
"""Exact empirical-CDF baselines, selected on STOP, evaluated on full TEST."""
import json, pickle, sys, time
from pathlib import Path
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import NearestNeighbors
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
from pcontrol.time_attention_pipeline import common as c
from pcontrol.publication_pipeline.final_execution_data import CompleteFinalDataset
from pcontrol.research import train_natural_scene_reference as io

OUT=ROOT/'outputs/natural_percentile/baselines_v1/reference'
FINAL=ROOT/'outputs/natural_percentile/transformer_publication_v1_20260916/final_validation_v1/TEST'

def vector(f):
    """All actors enter permutation-invariant pooled history statistics."""
    mask=f['agent_mask'];h=f['history'][:,mask];d=f['dimensions'][mask]
    ego=int(np.flatnonzero(f['ego_mask'][mask])[0]);env=np.arange(len(d))!=ego
    rel=h[:,env]-h[:,ego,None];road=f['road_boundaries']
    if 'road_boundary_mask' in f:road=road[f['road_boundary_mask']]
    return np.concatenate((h[:,ego].ravel(),rel.mean(1).ravel(),rel.std(1).ravel(),
        rel.min(1).ravel(),rel.max(1).ravel(),d[ego],d.mean(0),d.max(0),
        [len(d),len(road),road[0],road[-1]]))

def physical(pack):
    n=len(pack['target']);keys=('history','dimensions','ego_mask','agent_mask','road_boundaries','road_boundary_mask')
    return np.array([vector({k:pack[k][i] for k in keys}) for i in range(n)])

class EmpiricalForest:
    """Meinshausen leaf weights over *all FIT observations* in each leaf."""
    def __init__(self,leaf=10,features='sqrt'):
        self.forest=RandomForestRegressor(n_estimators=100,min_samples_leaf=leaf,max_features=features,
                                          random_state=20260922,n_jobs=2)
    def fit(self,x,y):
        self.order=np.argsort(y,kind='stable');self.y=y[self.order];xs=x[self.order]
        self.forest.fit(xs,self.y);leaves=self.forest.apply(xs)
        sizes=np.array([t.tree_.node_count for t in self.forest.estimators_]);self.offsets=np.r_[0,np.cumsum(sizes)[:-1]]
        rows=(leaves+self.offsets).T.ravel();cols=np.tile(np.arange(len(y)),100)
        counts=np.bincount(rows,minlength=int(sizes.sum()))
        self.leaf_weights=csr_matrix((1/(100*counts[rows]),(rows,cols)),shape=(sizes.sum(),len(y)))
        return self
    def weights(self,x):
        cols=(self.forest.apply(x)+self.offsets).ravel();rows=np.repeat(np.arange(len(x)),100)
        query=csr_matrix((np.ones(len(cols)),(rows,cols)),shape=(len(x),self.leaf_weights.shape[0]))
        result=(query@self.leaf_weights).toarray()
        if not np.allclose(result.sum(1),1.,atol=1e-12):raise ValueError('QRF weight mass')
        return result

def scores(w,support,y):
    """Exact discrete CRPS; ties have zero integration width, not jitter."""
    if not np.all(np.diff(support)>=0) or np.any(w<0):raise ValueError('sorted nonnegative measure')
    cumulative=np.cumsum(w,axis=1)
    first=(w*abs(support[None]-y[:,None])).sum(1)
    half_pair=(w*support[None]*(2*cumulative-w-1)).sum(1)
    crps=first-half_pair
    left=(w*(support[None]<y[:,None])).sum(1);right=(w*(support[None]<=y[:,None])).sum(1)
    grid=np.arange(1,20)/20
    width=right-left
    pit=np.where(width[:,None]>1e-14,
        np.clip((grid[None]-left[:,None])/np.maximum(width[:,None],1e-14),0,1),
        (right[:,None]<=grid[None]).astype(float))
    return dict(CRPS_seconds=crps,PIT=pit,cdf_left=left,cdf_right=right,
                cap_probability=w[:,support==4.].sum(1))

def metrics(a):
    err=a['CRPS_seconds'];n=a['num_agents']
    return dict(scenes=len(err),CRPS_seconds=float(err.mean()),
        PIT_grid_KS=float(abs(a['PIT'].mean(0)-np.arange(1,20)/20).max()),
        cap_Brier=float(np.mean((a['cap_probability']-(a['target']==4.))**2)),
        by_N={name:dict(scenes=int(m.sum()),CRPS_seconds=float(err[m].mean())) for name,m in
              [('N3_5',n<=5),('N6_8',(n>=6)&(n<=8)),('N9plus',n>=9)]})

def main():
    OUT.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    policy=c.json_file(c.bind(ROOT/'configs/natural_percentile/guarded_generator_adaptation_v1.json'))
    lm=c.json_file(policy['labels_manifest'])
    fit=c.arrays(lm['physical_packs']['FIT']);stop=c.arrays(lm['physical_packs']['STOP'])
    assert set(fit['role'])=={'FIT'} and set(stop['role'])=={'STOP'}
    x=physical(fit);xs=physical(stop);y=fit['target'];ys=stop['target']
    scale=np.maximum(x.std(0),1e-6);x=x/scale;xs=xs/scale
    candidates=[];chosen={};frozen=[]
    for leaf in (10,40):
        for features in ('sqrt',.5):
            model=EmpiricalForest(leaf,features).fit(x,y)
            val=np.concatenate([scores(model.weights(xs[s:s+64]),model.y,ys[s:s+64])['CRPS_seconds'] for s in range(0,len(xs),64)]).mean()
            row=dict(family='QRF',leaf=leaf,max_features=features,STOP_CRPS=float(val));candidates.append(row)
            if 'QRF' not in chosen or val<chosen['QRF'][0]:chosen['QRF']=(val,model,row)
            print(json.dumps(row),flush=True)
    neighbors=NearestNeighbors(n_neighbors=128,n_jobs=2).fit(x);idx=neighbors.kneighbors(xs,return_distance=False)
    for k in (32,128):
        values=np.sort(y[idx[:,:k]],axis=1)
        val=np.mean([scores(np.full((1,k),1/k),v,np.array([t]))['CRPS_seconds'][0] for v,t in zip(values,ys)])
        row=dict(family='kNN',k=k,STOP_CRPS=float(val));candidates.append(row)
        if 'kNN' not in chosen or val<chosen['kNN'][0]:chosen['kNN']=(val,k,row)
    with (OUT/'models.pkl').open('xb') as f:pickle.dump(dict(QRF=chosen['QRF'][1],neighbors=neighbors,
        k=chosen['kNN'][1],target=y,scale=scale),f)
    freeze=io.write_json(OUT/'selection.json',dict(status='selected_on_STOP_before_new_TEST_predictions',
        protocol=c.bind(ROOT/'docs/baseline_protocol.md'),
        FIT=lm['physical_packs']['FIT'],STOP=lm['physical_packs']['STOP'],candidates=candidates,
        selected={k:v[2] for k,v in chosen.items()},models=c.bind(OUT/'models.pkl'),feature_dim=x.shape[1],
        code=c.source_bindings(('pcontrol/research/classical_reference_baselines.py',)),CAL_used=False))
    db=c.bind(FINAL/'dataset.json');dm=c.json_file(db)
    dataset=CompleteFinalDataset(db,scope='TEST',allowed_recordings=dm['expected_recordings'])
    xx=[];targets=[];counts=[];sids=[];recs=[]
    for i in range(len(dataset)):
        e=dataset[i];xx.append(vector(e['features'])/scale);targets.append(e['target']);counts.append(len(e['agent_ids']))
        sids.append(e['metadata']['scene_id']);recs.append(e['metadata']['recording_id'])
    xx=np.array(xx);targets=np.array(targets)
    result={};model=chosen['QRF'][1];k=chosen['kNN'][1];support=np.sort(y)
    for family in ('ECDF','kNN','QRF'):
        acc={}
        for start in range(0,len(xx),64):
            z=xx[start:start+64];t=targets[start:start+64]
            if family=='QRF':item=scores(model.weights(z),model.y,t)
            elif family=='ECDF':item=scores(np.full((len(z),len(y)),1/len(y)),support,t)
            else:
                idx=neighbors.kneighbors(z,return_distance=False)[:,:k];w=np.zeros((len(z),len(y)))
                w[np.arange(len(z))[:,None],idx]=1/k;order=np.argsort(y,kind='stable')
                item=scores(w[:,order],y[order],t)
            for name,values in item.items():acc.setdefault(name,[]).append(values)
        a={name:np.concatenate(values) for name,values in acc.items()}
        a.update(target=targets,scene_id=np.array(sids),recording_id=np.array(recs),num_agents=np.array(counts))
        np.savez_compressed(OUT/(family+'_TEST.npz'),**a)
        result[family]=dict(metrics=metrics(a),predictions=c.bind(OUT/(family+'_TEST.npz')))
        print(json.dumps(dict(family=family,**result[family]['metrics'])),flush=True)
    io.write_json(OUT/'result.json',dict(status='complete',selection=freeze,dataset=db,
        baselines=result,TEST_selection=False,CAL_used=False,wall_seconds=time.monotonic()-started))

if __name__=='__main__':main()
