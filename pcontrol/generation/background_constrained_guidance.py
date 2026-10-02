"""Late-DDIM background separation interleaved with road and risk guidance."""
from .road_constrained_guidance import RoadConstrainedGuidance
from .background_envelope import BackgroundEnvelope


class _SceneEnvelope:
    def __init__(self,road,background):
        self.road=road;self.background=background;self.background_enabled=False

    def project(self,coefficients):
        result,road=self.road.project(coefficients)
        if self.background_enabled:
            result,bg=self.background.project(result)
            unexpected=[p for p in bg['remaining_background_pairs'] if p not in bg['initial_background_pairs']]
            bad=sorted({i for pair in unexpected for i in pair})
            road=dict(road,background_projection=bg,failed_actors=sorted(set(road['failed_actors']+bad)))
        return result,road

    def report(self):return self.road.report()


class BackgroundConstrainedGuidance(RoadConstrainedGuidance):
    def __init__(self,*args,background_last_steps=5,background_clearance_m=.1,background_max_passes=4,**kwargs):
        super().__init__(*args,**kwargs)
        if type(background_last_steps) is not int or not 1<=background_last_steps<=self.last_steps:raise ValueError('background window must be inside risk window')
        self.background_last_steps=background_last_steps
        self.background=BackgroundEnvelope(self.decoder,self.dimensions,self.inverse_metric[0],self.ego_index,
            clearance_m=background_clearance_m,max_passes=background_max_passes)
        self.envelope=_SceneEnvelope(self.envelope,self.background);self.background_trace=[]

    def __call__(self,x0,timesteps,x_t):
        step=self.calls;active=self.strength>0 and step>=self.total_steps-self.background_last_steps
        self.envelope.background_enabled=active
        result=super().__call__(x0,timesteps,x_t)
        if active:
            # Convex blending preserves road bounds but not every disjunctive
            # collision region if lateral windows change. Recheck before the
            # DDIM transition, NOT after the sampler has returned a trajectory.
            corrected,info=self.background.project(result[0].detach().double())
            self.background_trace.append(dict(step=step,phase='before_DDIM_transition',**info))
            result=corrected[None].to(result.dtype)
        return result

    def report(self):
        result=super().report()
        result.update(background_envelope=self.background.report(),background_trace=self.background_trace,
            background_last_steps=self.background_last_steps,background_ego_coefficients_fixed=True,
            background_projection_inside_sampler=True,background_projection_not_subject_to_risk_step_cap=True)
        return result
