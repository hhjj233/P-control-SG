"""Same dynamic-risk forward function, with explicit full-generator tuning.

Only generator parameters are exposed. History-conditioned CDFs/physical
pieces remain external frozen inputs. The learned null is loaded, not reset;
unconditional outputs need not remain equal to the old generator after tuning.
"""
from .dynamic_risk_direct_p import DynamicRiskPercentileDenoiser

ADAPTER_PREFIXES=('ruler_encoder.','target_projection.','dynamic_blocks.')


class TrainableRiskGenerator(DynamicRiskPercentileDenoiser):
    VERSION='natural_full_generator_risk_finetuning_v1'

    def train_adapter_only(self):
        raise ValueError('this version requires explicit full-generator parameter groups')

    def finetuning_groups(self,core_lr,adapter_lr):
        if not 0<core_lr<=adapter_lr:raise ValueError('positive core LR no larger than adapter LR required')
        self.requires_grad_(True)
        core=[];adapters=[]
        for name,p in self.named_parameters():
            (adapters if name.startswith(ADAPTER_PREFIXES) else core).append(p)
        return [dict(params=core,lr=core_lr,name='generator_core'),dict(params=adapters,lr=adapter_lr,name='risk_adapters')]

    def architecture_config(self):
        config=super().architecture_config()
        config.update(version=self.VERSION,generator_core_frozen=False,estimator_remains_frozen=True,
            learned_null_loaded_without_reset=True,learned_null_frozen=False,
            unconditional_parent_function_preserved_after_training=False,
            EMA_update_scope_required='all_generator_parameters; buffers_copied',
            model_architecture_operators_unchanged=True)
        return config
