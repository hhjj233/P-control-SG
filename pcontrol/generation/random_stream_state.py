"""Serializable RNG/cycle snapshots for Torch1.12 continuation checkpoints."""
import copy


def paired_stream_state(streams):
    return dict(order=copy.deepcopy(streams.order_rng.bit_generator.state),
                diffusion=streams.diffusion_rng.get_state().clone(),
                drop=copy.deepcopy(streams.drop_rng.bit_generator.state))


def restore_paired_streams(streams,state):
    streams.order_rng.bit_generator.state=copy.deepcopy(state['order'])
    streams.diffusion_rng.set_state(state['diffusion'].cpu())
    streams.drop_rng.bit_generator.state=copy.deepcopy(state['drop'])


def slow_cycle_state(cycle):
    state={k:copy.deepcopy(getattr(cycle,k)) for k in ('pools','queues','record_orders','positions')}
    state['rng']=copy.deepcopy(cycle.rng.bit_generator.state)
    return state


def restore_slow_cycle(cycle,state):
    for k in ('pools','queues','record_orders','positions'):setattr(cycle,k,copy.deepcopy(state[k]))
    cycle.rng.bit_generator.state=copy.deepcopy(state['rng'])
