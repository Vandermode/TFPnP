from .network import ResNetActor_ADMM, ResNetActor_HQS, ResNetActor_PG, ResNetActor_APG, ResNetActor_IADMM, ResNetActor_RED, ResNetActor_AMP


_solver_map = {
    'admm': ResNetActor_ADMM,
    'hqs': ResNetActor_HQS,
    'pg': ResNetActor_PG,
    'apg': ResNetActor_APG,
    'redadmm': ResNetActor_RED,
    'amp': ResNetActor_AMP,
    'iadmm': ResNetActor_IADMM
}


def create_policy_network(opt, num_aux_inputs, action_range=None):
    if opt.solver in _solver_map:
        Policy = _solver_map[opt.solver]
        actor = Policy(num_aux_inputs, opt.action_pack, action_range)
    else:
        raise NotImplementedError
    return actor
