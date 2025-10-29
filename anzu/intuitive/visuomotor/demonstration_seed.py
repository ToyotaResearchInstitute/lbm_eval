def get_demonstration_seed(demonstration_index, use_eval_seed):
    # Legacy.
    demonstration_seed = demonstration_index + 100
    if use_eval_seed:
        demonstration_seed += 100000000
    return demonstration_seed
