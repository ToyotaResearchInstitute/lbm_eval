from copy import deepcopy


def is_directives_field(candidate_node):
    """Duck type the given node to test if it is the `directives` field of a
    ModelDirectives node.
    """
    # We can't just test for this parsing as a ModelDirectives because any use
    # of preparse distributions like UniformDiscreteString will cause the
    # parse to fail. In addition we are explicitly augmenting ModelDirectives
    # with the ability to specify multiple copies of a directive. Thus we duck
    # the ModelDirectives type by its members and count on later parses to
    # catch misstructured items.
    if not isinstance(candidate_node, list):
        return False
    for candidate_item in candidate_node:
        if not isinstance(candidate_item, dict):
            return False
        keys = set(candidate_item.keys())
        # There is no good set of minimal keys; instead there must be exactly
        # one key from the valid choices, as well as optionally num_copies.
        if "num_copies" in keys:
            if len(keys) != 2:
                return False
        else:
            if len(keys) != 1:
                return False
        # TODO(ggould) populate these from ModelDirective.__fields__ or
        # similar.
        maximal_keys = {
            "add_model",
            "add_model_instances",
            "add_frame",
            "add_weld",
            "add_collision_filter_group",
            "add_directives",
            "num_copies",
        }
        if not keys <= maximal_keys:
            return False
    return True


def resolve_directives_num_copies(directives_node, seed, resolver):
    """Resolves directive_node.num_copies for each member of directives_node,
    then taking each resolved value as an integer makes that many copies of
    that directive.  The num_copies element is removed.

    Only supports num_copies for the add_model field.
    """
    # TODO(dale.mcconachie) Relax the add_model restriction to all directives
    # for which num_copies would be semantically meaningful.
    result = []
    for directive_node in directives_node:
        num_copies = 1
        if "num_copies" in directive_node:
            if "add_model" not in directive_node:
                raise ValueError(
                    "num_copies is only supported for `add_model` directives"
                )
            num_copies = directive_node["num_copies"]
            if isinstance(num_copies, dict):
                num_copies = resolver(num_copies, seed)
            del directive_node["num_copies"]
        # int() cast is required here because python might have given us the
        # field as a float or str depending on hairy details of yaml parsing.
        for i in range(int(num_copies)):
            result.append(deepcopy(directive_node))
            if num_copies > 1:
                name = result[-1]["add_model"]["name"]
                result[-1]["add_model"]["name"] = f"{name}_{i}"
    return result
