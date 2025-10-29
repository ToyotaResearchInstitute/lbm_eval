from copy import deepcopy


def is_list_node(candidate_node):
    """Test if the given node is a list. Used to filter down to only those
    nodes that `resolve_list_num_copies` operates on.
    """
    return isinstance(candidate_node, list)


def resolve_list_num_copies(list_node, seed, resolver):
    """Resolves any !SampleNumCopiesAndFlatten entries in the list, creating
    the desired number of copies and placing them in the list in place of the
    original !SampleNumCopiesAndFlatten node. Does not perform any
    uniquification of the copies that are generated.
    """
    result = []
    for item in list_node:
        num_copies = 1
        elements = [item]
        if isinstance(item, dict) and "_tag" in item:
            if item["_tag"] == "!SampleNumCopiesAndFlatten":
                num_copies = item["num_copies"]
                if isinstance(num_copies, dict):
                    num_copies = resolver(num_copies, seed)
                elements = item["elements"]
        # int() cast is required here because python might have given us the
        # field as a float or str depending on hairy details of yaml parsing.
        for _ in range(int(num_copies)):
            result.extend(deepcopy(elements))
    return result
