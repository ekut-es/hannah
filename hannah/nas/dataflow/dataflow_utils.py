
def register_scope(scope, scope_counters):
    if scope not in scope_counters:
        scope_counters[scope] = 0
    else:
        scope_counters[scope] += 1
    return scope + ".{{{}}}".format(scope_counters[scope])


def scope_nester(scope, current_scope, nesting):
    nesting[".".join(current_scope)] = scope
    return nesting


def reset_nested_counters(ended_scope, nesting, scope_counter):
    if ended_scope in nesting:
        base_scope_str = nesting[ended_scope].split(".")[0]
        scope_counter.pop(base_scope_str)
