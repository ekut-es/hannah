

def get_id_and_update_counters(current_scope, counters):
    if len(current_scope) > 1:
        scope = '.'.join([current_scope[-2].id, current_scope[-1].name])
    else:
        scope = current_scope[-1].name
    if scope not in counters:
        counters[scope] = 0
    else:
        counters[scope] += 1

    return '{}.{}'.format(scope, counters[scope])


def update_scope(node, current_scope):
    return current_scope + [node]
