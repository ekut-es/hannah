def lazy(argument):
    if hasattr(argument, 'evaluate'):
        return argument.evaluate()
    else:
        return argument
