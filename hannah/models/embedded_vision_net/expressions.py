def expr_product(expressions: list):
    res = None
    for expr in expressions:
        if res:
            res = res * expr
        else:
            res = expr
    return res