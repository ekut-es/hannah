from hannah.nas.dataflow.op_type import OpType
from hannah.nas.dataflow.tensor_expression import TensorExpression
from hannah.nas.dataflow.registry import _OPS, _SHAPE_FUNCS, _CONVERSIONS


MISSING = 'missing'


def add_op(op_class):
    operands = {}
    attributes_annotations = {}
    attributes_defaults = {}
    cls_annotations = op_class.__annotations__
    for name, annotation in cls_annotations.items():
        if annotation is TensorExpression:
            operands[name] = annotation
        else:
            attributes_annotations[name] = annotation
            default = getattr(op_class, name, MISSING)
            attributes_defaults[name] = default
        # if hasattr(op_class, name):
        #     delattr(op_class, name)
    op_class.operands = operands
    op_class.attributes_annotations = attributes_annotations
    op_class.attributes_defaults = attributes_defaults

    op_class.create_op = _create_op

    _OPS[op_class.__name__] = op_class

    return op_class


@classmethod
def _create_op(cls, *operands, **attributes):

    # just check for the same amount of operands, not names
    assert len(operands) == len(cls.operands), "{} expects (exactly) the following operands: {}".format(cls.__name__, cls.operands)
    for operand, operand_name in zip(operands, cls.operands):
        assert isinstance(operand, cls.operands[operand_name]), \
               "Wrong operand type: Expected {} and got {}".format(type(operand), cls.operands[operand_name])

    missing_keys = set(cls.attributes_annotations.keys()) - set(attributes.keys())
    for name, attribute in attributes.items():
        assert name in cls.attributes_annotations, "{} has no attribute \"{}\"".format(cls.__name__, name)
        assert isinstance(attribute, cls.attributes_annotations[name]), "Attribute {} of {} must be of type {}".format(name, cls.__name__, cls.attributes_annotations[name].__name__)

    full_attributes = attributes
    for name in missing_keys:
        assert cls.attributes_defaults[name] is not MISSING, \
            "{} requires a named attribute {}: {} because no default is specified.".format(cls.__name__, name, cls.attributes_annotations[name])
        full_attributes[name] = cls.attributes_defaults[name]
    optype = OpType(*operands, **full_attributes, name=str(cls.__name__))

    # retrospectively set operands as fields with keyword name
    for operand, operand_name in zip(operands, cls.operands):
        setattr(optype, operand_name, operand)

    return optype


def add_shape_func(op_name):
    def register_func(func):
        _SHAPE_FUNCS[op_name] = func
        return func
    return register_func


def add_conversion(op_name, target):
    def wrapper(func):
        _CONVERSIONS[op_name] = func
        return func
    return wrapper


# Register default pass-through shape function
@add_shape_func('Default')
def default_shape(op: OpType):
    input_tensor = op.operands[0].output_tensor()
    return input_tensor.tensor_type
