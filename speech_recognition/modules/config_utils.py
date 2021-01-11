import json
import os
import logging

import torch
import torch.nn as nn

from hydra.utils import instantiate


def get_loss_function(model, config):

    ce = nn.CrossEntropyLoss()

    def ce_loss_func(scores, labels):
        scores = scores.view(scores.size(0), -1)
        return ce(scores, labels)

    criterion = ce_loss_func

    try:
        criterion = model.get_loss_function()
    except Exception as e:
        logging.info(str(e))
        criterion = nn.CrossEntropyLoss()

    return criterion


def get_model(config):
    if "_target_" in config:
        model = instantiate(config)
    else:
        model = _locate(config.cls)(config)
    return model


def dump_config(output_dir, config):
    """Dumps the configuration to json format

    Creates file config.json in output_dir

    Parameters
    ----------
    output_dir : str
       Output directory
    config  : dict
       Configuration to dump
    """

    with open(os.path.join(output_dir, "config.json"), "w") as o:
        s = json.dumps(dict(config), default=lambda x: str(x), indent=4, sort_keys=True)
        o.write(s)


def save_model(output_dir, model):
    """Creates serialization of the model for later inference, evaluation

    Creates the following files:

    - model.pt: Serialized version of network parameters in pytorch
    - model.json: Serialized version of network parameters in json format
    - model.onnx: full model including paramters in onnx format

    Parameters
    ----------

    output_dir : str
        Directory to put serialized models
    model : LightningModule
        Model to serialize
    """
    msglogger = logging.getLogger()

    msglogger.info("saving weights to json...")
    filename = os.path.join(output_dir, "model.json")
    state_dict = model.state_dict()
    with open(filename, "w") as f:
        json.dump(state_dict, f, default=lambda x: x.tolist(), indent=2)

    msglogger.info("saving onnx...")
    try:
        dummy_input = model.example_feature_array

        torch.onnx.export(
            model.model,
            dummy_input,
            os.path.join(output_dir, "model.onnx"),
            verbose=False,
            opset_version=12,
        )
    except Exception as e:
        msglogger.error("Could not export onnx model ...\n {}".format(str(e)))


def _fullname(cls):
    # See: https://stackoverflow.com/questions/2020014/get-fully-qualified-class-name-of-an-object-in-python
    # o.__module__ + "." + o.__class__.__qualname__ is an example in
    # this context of H.L. Mencken's "neat, plausible, and wrong."
    # Python makes no guarantees as to whether the __module__ special
    # attribute is defined, so we take a more circumspect approach.
    # Alas, the module name is explicitly excluded from __qualname__
    # in Python 3.

    module = cls.__module__
    if module is None or module == str.__module__:
        return cls.__name__  # Avoid reporting __builtin__
    else:
        return module + "." + cls.__name__


def _locate(path):
    """
    Locate an object by name or dotted path, importing as necessary.
    This is similar to the pydoc function `locate`, except that it checks for
    the module from the given path from back to front.

    #FIXME: this should be removed if and when hydra is adapted as configuration manager
    """
    import builtins
    from importlib import import_module

    logger = logging.getLogger()

    parts = [part for part in path.split(".") if part]
    module = None
    for n in reversed(range(len(parts))):
        try:
            module = import_module(".".join(parts[:n]))
        except Exception as e:
            if n == 0:
                logger.error(f"Error loading module {path} : {e}")
                raise e
            continue
        if module:
            break
    if module:
        obj = module
    else:
        obj = builtins
    for part in parts[n:]:
        if not hasattr(obj, part):
            raise ValueError(
                f"Error finding attribute ({part}) in class ({obj.__name__}): {path}"
            )
        obj = getattr(obj, part)
    if isinstance(obj, type):
        obj_type: type = obj
        return obj_type
    elif callable(obj):
        obj_callable: Callable[..., Any] = obj
        return obj_callable
    else:
        # dummy case
        raise ValueError(f"Invalid type ({type(obj)}) found for {path}")
