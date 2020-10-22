import json
import os

import torch
import torch.nn as nn

from .utils import _locate


def get_loss_function(model, config):

    ce = nn.CrossEntropyLoss()

    def ce_loss_func(scores, labels):
        scores = scores.view(scores.size(0), -1)
        return ce(scores, labels)

    criterion = ce_loss_func

    try:
        criterion = model.get_loss_function()
    except Exception as e:
        print(str(e))
        if "loss" in config:
            if config["loss"] == "cross_entropy":
                criterion = nn.CrossEntropyLoss()
            elif config["loss"] == "ctc":
                criterion = ce_loss_func
            else:
                raise Exception(
                    "Loss function not supported: {}".format(config["loss"])
                )

    return criterion


def get_model(config):
    model = _locate(config["model_class"])(config)
    return model


def reset_symlink(src, dest):
    if os.path.exists(dest):
        os.unlink(dest)
    os.symlink(src, dest)


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


def save_model(
    output_dir, model, test_set=None, config=None, model_prefix="", msglogger=None
):
    """ Creates serialization of the model for later inference, evaluation

    Creates the following files:

    - model.pt: Serialized version of network parameters in pytorch
    - model.json: Serialized version of network parameters in json format
    - model.onnx: full model including paramters in onnx format

    Parameters
    ----------

    output_dir : str
        Directory to put serialized models
    model : torch.nn.Module
        Model to serialize
    test_set : dataset.SpeechDataset
        DataSet used to derive dummy input to use for onnx export.
        If None no onnx will be generated
    """

    # TODO model save doesnt work "AttributeError: model has no attribute save"
    # msglogger.info("saving best model...")
    # model.save(os.path.join(output_dir, model_prefix+"model.pt"))

    msglogger.info("saving weights to json...")
    filename = os.path.join(output_dir, model_prefix + "model.json")
    state_dict = model.state_dict()
    with open(filename, "w") as f:
        json.dump(state_dict, f, default=lambda x: x.tolist(), indent=2)

    msglogger.info("saving onnx...")
    try:
        dummy_width, dummy_height = test_set.width, test_set.height
        dummy_input = torch.randn((1, dummy_height, dummy_width))

        if config["cuda"]:
            dummy_input = dummy_input.cuda()

        torch.onnx.export(
            model,
            dummy_input,
            os.path.join(output_dir, model_prefix + "model.onnx"),
            verbose=False,
        )
    except Exception as e:
        msglogger.error("Could not export onnx model ...\n {}".format(str(e)))
