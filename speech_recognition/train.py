from collections import ChainMap, OrderedDict
from .config import ConfigBuilder, ConfigOption
import argparse
import os
import random
import sys
import json
import time
import math

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import itertools

from . import models as mod
from . import dataset
from .utils import set_seed, config_pylogger, log_execution_env_state

sys.path.append(os.path.join(os.path.dirname(__file__), "distiller"))

import distiller
from distiller.data_loggers import *
import distiller.apputils as apputils
import torchnet.meter as tnt

from .summaries import *


msglogger = None

def get_loss_function(config):
     
    criterion = nn.CrossEntropyLoss()
    if "loss" in config:
        if config["loss"] == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif config["loss"] == "ctc":
            criterion = nn.CTCLoss()
        else:
            raise Exception("Loss function not supported {}".format(config["loss"]))
            
    return criterion
            
def get_output_dir(model_name, config):
    
    output_dir = os.path.join(config["output_dir"], config["experiment_id"], model_name)

    if config["compress"]:
        compressed_name = config["compress"]
        compressed_name = os.path.splitext(os.path.basename(compressed_name))[0]
        output_dir = os.path.join(output_dir, compressed_name)

    output_dir = os.path.abspath(output_dir)
        
    return output_dir

def get_eval(scores, labels, loss):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()

    return accuracy.item(), loss

def validate(data_loader, model, criterion, config, loggers=[], epoch=-1):
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1,))
    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    confusion = tnt.ConfusionMeter(config["n_labels"])

    total_steps = total_samples // batch_size
    log_every = total_steps // 10


    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    model.eval()

    end = time.time()

    for validation_step, (inputs, target) in enumerate(data_loader):
        with torch.no_grad():
            if config["cuda"]:
                inputs, target = inputs.cuda(), target.cuda()
            # compute output from model
            output = model(inputs)

            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.data, target)
            confusion.add(output.data, target)

        batch_time.add(time.time() - end)
        end = time.time()
        
        steps_completed = (validation_step+1)
    
        stats = ('Performance/Validation/',
                 OrderedDict([('Loss', losses['objective_loss'].mean),
                              ('Top1', classerr.value(1))]))

        if steps_completed % log_every == 0:
            distiller.log_training_progress(stats, None, epoch, steps_completed,
                                            total_steps, log_every, loggers)

    msglogger.info('==> Top1: %.3f      Loss: %.3f\n',
                   classerr.value(1), losses['objective_loss'].mean)

    msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))

    return classerr.value(1), losses['objective_loss'].mean

def get_model(config, model=None):
    if not model:
        model = config["model_class"](config)
        if config["input_file"]:
            model.load(config["input_file"])
        
    return model

def evaluate(model_name, config, model=None, test_loader=None, loggers=[]):
    global msglogger
    if not msglogger:
        output_dir = get_output_dir(model_name, config)
        msglogger = config_pylogger('logging.conf', "eval", output_dir)

    if not loggers:
        loggers = [PythonLogger(msglogger)]
        
    msglogger.info("Evaluating network")
    
    if not test_loader:
        _, _, test_set = config["dataset_cls"].splits(config)
        test_loader = data.DataLoader(test_set, batch_size=1)
        
    model = get_model(config, model)
        
    if config["cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    criterion = get_loss_function(config)
    
    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1,))
    batch_time = tnt.AverageValueMeter()
    total_samples = len(test_loader.sampler)
    batch_size = test_loader.batch_size
    confusion = tnt.ConfusionMeter(config["n_labels"])

    total_steps = total_samples // batch_size
    log_every = total_steps // 10

    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)
        
    model.eval()
    

    end = time.time()

    # Print network statistics
    dummy_input, _ = next(iter(test_loader))
    model.eval()
    if config["cuda"]:
        dummy_input.cuda()
        model.cuda()
        
    performance_summary = model_summary(model, dummy_input, 'performance')
    
    for test_step, (model_in, target) in enumerate(test_loader):
        with torch.no_grad():
            model_in = Variable(model_in)
            target = Variable(target)
            if config["cuda"]:
                model_in = model_in.cuda()
                target = target.cuda()
             
            output = model(model_in)
            
            loss = criterion(output, target)
            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())
            classerr.add(output.data, target)
            confusion.add(output.data, target)

        batch_time.add(time.time() - end)
        end = time.time()
        
        steps_completed = (test_step+1)
    
        stats = ('Performance/Test/',
                 OrderedDict([('Loss', losses['objective_loss'].mean),
                              ('Top1', classerr.value(1))]))

        if steps_completed % log_every == 0:
            distiller.log_training_progress(stats, None, 0, steps_completed,
                                            total_steps, log_every, loggers)

    msglogger.info('==> Top1: %.3f      Loss: %.3f\n',
                   classerr.value(1), losses['objective_loss'].mean)

    msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))
    return classerr.value(1)


def dump_config(output_dir, config):
    with open(os.path.join(output_dir, 'config.json'), "w") as o:
              s = json.dumps(dict(config), default=lambda x: str(x), indent=4, sort_keys=True)
              o.write(s)

def train(model_name, config):
    global msglogger

    output_dir = get_output_dir(model_name, config)
    
    #Configure logging
    msglogger = config_pylogger('logging.conf', "train", output_dir)
    pylogger = PythonLogger(msglogger)
    loggers  = [pylogger]  
    if config["tblogger"]:
        tblogger = TensorBoardLogger(msglogger.logdir)
        tblogger.log_gradients = True
        loggers.append(tblogger)

    log_execution_env_state(distiller_gitroot=os.path.join(os.path.dirname(__file__), "distiller"))    

    
    print("All information will be saved to: ", output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    train_set, dev_set, test_set = config["dataset_cls"].splits(config)

    config["width"] = train_set.width
    config["height"] = train_set.height

    dump_config(output_dir, config)

    model = get_model(config)
    
    if config["cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    # Setup optimizers
    optimizer = None
    if config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config["lr"], 
                                    nesterov=config["use_nesterov"],
                                    weight_decay=config["weight_decay"], 
                                    momentum=config["momentum"])
    elif config["optimizer"] == "adadelta":
        optimizer = torch.optim.Adadelta(model.parameters(),
                                         lr=config["lr"],
                                         rho=config["opt_rho"],
                                         eps=config["opt_eps"],
                                         weight_decay=config["weight_decay"])
    elif config["optimizer"] == "adagrad":
        optimizer = torch.optim.Adagrad(model.parameters(),
                                        lr=config["lr"],
                                        lr_decay=config["lr_decay"],
                                        weight_decay=config["weight_decay"])

    elif config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=config["lr"],
                                     betas=config["opt_betas"],
                                     eps=config["opt_eps"],
                                     weight_decay=config["weight_decay"],
                                     amsgrad=config["use_amsgrad"])
    else:
        raise Exception("Unknown Optimizer: {}".format(config["optimizer"]))
        
    sched_idx = 0
    criterion = get_loss_function(config)
    max_acc = 0

    n_epochs = config["n_epochs"]
    
    # Setup learning rate optimizer
    lr_scheduler = config["lr_scheduler"]
    scheduler = None
    if lr_scheduler == "step":
        gamma = config["lr_gamma"]
        stepsize = config["lr_stepsize"]
        if stepsize == 0:
            stepsize = max(10, n_epochs // 3)
        
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=stepsize, gamma=gamma)
        
    elif lr_scheduler == "multistep":
        gamma = config["lr_gamma"]
        steps = config["lr_steps"]
        if steps == [0]:
            steps = itertools.count(max(1, n_epochs//10), max(1, n_epochs//10))

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                         steps,
                                                         gamma=gamma)
            
    elif lr_scheduler == "exponential":
        gamma = config["lr_gamma"]
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)  
    elif lr_scheduler == "plateau":
        gamma = config["lr_gamma"]
        patience = config["lr_patience"]
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               mode='min',
                                                               factor=gamma,
                                                               patience=patience,
                                                               threshold=0.0001,
                                                               threshold_mode='rel',
                                                               cooldown=0,
                                                               min_lr=0,
                                                               eps=1e-08)

    else:
        raise Exception("Unknown learing rate scheduler: {}".format(lr_scheduler))
    
    # setup datasets
    train_batch_size = config["batch_size"]
    train_loader = data.DataLoader(train_set, batch_size=train_batch_size, shuffle=True, drop_last=True)
    dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
    test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True)

    # Print network statistics
    dummy_input, _ = next(iter(test_loader))
    model.eval()
    if config["cuda"]:
        dummy_input.cuda()
        model.cuda()

    draw_classifier_to_file(model,
                            os.path.join(output_dir, 'model.png'),
                            dummy_input)

    performance_summary = model_summary(model, dummy_input, 'performance')

    # Setup distiller for model minimization
    compression_scheduler = None
    if config["compress"]:
        print("Activating compression scheduler")

        compression_scheduler = distiller.file_config(model, optimizer, config["compress"])
    if config["cuda"]:
        model.cuda()

    
    # iteration counters 
    step_no = 0
    batches_per_epoch = len(train_loader)
    log_every = max(1, batches_per_epoch // 15)
    last_log = 0
    
    for epoch_idx in range(n_epochs):
        msglogger.info("Training epoch {} of {}".format(epoch_idx, config["n_epochs"]))
        
        if compression_scheduler is not None:
            compression_scheduler.on_epoch_begin(epoch_idx)

        batch_time = tnt.AverageValueMeter()
        end = time.time()
        for batch_idx, (model_in, labels) in enumerate(train_loader):
            model.train()
            if compression_scheduler is not None:
                compression_scheduler.on_minibatch_begin(epoch_idx, batch_idx, batches_per_epoch)
                
            optimizer.zero_grad()
            if config["cuda"]:
                model_in = model_in.cuda()
                labels = labels.cuda()
            model_in = Variable(model_in)
            scores = model(model_in)
            
            labels = Variable(labels)
            if config["loss"] == "ctc":
                scores = scores.view(scores.size(0), scores.size(1), -1)
                scores = scores.permute(2,0,1)
                scores = scores.view(scores.size(0), scores.size(1), -1)

                input_lengths = torch.Tensor([scores.size(0)] * scores.size(1)).long()
                label_lengths = torch.Tensor([1] * scores.size(1)).long()
                scores = torch.nn.functional.log_softmax(scores, dim=2)
                labels = labels.unsqueeze(1)
                
                loss = criterion(scores, labels, input_lengths, label_lengths)
            else:
                scores = scores.view(scores.size(0), -1)
                loss = criterion(scores, labels)

            
            if compression_scheduler is not None:
                compression_scheduler.before_backward_pass(epoch_idx, batch_idx, batches_per_epoch, loss)
            loss.backward()
            optimizer.step()
            if compression_scheduler is not None:
                compression_scheduler.on_minibatch_end(epoch_idx, batch_idx, batches_per_epoch)

            #Log statistics
            stats_dict = OrderedDict()
            batch_time.add(time.time() - end)
            step_no += 1
            
            if last_log + log_every <= step_no:
                scalar_accuracy, scalar_loss = get_eval(scores, labels, loss)
        
                last_log = step_no
                stats_dict["Accuracy"] = scalar_accuracy
                stats_dict["Loss"] = scalar_loss
                stats_dict["Time"] = batch_time.mean
                stats_dict['LR'] = optimizer.param_groups[0]['lr']
                stats = ('Peformance/Training/', stats_dict)
                params = model.named_parameters()
                distiller.log_training_progress(stats,
                                                params,
                                                epoch_idx,
                                                batch_idx,
                                                batches_per_epoch,
                                                log_every,
                                                loggers)
                 
         
            end = time.time()

        msglogger.info("Validation epoch {} of {}".format(epoch_idx, config["n_epochs"]))
        avg_acc, avg_loss = validate(dev_loader, model, criterion, config, loggers=loggers, epoch=epoch_idx)    
        

        if avg_acc > max_acc:
            msglogger.info("saving best model...")
            max_acc = avg_acc
            model.save(os.path.join(output_dir, "model.pt"))


            msglogger.info("saving onnx...")
            try:
                model_in, label = next(iter(test_loader), (None, None))
                if config["cuda"]:
                    model_in = model_in.cuda()
                model.save_onnx(os.path.join(output_dir, "model.onnx"), model_in)
            except Exception as e:
                msglogger.error("Could not export onnx model ...\n {}".format(str(e)))
                
        if scheduler is not None:
            if type(lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                scheduler.step(avg_loss)
            else:
                scheduler.step()
                
        if compression_scheduler is not None:
            compression_scheduler.on_epoch_begin(epoch_idx)

    msglogger.info("Running final test")
    model.load(os.path.join(output_dir, "model.pt"))
    test_accuracy = evaluate(model_name, config, model, test_loader)

def build_config(extra_config={}):
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "trained_models")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="ekut-raw-cnn3", type=str)
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--dataset", choices=["keywords", "hotword"], default="keywords", type=str)
    config, _ = parser.parse_known_args()

    model_name = config.model
    dataset_name = config.dataset
    
    default_config = {}
    if config.config:
        with open(config.config, 'r') as f:  
            default_config = json.load(f)
            model_name = default_config["model_name"]

            #Delete model from config for now to avoid showing
            #them as commandline otions
            if "model_name" in default_config:
                del default_config["model_name"]
            else:
                print("Your model config does not include a model_name")
                print(" these configurations are not loadable")
                sys.exit(-1)
            if "model_class" in default_config:
                del default_config["model_class"]
            if "type" in default_config:
                del default_config["type"]
            if "dataset" in default_config:
                dataset_name = default_config["dataset"]
                del default_config["dataset"]
            if "dataset_cls" in default_config:
                del default_config["dataset_cls"]
            
    global_config = dict(cuda=ConfigOption(default=True,
                                           desc="Disable cuda"),
                         n_epochs=ConfigOption(default=500,
                                               desc="Number of epochs for training"),

                         optimizer=ConfigOption(default="sgd",
                                                desc="Optimizer to choose",
                                                category="Optimizer Config", 
                                                choices=["sgd",
                                                         "adadelta",
                                                         "adagrad",
                                                         "adam"]),
                         
                         opt_rho     = ConfigOption(category="Optimizer Config",
                                                    desc="Parameter rho for Adadelta optimizer",
                                                    default=0.9),
                         opt_eps     = ConfigOption(category="Optimizer Config",
                                                    desc="Paramter eps for Adadelta and Adam",
                                                    default=1e-06),
                         lr_decay    = ConfigOption(category="Optimizer Config",
                                                    desc="Parameter lr_decay for optimizers",
                                                    default=0),
                         use_amsgrad = ConfigOption(category="Optimizer Config",
                                                    desc="Use amsgrad with Adam optimzer",
                                                    default=False),
                         opt_betas   = ConfigOption(category="Optimizer Config",
                                                    desc="Parameter betas for Adam optimizer",
                                                    default=[0.9, 0.999]),
                         momentum    = ConfigOption(category="Optimizer Config",
                                                    desc="Momentum for SGD optimizer",
                                                    default=0.9),
                         weight_decay= ConfigOption(category="Optimizer Config",
                                                    desc="Weight decay for optimizer",
                                                    default=0.00001),
                         use_nesterov= ConfigOption(category="Optimizer Config",
                                                    desc="Use nesterov momentum with SGD optimizer",
                                                    default=False),
                         lr           = ConfigOption(category="Learning Rate Config",
                                                     desc="Initial Learining Rate",
                                                     default=0.1),
                         lr_scheduler = ConfigOption(category="Learning Rate Config",
                                                     desc="Learning Rate Scheduler to use",
                                                     choices=["step", "multistep", "exponential", "plateau"], 
                                                     default="step"),
                         lr_gamma     = ConfigOption(category="Learning Rate Config",
                                                     desc="Parameter gamma for lr scheduler",
                                                     default=0.1),
                         lr_stepsize  = ConfigOption(category="Learning Rate Config",
                                                     desc="Stepsize for step scheduler",
                                                     default=0),
                         lr_steps     = ConfigOption(category="Learning Rate Config",
                                                     desc="List of steps for multistep scheduler",
                                                     default=[0]),
                         lr_patience  = ConfigOption(category="Learning Rate Config",
                                                     desc="Parameter patience for plateau scheduler",
                                                     default=10),
                         
                         batch_size=ConfigOption(default=128,
                                                 desc="Default minibatch size for training"),
                         seed=ConfigOption(default=0,
                                           desc="Seed for Random number generators"),
                         input_file=ConfigOption(default="",
                                                 desc="Input model file for finetuning (.pth) or code generation (.onnx)"),
                         output_dir=ConfigOption(default=output_dir,
                                                 desc="Toplevel directory for output of trained models and logs"),
                         gpu_no=ConfigOption(default=0,
                                             desc="Number of GPU to use for training"),
                         compress=ConfigOption(default="",
                                               desc="YAML config file for nervana distiller"),
                         tblogger=ConfigOption(default=False,
                                               desc="Enable logging of learning progress and network parameter statistics to Tensorboard"),
                         experiment_id=ConfigOption(default="test",
                                                    desc="Unique id to identify the experiment, overwrites all output files with same experiment id, output_dir, and model_name"))
    
    mod_cls = mod.find_model(model_name)
    dataset_cls = dataset.find_dataset(dataset_name)
    builder = ConfigBuilder(
        default_config,
        mod.find_config(model_name),
        dataset_cls.default_config(),
        global_config,
        extra_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    config["model_name"] = model_name
    config["dataset"] = dataset_name
    config["dataset_cls"] = dataset_cls 
    
    return (model_name, config)
    
def main():
    model_name, config = build_config()
    
    #TODO: Check if results are actually reproducible when seeds are set
    set_seed(config)

    if config["type"] == "train":
        train(model_name, config)
    elif config["type"] == "check_sanity":
        raise Exception("TODO: Implement sanity check")
    elif config["type"] == "eval":
        evaluate(model_name, config)

if __name__ == "__main__":
    main()
    
