from collections import ChainMap, OrderedDict, defaultdict
from .config import ConfigBuilder, ConfigOption
import argparse
import os
import random
import sys
import json
import time
import math
import hashlib
import csv
import fcntl

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data
import itertools

from . decoder import Decoder
from . import models as mod
from . import dataset
from .utils import set_seed, config_pylogger, log_execution_env_state, EarlyStopping

sys.path.append(os.path.join(os.path.dirname(__file__), "distiller"))

import distiller
from distiller.data_loggers import *
import distiller.apputils as apputils
import torchnet.meter as tnt
from tabulate import tabulate

from .summaries import *


msglogger = None

def get_lr_scheduler(config, optimizer):
    n_epochs = config["n_epochs"]
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

    return scheduler

def get_optimizer(config, model):
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
    elif config["optimizer"] == "rmsprop":
        optimizer = torch.optim.RMSprop(model.parameters(),
                                        lr=config["lr"],
                                        alpha=config["opt_alpha"],
                                        eps=config["opt_eps"],
                                        weight_decay=config["weight_decay"],
                                        momentum=config["momentum"])
    else:
        raise Exception("Unknown Optimizer: {}".format(config["optimizer"]))

    return optimizer

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

def get_config_logdir(model_name, config):
    return os.path.join(get_output_dir(model_name, config), "configs", config["config_hash"])

def get_model(config, config2=None, model=None, vad_keyword = 0):
    if not model:
        if vad_keyword == 0:
            model = config["model_class"](config)
            if config["input_file"]:
                model.load(config["input_file"])
        elif vad_keyword == 1:
            model = config2["model_class"](config2)
            if config["input_file_vad"]:
                model.load(config["input_file_vad"])
        else:
            model = config2["model_class"](config2)
            if config["input_file_keyword"]:
                model.load(config["input_file_keyword"])
    return model

def reset_symlink(src, dest):
    if os.path.exists(dest):
        os.unlink(dest)
    os.symlink(src, dest)

def dump_config(output_dir, config):
    """ Dumps the configuration to json format

    Creates file config.json in output_dir

    Parameters
    ----------
    output_dir : str
       Output directory
    config  : dict
       Configuration to dump
    """

    with open(os.path.join(output_dir, 'config.json'), "w") as o:
              s = json.dumps(dict(config), default=lambda x: str(x), indent=4, sort_keys=True)
              o.write(s)

def save_model(output_dir, model, test_set=None, config=None):
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
    msglogger.info("saving best model...")
    model.save(os.path.join(output_dir, "model.pt"))

    msglogger.info("saving weights to json...")
    filename = os.path.join(output_dir, "model.json")
    state_dict = model.state_dict()
    with open(filename, "w") as f:
        json.dump(state_dict, f, default=lambda x: x.tolist(), indent=2)


    msglogger.info("saving onnx...")
    try:
        dummy_width, dummy_height = test_set.width, test_set.height
        dummy_input = torch.randn((1, dummy_height, dummy_width))

        if config["cuda"]:
            dummy_input = dummy_input.cuda()

        torch.onnx.export(model,
                          dummy_input,
                          os.path.join(output_dir, "model.onnx"),
                          verbose=False)
    except Exception as e:
        msglogger.error("Could not export onnx model ...\n {}".format(str(e)))

def validate(data_loader, model, model2, criterion, config, config_vad, config_keywords, loggers=[], epoch=-1):

    combined_evaluation = True
    if model2 is None:
        combined_evaluation = False

    losses = {'objective_loss': tnt.AverageValueMeter()}
    classerr = tnt.ClassErrorMeter(accuracy=True, topk=(1,))
    batch_time = tnt.AverageValueMeter()
    total_samples = len(data_loader.sampler)
    batch_size = data_loader.batch_size
    n_labels = config["n_labels"]
    if combined_evaluation:
        batch_size = 1
        n_labels = n_labels + 1
    confusion = tnt.ConfusionMeter(config["n_labels"])
    total_steps = total_samples // batch_size
    log_every = total_steps // 10

    msglogger.info('%d samples (%d per mini-batch)', total_samples, batch_size)

    model.eval()
    if combined_evaluation:
        model2.eval()


    end = time.time()

    for validation_step, (inputs, in_lengths, targets, target_lengths) in enumerate(data_loader):
        with torch.no_grad():
            if config["cuda"]:
                inputs, targets = inputs.cuda(), targets.cuda()
            # compute output from model
            if combined_evaluation:
                output_vad = model(inputs)
                if output_vad.max(1)[1] == 0:
                    print("classified as noise", targets)
                    output = torch.zeros(1, config["n_labels"]+1)
                    output[0,config["n_labels"]] = 1
                else:
                    print("not classified as noise", targets)
                    output_keyword = model2(inputs)
                    output = torch.cat((output_keyword,torch.zeros(1,1)), dim=1)
            else:
                output = model(inputs)

            if config["loss"] == "ctc":
                loss = criterion(output, targets)
            else:
                targets=targets.view(-1)
                output=output.view(output.size(0), -1)
                loss = criterion(output, targets)



                classerr.add(output.data, targets)
                confusion.add(output.data, targets)


            # measure accuracy and record loss
            losses['objective_loss'].add(loss.item())

        batch_time.add(time.time() - end)
        end = time.time()

        steps_completed = (validation_step+1)

        stats = ('Performance/Validation/',
                 OrderedDict([('Loss', losses['objective_loss'].mean),
                              ('Accuracy', classerr.value(1))]))

        if steps_completed % log_every == 0:
            distiller.log_training_progress(stats, None, epoch, steps_completed,
                                            total_steps, log_every, loggers)

    msglogger.info('==> Accuracy: %.3f      Loss: %.3f\n',
                   classerr.value(1), losses['objective_loss'].mean)

    msglogger.info('==> Confusion:\n%s\n', str(confusion.value()))

    return classerr.value(1), losses['objective_loss'].mean, confusion.value()

def evaluate(model_name, config, config_vad=None, config_keyword=None, model=None, test_set=None, loggers=[]):

    # combined evaluation of vad and keyword spotting
    combined_evaluation = True
    if config_vad is None:
        combined_evaluation = False

    global msglogger
    if not msglogger:
        log_dir = get_config_logdir(model_name, config)
        output_dir = get_output_dir(model_name, config)
        msglogger = config_pylogger('logging.conf', "eval", log_dir)
        
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        reset_symlink(os.path.join(log_dir, "eval.log"), os.path.join(output_dir, "eval.log"))    
    
    if not loggers:
        loggers = [PythonLogger(msglogger)]

    msglogger.info("Evaluating network")

    if not test_set:
        _, _, test_set = config["dataset_cls"].splits(config)

    test_loader = data.DataLoader(test_set, batch_size=1) 

    if model is None:
        config["width"] = test_set.width
        config["height"] = test_set.height

        if combined_evaluation:
            model_vad = get_model(config, config_vad, vad_keyword = 1)
            model_keyword = get_model(config, config_keyword, vad_keyword = 2)
        else:
            model = get_model(config)

    criterion = get_loss_function(config)

    # Print network statistics
    dummy_width, dummy_height = test_set.width, test_set.height
    dummy_input = torch.randn((1, dummy_height, dummy_width))

    if combined_evaluation:
        model_vad.eval()
        model_keyword.eval()

        if config["cuda"]:
            dummy_input.cuda()
            model_vad.cuda()
            model_keyword.cuda()

        performance_summary_vad = model_summary(model_vad, dummy_input, 'performance')
        performance_summary_keyword = model_summary(model_keyword, dummy_input, 'performance')
        accuracy, loss, confusion_matrix = validate(test_loader, model_vad, model_keyword, criterion, config, config_vad, config_keyword, loggers)
    else:
        model.eval()
        if config["cuda"]:
            dummy_input.cuda()
            model.cuda()

        performance_summary = model_summary(model, dummy_input, 'performance')
        accuracy, loss, confusion_matrix = validate(test_loader, model, None, criterion, config, None, None, loggers)

    msglogger.info('==> Per class accuracy metrics')
    
    FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
    FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
    TP = np.diag(confusion_matrix)
    TN = confusion_matrix.sum() - (FP + FN + TP)
     
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP/(TP+FN)
    # Specificity or true negative rate
    TNR = TN/(TN+FP) 
    # Precision or positive predictive value
    PPV = TP/(TP+FP)
    # Negative predictive value
    NPV = TN/(TN+FN)
    # Fall out or false positive rate
    FPR = FP/(FP+TN)
    # False negative rate
    FNR = FN/(TP+FN)
    # False discovery rate
    FDR = FP/(TP+FP)
     
    # Overall accuracy
    ACC = (TP+TN)/(TP+FP+FN+TN)
    
    accuracy_table = []
    for num in range(len(TPR)):
        accuracy_table.append([test_set.label_names[num],
                               TPR[num], TNR[num], PPV[num],
                               NPV[num], FPR[num], FNR[num],
                               FDR[num], ACC[num]])
    
    msglogger.info(tabulate(accuracy_table, headers=["Class", "TPR",
                                                     "TNR", "PPV",
                                                     "NPV", "FPR",
                                                     "FNR", "FDR",
                                                     "ACC"]))
    
    
    return  accuracy, loss, confusion_matrix

def train(model_name, config, check_sanity=False):
    global msglogger

    output_dir = get_output_dir(model_name, config)
    log_dir = get_config_logdir(model_name, config)
    
    #Configure logging
    log_name = "train" if not check_sanity else "sanity_check"
    msglogger = config_pylogger('logging.conf', log_name, log_dir)
    pylogger = PythonLogger(msglogger)
    loggers  = [pylogger]
    if config["tblogger"]:
        tblogger = TensorBoardLogger(msglogger.logdir)
        tblogger.log_gradients = True
        loggers.append(tblogger)

    log_execution_env_state(distiller_gitroot=os.path.join(os.path.dirname(__file__), "distiller"))


    print("All information will be saved to: ", output_dir, "logdir:", log_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    reset_symlink(os.path.join(log_dir, "train.log"), os.path.join(output_dir, "train.log"))
    
        
    dump_config(log_dir, config)
    reset_symlink(os.path.join(log_dir, "config.json"), os.path.join(output_dir, "config.json"))
        
    csv_log_name = os.path.join(log_dir, "train.csv")
    csv_log_file = open(csv_log_name, "w")
    csv_log_writer = csv.DictWriter(csv_log_file, fieldnames=["Phase", "Epoch", "Accuracy", "Loss", "Macs", "Weights", "LR"])
    csv_log_writer.writeheader()
    reset_symlink(csv_log_name, os.path.join(output_dir, "train.csv"))
    
    train_set, dev_set, test_set = config["dataset_cls"].splits(config)

    msglogger.info("Dataset config:")
    msglogger.info("  train: %d", len(train_set))
    class_nums = train_set.get_class_nums()
    for k in sorted(class_nums.keys()):
        msglogger.info("     {}: {}".format(train_set.label_names[k], class_nums[k]))

    msglogger.info("  dev:   %d", len(dev_set))
    class_nums = dev_set.get_class_nums()
    for k in sorted(class_nums.keys()):
        msglogger.info("     {}: {}".format(dev_set.label_names[k], class_nums[k]))

    msglogger.info("  test:  %d", len(test_set))
    class_nums = test_set.get_class_nums()
    for k in sorted(class_nums.keys()):
        msglogger.info("     {}: {}".format(test_set.label_names[k], class_nums[k]))

    msglogger.info("  total: %d", len(train_set)+len(dev_set)+len(test_set))
    msglogger.info("")


    config["width"] = train_set.width
    config["height"] = train_set.height


    model = get_model(config)

    if config["cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()

    # Setup optimizers
    optimizer = get_optimizer(config, model)

    criterion = get_loss_function(config)

    # Setup learning rate optimizer
    lr_scheduler = get_lr_scheduler(config, optimizer)

    # Setup early stopping
    early_stopping = EarlyStopping(config["early_stopping"])

    # setup datasets

    collate_fn = dataset.ctc_collate_fn #if train_set.loss_function == "ctc" else None
    train_batch_size = config["batch_size"]
    train_loader = data.DataLoader(train_set,
                                   batch_size=train_batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   collate_fn=collate_fn)


    if check_sanity:
        indices = (np.random.random(20)*len(train_set)).astype(int)
        train_loader = data.DataLoader(train_set,
                                       batch_size=20,
                                       sampler=torch.utils.data.SubsetRandomSampler(indices),
                                       drop_last=True,
                                       collate_fn=collate_fn)

    dev_loader = data.DataLoader(dev_set,
                                 batch_size=min(len(dev_set), 16),
                                 shuffle=False,
                                 collate_fn=collate_fn)

    test_loader = data.DataLoader(test_set,
                                  batch_size=1,
                                  shuffle=False,
                                  collate_fn=collate_fn)

    # Setup Decoder
    decoder = Decoder(train_set.label_names)

    # Print network statistics
    dummy_width, dummy_height = test_set.width, test_set.height
    dummy_input = torch.randn((1, dummy_height, dummy_width))
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
        compression_scheduler = distiller.file_config(model,
                                                      optimizer,
                                                      config["compress"])

    if config["cuda"]:
        model.cuda()

    sched_idx = 0
    n_epochs = config["n_epochs"]


    # iteration counters
    step_no = 0
    batches_per_epoch = len(train_loader)
    log_every = max(1, batches_per_epoch // 15)
    last_log = 0
    max_acc = 0
    last_lr = config["lr"]

    for epoch_idx in range(n_epochs):
        msglogger.info("Training epoch {} of {}".format(epoch_idx, config["n_epochs"]))

        if compression_scheduler is not None:
            compression_scheduler.on_epoch_begin(epoch_idx)

        avg_training_loss = tnt.AverageValueMeter()
        avg_training_accuracy = tnt.AverageValueMeter()

        batch_time = tnt.AverageValueMeter()
        end = time.time()
        for batch_idx, (model_in, in_lengths, labels, label_lengths) in enumerate(train_loader):
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
                scores = scores.permute(2,0,1,3)
                scores = scores.view(scores.size(0), scores.size(1), -1)
                scores = scores.log_softmax(2)

                in_lengths = in_lengths / max(in_lengths) * scores.size(0)
                in_lengths = in_lengths.round()
                in_lengths = in_lengths.int()

                print("\nLoss inputs:")
                print(scores)
                print(labels)
                print(in_lengths)
                print(label_lengths)


                loss = criterion(scores, labels, in_lengths, label_lengths)

                print("loss:", loss.item())

                scalar_loss = loss.item()
                error, cer = decoder.calculate_error(scores, in_lengths,
                                                     labels, label_lengths)
                scalar_accuracy = 1.0 - error
            else:
                scores = scores.view(scores.size(0), -1)
                labels = labels.view(-1)

                loss = criterion(scores, labels)
                scalar_loss = loss.item()

                scalar_accuracy = (torch.max(scores, 1)[1].view(labels.size(0)).data == labels.data).float().sum() / labels.size(0)
                scalar_accuracy = scalar_accuracy.item()

            avg_training_loss.add(scalar_loss)
            avg_training_accuracy.add(scalar_accuracy)

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

            if check_sanity or (last_log + log_every <= step_no):
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

        msglogger.info('==> Accuracy: %.3f      Loss: %.3f\n',
                   avg_training_accuracy.mean, avg_training_loss.mean)

        performance_summary = model_summary(model, dummy_input, 'performance')
        csv_log_writer.writerow({"Phase" : "Train", "Epoch" : epoch_idx, "Accuracy" : avg_training_accuracy.mean, "Loss" : avg_training_loss.mean, "Macs" : performance_summary["Total MACs"], "Weights" : performance_summary["Total Weights"], "LR" : optimizer.param_groups[0]['lr']})

        if check_sanity:
            if avg_training_accuracy.mean > 0.95:
                msglogger.info("Sanity check passed accuracy: {} loss: {}".format(avg_training_accuracy.mean, avg_training_loss.mean))
                return
        else:
            msglogger.info("Validation epoch {} of {}".format(epoch_idx, config["n_epochs"]))

            avg_acc, avg_loss, confusion_matrix = validate(dev_loader, model,None, criterion, config,None, None, loggers=loggers, epoch=epoch_idx)
            csv_log_writer.writerow({"Phase" : "Val", "Epoch" : epoch_idx, "Accuracy" : avg_acc, "Loss" : avg_loss, "Macs" : performance_summary["Total MACs"], "Weights" : performance_summary["Total Weights"], "LR" : optimizer.param_groups[0]['lr']})

            if avg_acc > max_acc:
                save_model(output_dir, model, test_set, config=config)
                max_acc = avg_acc

            # Stop training if the validation loss has not improved for multiple iterations
            # and early stopping is configured
            es = early_stopping(avg_loss)
            if(es and config["early_stopping"] > 0):
                break

        if lr_scheduler is not None:
            if type(lr_scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                lr_scheduler.step(avg_loss)

                # Reload best model at learning rate changes
                new_lr = optimizer.param_groups[0]['lr']
                if new_lr != last_lr:
                    last_lr = new_lr
                    model.load(os.path.join(output_dir, "model.pt"))

            else:
                lr_scheduler.step()

        if compression_scheduler is not None:
            compression_scheduler.on_epoch_begin(epoch_idx)




    if check_sanity:
        msglogger.info("Sanity check has not ended early accuracy: {} loss: {}".format(avg_training_accuracy.mean,
                                                                                       avg_training_loss.mean))

    else:
        msglogger.info("Running final test")
        model.load(os.path.join(output_dir, "model.pt"))

        test_accuracy, test_loss, confusion_matrix = evaluate(model_name, config,None, None, model, test_set)
        csv_log_writer.writerow({"Phase" : "Test", "Epoch" : epoch_idx, "Accuracy" : test_accuracy, "Loss" : test_loss, "Macs" : performance_summary["Total MACs"], "Weights" : performance_summary["Total Weights"], "LR" : optimizer.param_groups[0]['lr']})

        csv_eval_log_name = os.path.join(output_dir, "eval.csv")

        with open(csv_eval_log_name, 'a') as csv_eval_file:
            fcntl.lockf(csv_eval_file, fcntl.LOCK_EX)
            csv_eval_writer = csv.DictWriter(csv_eval_file, fieldnames=["Hash","Phase", "Epoch", "Accuracy", "Loss", "Macs", "Weights", "LR"])
            if os.stat(csv_eval_log_name).st_size == 0:
                csv_eval_writer.writeheader()
            csv_eval_writer.writerow({"Hash": config["config_hash"], "Phase" : "Test", "Epoch" : epoch_idx, "Accuracy" : test_accuracy, "Loss" : test_loss, "Macs" : performance_summary["Total MACs"], "Weights" : performance_summary["Total Weights"], "LR" : optimizer.param_groups[0]['lr']})
            fcntl.lockf(csv_eval_file, fcntl.LOCK_UN)

        return

def build_config(extra_config={}):
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "trained_models")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="ekut-raw-cnn3", type=str)
    parser.add_argument("--config", default="", type=str)
    parser.add_argument("--config_vad", default="", type=str)
    parser.add_argument("--config_keyword", default="", type=str)
    parser.add_argument("--dataset", choices=["keywords", "hotword", "vad", "keywords_and_noise"], default="keywords", type=str)
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

    default_config_vad = {}
    if config.config_vad:
            with open(config.config_vad, 'r') as f:
                default_config_vad = json.load(f)
                model_name = default_config_vad["model_name"]

                #Delete model from config for now to avoid showing
                #them as commandline otions
                if "model_name" in default_config_vad:
                    del default_config_vad["model_name"]
                else:
                    print("Your model config does not include a model_name")
                    print(" these configurations are not loadable")
                    sys.exit(-1)
                if "model_class" in default_config_vad:
                    del default_config_vad["model_class"]
                if "type" in default_config:
                    del default_config_vad["type"]


    default_config_keyword = {}
    if config.config_keyword:
            with open(config.config_keyword, 'r') as f:
                default_config_keyword = json.load(f)
                model_name = default_config_keyword["model_name"]

                #Delete model from config for now to avoid showing
                #them as commandline otions
                if "model_name" in default_config_keyword:
                    del default_config_keyword["model_name"]
                else:
                    print("Your model config does not include a model_name")
                    print(" these configurations are not loadable")
                    sys.exit(-1)
                if "model_class" in default_config_keyword:
                    del default_config_keyword["model_class"]
                if "type" in default_config_keyword:
                    del default_config_keyword["type"]


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
                                                         "adam",
                                                         "rmsprop"]),

                         opt_rho     = ConfigOption(category="Optimizer Config",
                                                    desc="Parameter rho for Adadelta optimizer",
                                                    default=0.9),
                         opt_eps     = ConfigOption(category="Optimizer Config",
                                                    desc="Paramter eps for Adadelta and Adam and SGD",
                                                    default=1e-06),
                         opt_alpha   = ConfigOption(category="Optimizer Config",
                                                    desc="Parameter alpha for RMSprop",
                                                    default=0.99),
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
                         early_stopping = ConfigOption(default=0,
                                                       desc="Stops the training if the validation loss has not improved for the last EARLY_STOPPING epochs"),

                         batch_size=ConfigOption(default=128,
                                                 desc="Default minibatch size for training"),
                         seed=ConfigOption(default=0,
                                           desc="Seed for Random number generators"),
                         input_file=ConfigOption(default="",
                                                 desc="Input model file for finetuning (.pth) or code generation (.onnx)"),
                         input_file_vad=ConfigOption(default="",
                                                 desc="Input vad model file for combined evaluation of vad and keyword spotting"),
                         input_file_keyword=ConfigOption(default="",
                                                 desc="Input keyword model file for combined evaluation of vad and keyword spotting"),
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

    parser = builder.build_argparse(parser)

    parser.add_argument("--type", choices=["train", "eval", "check_sanity", "eval_vad_keyword"], default="train", type=str)
    config = builder.config_from_argparse(parser)

    config["model_class"] = mod_cls
    default_config_vad["model_class"] = mod.find_model("small-vad") # als command line option umändern
    default_config_keyword["model_class"] = mod.find_model("honk-res15")
    config["model_name"] = model_name
    config["dataset"] = dataset_name
    config["dataset_cls"] = dataset_cls

    return (model_name, config, default_config_vad, default_config_keyword)

def main():
    model_name, config, config_vad, config_keyword = build_config()
    set_seed(config)
    # Set deterministic mode for CUDNN backend
    # Check if the performance penalty might be too high
    if config["cuda"]:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if config["type"] == "train":
        train(model_name, config)
    elif config["type"] == "check_sanity":
        train(model_name, config, check_sanity=True)
    elif config["type"] == "eval":
        accuracy, _ , _= evaluate(model_name, config)
        print("final accuracy is", accuracy)
    elif config["type"] == "eval_vad_keyword":
        accuracy, _, _ = evaluate(model_name, config, config_vad, config_keyword)
        print("final accuracy is", accuracy)

if __name__ == "__main__":
    main()
