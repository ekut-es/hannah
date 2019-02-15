from collections import ChainMap
from .config import ConfigBuilder
import argparse
import os
import random
import sys
import json

from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as data

from tensorboardX import SummaryWriter

from . import models as mod
from . import dataset



def get_eval(scores, labels, loss):
    batch_size = labels.size(0)
    accuracy = (torch.max(scores, 1)[1].view(batch_size).data == labels.data).float().sum() / batch_size
    loss = loss.item()

    return accuracy.item(), loss

def print_eval(name, scores, labels, loss, end="\n"):
    accuracy, loss = get_eval(scores, labels, loss) 
    print("{} accuracy: {:>5}, loss: {:<25}".format(name, accuracy, loss), end=end)
    return accuracy


def set_seed(config):
    seed = config["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not config["no_cuda"]:
        torch.cuda.manual_seed(seed)
    random.seed(seed)

def evaluate(model_name, config, model=None, test_loader=None, logfile=None):
    print("Evaluating network")
    
    if not test_loader:
        _, _, test_set = dataset.SpeechDataset.splits(config)
        test_loader = data.DataLoader(test_set, batch_size=1)
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
    if not model:
        model = config["model_class"](config)
        model.load(config["input_file"])
    if not config["no_cuda"]:
        torch.cuda.set_device(config["gpu_no"])
        model.cuda()
    model.eval()
    criterion = nn.CrossEntropyLoss()
    results = []
    total = 0
    for model_in, labels in test_loader:
        model_in = Variable(model_in, requires_grad=False)
        if not config["no_cuda"]:
            model_in = model_in.cuda()
            labels = labels.cuda()
        if total == 0:
            scores = model(model_in, export=True)
        else:
            scores = model(model_in, export=False)
        labels = Variable(labels, requires_grad=False)
        loss = criterion(scores, labels)
        results.append(print_eval("test", scores, labels, loss) * model_in.size(0))
        total += model_in.size(0)
        
    print("final test accuracy: {}".format(sum(results) / total))
    return sum(results) / total

def train(model_name, config):
    output_dir = os.path.join(config["output_dir"], model_name)
    print("All information will be saved to: ", output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    train_log = open(os.path.join(output_dir, "train.csv"), "w")
    with SummaryWriter() as summary_writer:
        train_set, dev_set, test_set = dataset.SpeechDataset.splits(config)

        config["width"] = train_set.width
        config["height"] = train_set.height

        with open(os.path.join(output_dir, 'config.json'), "w") as o:
                  s = json.dumps(dict(config), default=lambda x: str(x), indent=4, sort_keys=True)
                  o.write(s)
        
        model = config["model_class"](config)
        if config["input_file"]:
            model.load(config["input_file"])
        if not config["no_cuda"]:
            torch.cuda.set_device(config["gpu_no"])
            model.cuda()
        optimizer = torch.optim.SGD(model.parameters(), 
                                    lr=config["lr"][0], 
                                    nesterov=config["use_nesterov"],
                                    weight_decay=config["weight_decay"], 
                                    momentum=config["momentum"])
        schedule_steps = config["schedule"]
        schedule_steps.append(np.inf)
        sched_idx = 0
        criterion = nn.CrossEntropyLoss()
        max_acc = 0
         
        train_loader = data.DataLoader(train_set, batch_size=config["batch_size"], shuffle=True, drop_last=True)
        dev_loader = data.DataLoader(dev_set, batch_size=min(len(dev_set), 16), shuffle=True)
        test_loader = data.DataLoader(test_set, batch_size=1, shuffle=True)
        step_no = 0

        dummy_input, dummy_label = next(iter(test_loader))
        if not config["no_cuda"]:
            dummy_input = dummy_input.cuda()
       
        model.eval()
                    
        summary_writer.add_graph(model, dummy_input)
        
        for epoch_idx in range(config["n_epochs"]):
            print("Training epoch", epoch_idx, "of", config["n_epochs"])
            for batch_idx, (model_in, labels) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                if not config["no_cuda"]:
                    model_in = model_in.cuda()
                    labels = labels.cuda()
                model_in = Variable(model_in, requires_grad=False)
                scores = model(model_in)
                labels = Variable(labels, requires_grad=False)
                loss = criterion(scores, labels)

                loss.backward()
                optimizer.step()

                step_no += 1

                scalar_accuracy, scalar_loss = get_eval(scores, labels, loss)

                summary_writer.add_scalars('training', {'accuracy': scalar_accuracy,
                                                        'loss': scalar_loss},
                                           step_no)
                train_log.write("train,"+str(step_no)+","+str(scalar_accuracy)+","+str(scalar_loss)+"\n")
                
                if step_no > schedule_steps[sched_idx]:
                    sched_idx += 1
                    print("changing learning rate to {}".format(config["lr"][sched_idx]))
                    optimizer = torch.optim.SGD(model.parameters(), lr=config["lr"][sched_idx],
                        nesterov=config["use_nesterov"], momentum=config["momentum"], weight_decay=config["weight_decay"])
         
                if step_no % 100 == 0:
                    print_eval("train step #{}".format(step_no), scores, labels, loss)
                        
         
            if epoch_idx % config["dev_every"] == config["dev_every"] - 1:
                model.eval()
                accs = []
                for model_in, labels in dev_loader:
                    model_in = Variable(model_in, requires_grad=False)
                    if not config["no_cuda"]:
                        model_in = model_in.cuda()
                        labels = labels.cuda()
                    scores = model(model_in)
                    labels = Variable(labels, requires_grad=False)
                    loss = criterion(scores, labels)
                    loss_numeric = loss.item()
                    accs.append(print_eval("dev", scores, labels, loss))
                avg_acc = np.mean(accs)
                print("final dev accuracy: {}".format(avg_acc))
                train_log.write("val,"+str(step_no)+","+str(avg_acc)+"\n")
                
                summary_writer.add_scalars('validation', {'accuracy': avg_acc},
                                           step_no)
                
                if avg_acc > max_acc:
                    print("saving best model...")
                    max_acc = avg_acc
                    model.save(os.path.join(output_dir, "model.pt"))
                    print("saving onnx...")
                    print(dir(dev_loader))
                    model_in, label = next(iter(test_loader), (None, None))
                    if not config["no_cuda"]:
                        model_in = model_in.cuda()
                    model.save_onnx(os.path.join(output_dir, "model.onnx"), model_in)
         
        model.load(os.path.join(output_dir, "model.pt"))
        test_accuracy = evaluate(model_name, config, model, test_loader)
        train_log.write("test,"+str(step_no)+","+str(test_accuracy)+"\n")
        train_log.close()
        
def main():
    output_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", "trained_models")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=[x.value for x in list(mod.ConfigType)], default="honk-cnn-trad-pool2", type=str)
    config, _ = parser.parse_known_args()

    model_name = config.model
    
    global_config = dict(no_cuda=False, n_epochs=500, lr=[0.001], schedule=[np.inf], batch_size=64, dev_every=10, seed=0,
        use_nesterov=False, input_file="", output_dir=output_dir, gpu_no=0, cache_size=32768, momentum=0.9, weight_decay=0.00001)
    mod_cls = mod.find_model(config.model)
    builder = ConfigBuilder(
        mod.find_config(config.model),
        dataset.SpeechDataset.default_config(),
        global_config)
    parser = builder.build_argparse()
    parser.add_argument("--type", choices=["train", "eval"], default="train", type=str)
    config = builder.config_from_argparse(parser)
    config["model_class"] = mod_cls
    set_seed(config)
    if config["type"] == "train":
        train(model_name, config)
    elif config["type"] == "eval":
        evaluate(model_name, config)

if __name__ == "__main__":
    main()
