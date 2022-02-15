#!/bin/bash

#Starts an interactive bash on a node with two rtx2080ti gpus

srun --job-name "InteractiveJob" --ntasks=1 --nodes=1 --time 12:00:00 --gres=gpu:rtx2080ti:2 --pty bash
