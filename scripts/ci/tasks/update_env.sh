#!/bin/bash
##
## Copyright (c) 2022 University of Tübingen.
##
## This file is part of hannah.
## See https://atreus.informatik.uni-tuebingen.de/ties/ai/hannah/hannah for further info.
##
## Licensed under the Apache License, Version 2.0 (the "License");
## you may not use this file except in compliance with the License.
## You may obtain a copy of the License at
##
##     http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
## See the License for the specific language governing permissions and
## limitations under the License.
##


#SBATCH --job-name=update_env

#resources:

#SBATCH --partition=cpu-short
# the slurm partition the job is queued to.
# FIXME: test if preemptable is avallable

#SBATCH --nodes=1
# requests that the cores are all on one node


#SBATCH --time=30
# the maximum time the scripts needs to run (720 minutes = 12 hours)

#SBATCH --error=jobs/%j.err
# write the error output to job.*jobID*.err

#SBATCH --output=jobs/%j.out
# write the standard output to your home directory job.*jobID*.out

#SBATCH --mail-type=ALL
#write a mail if a job begins, ends, fails, gets requeued or stages out

#SBATCH --mail-user=christoph.gerum@uni-tuebingen.de
# your mail address

poetry install
