#!/bin/bash

#export SINGULARITY_TMPDIR=/local/data/tmp/singularity_tmp
#export SINGULARITY_CACHEDIR=/local/data/tmp/singularity_cache

#mkdir -p $SINGULARITY_TMPDIR
#mkdir -p $SINGULARITY_CACHEDIR

rm -rf ml_cloud.simg
sudo -E /usr/local/bin/singularity build ml_cloud.simg ml_cloud.recipe
