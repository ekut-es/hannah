#!/bin/bash -e

#export SINGULARITY_TMPDIR=/local/data/tmp/singularity_tmp
#export SINGULARITY_CACHEDIR=/local/data/tmp/singularity_cache

#mkdir -p $SINGULARITY_TMPDIR
#mkdir -p $SINGULARITY_CACHEDIR

rm -rf ml_cloud.simg
sudo -E /usr/local/bin/singularity build --sandbox ml_cloud ml_cloud.recipe
singularity exec -w -f --bind $PWD/..:/opt/speech_recognition ml_cloud /bin/bash -c "cd /opt/speech_recognition && poetry install"
sudo singularity build ml_cloud.simg ml_cloud/
sudo rm -rf ml_cloud/
