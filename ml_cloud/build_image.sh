#!/bin/bash

fakeroot=""
if [[ $EUID -ne 0 ]]; then
   fakeroot="--fakeroot"
fi

poetry export --without-hashes | grep -v hannah-optimizer > requirements.txt
singularity build  $fakeroot  ml_cloud.sif  ml_cloud.recipe
