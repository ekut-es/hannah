#!/bin/sh

poetry export --without-hashes | grep -v hannah-optimizer > requirements.txt
singularity build -f  ml_cloud.sif  ml_cloud.recipe
