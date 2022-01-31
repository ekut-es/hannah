#!/bin/sh

poetry export --without-hashes | grep -v hannah-optimizer > requirements.txt
singularity build  ml_cloud.sif  ml_cloud.recipe
