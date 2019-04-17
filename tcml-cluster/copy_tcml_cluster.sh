#!/bin/bash

source .tcml_config

enable_datasets=0

while [[ $# -gt 0 ]]
do
    key=$1
    
    case $key in
	--datasets) # Do not exclude datasets from copy
	    enable_datasets=1
	    shift
	    ;;
	*)    # unknown option
	    echo "Found unknown option: $key"
	    exit 1
	    ;;
    esac
done

dataset_exclude="--exclude datasets"
if [ enable_datasets == 0 ]; then
    dataset_exclude=""
fi

echo "Copying files to $WD on tcml-cluster"

ssh ${USER}@tcml-master01.uni-tuebingen.de 'mkdir -p ~/speech_recognition'
rsync -arcv $dataset_exclude --exclude 'trained_models' --exclude '.git' --exclude 'orig' --exclude '.mypy_cache/'  . ${USER}@tcml-master01.uni-tuebingen.de:$WD



