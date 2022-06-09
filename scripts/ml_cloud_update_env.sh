#!/bin/bash

conda activate hannah

poetry export -E vision --without-hashes > requirements.txt

pip install -r requirements.txt
