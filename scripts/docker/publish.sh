#!/bin/bash
##
## Copyright (c) 2022 Hannah contributors.
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
export TAG=0.1.0

docker build -t "cgerum/hannah:{$TAG}" .
docker push "cgerum/hannah:${TAG}"

docker tag "cgerum/hannah:${TAG}" cgerum/hannah:latest
docker push "cgerum/hannah:latest"
