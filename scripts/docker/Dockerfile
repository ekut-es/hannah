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

FROM ubuntu:22.04
# COPY . /app
# RUN make /app

RUN  apt update
RUN  apt -y install git mesa-utils python3-dev libblas-dev liblapack-dev libsndfile1-dev libsox-dev cmake ninja-build curl build-essential python-is-python3


  # Install poetry using recommended method
RUN  curl -sSL https://install.python-poetry.org/ | python3
RUN  export PATH=${POETRY_HOME}/bin:${PATH}
