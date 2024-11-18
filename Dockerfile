##
## Copyright (c) 2024 Hannah contributors.
##
## This file is part of hannah.
## See https://github.com/ekut-es/hannah for further info.
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

# Target for python/mlonmcu/python
ARG target="hannah"

# ARG for Python version, defaulting to 3.12 only used if python target is selected
ARG python_version=3.12

FROM ubuntu:22.04 as hannah
FROM tumeda/mlonmcu-bench:latest as mlonmcu
FROM python:${python_version} as python

FROM ${target}

# These need to be set here again, as the FROM directive resets the ARGs on use
ARG python_version
ARG target

ENV POETRY_CACHE_DIR="/tmp/poetry_cache"


RUN  if [ "$target" = "hannah" ] || [ "$target" = "mlonmcu" ]; then\
        apt-get update -y && apt-get -y install tree git mesa-utils python3 python3-pip python3-dev libblas-dev liblapack-dev libsndfile1-dev libsox-dev cmake ninja-build curl build-essential python-is-python3; \
      else \
        apt-get update -y && apt-get -y install tree git mesa-utils  libblas-dev liblapack-dev libsndfile1-dev libsox-dev cmake ninja-build curl build-essential;  \
      fi


# Install poetry using recommended method
RUN  pip install poetry


# Copy only requirements to cache them in docker layer
COPY poetry.lock pyproject.toml /deps/
COPY external/ /deps/external/

WORKDIR /deps


RUN tree -L 3 -d

# Install dependencies
RUN poetry install --no-interaction --no-ansi --all-extras --no-root \
  && rm -rf $POETRY_CACHE_DIR
