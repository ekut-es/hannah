#!/bin/bash
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


# Define the Python versions to build
VERSIONS=(3.9 3.10 3.11 3.12)
# VERSIONS=(3.12)

HANNAH_FOLDER=$(pwd)/hannah

# Set the registry and namespace
REGISTRY="es-git-registry.cs.uni-tuebingen.de"
NAMESPACE="es/ai/hannah"
PROJECT="hannah"
IMAGE_NAME="testing/python"


get_tag() {
    local VERSION="$1"
    echo "${REGISTRY}/${NAMESPACE}/${PROJECT}/${IMAGE_NAME}:${VERSION}"
}

build_image() {
    local VERSION="$1"
    local TAG=$(get_tag "$VERSION")
    docker build -f Dockerfile.python3 --build-arg PYTHON_VERSION="${VERSION}" -t "${TAG}" .
}
# Function to push an image
push_image() {
    local VERSION="$1"
    local TAG=$(get_tag "$VERSION")
    echo "Pushing image ${TAG}..."
    docker push "${TAG}"
}

# Function to test an image
test_image() {
    local VERSION="$1"
    local VERSION_WITHOUT_DOT=${VERSION/.}
    cd $HANNAH_FOLDER
    # --docker-pull-policy "never"
    gitlab-runner exec docker --cicd-config-file ".gitlab-ci.yml" --clone-url "." "test_python_${VERSION_WITHOUT_DOT}"
}

# Function to echo an image tag
echo_image() {
    local VERSION="$1"
    local TAG=$(get_tag "$VERSION")
    echo "${TAG}"
}

# Check if a second argument is provided to override VERSIONS
if [ -n "$2" ]; then
    VERSIONS=("$2")
fi


# Default action is build
action=${1:-build}

case "$action" in
  push)
    for VERSION in "${VERSIONS[@]}"; do
      push_image "$VERSION"
    done
    ;;

  build)
    for VERSION in "${VERSIONS[@]}"; do
      build_image "$VERSION"
    done
    ;;

  test)
    for VERSION in "${VERSIONS[@]}"; do
      test_image "$VERSION"
    done
    ;;

  echo)
    for VERSION in "${VERSIONS[@]}"; do
      echo_image "$VERSION"
    done
    ;;

  all)
    for VERSION in "${VERSIONS[@]}"; do
      build_image "$VERSION"
      test_image "$VERSION"
      push_image "$VERSION"
    done
    ;;

  *)
    echo "Unknown action: $action. Options are push build test"
    ;;
esac
