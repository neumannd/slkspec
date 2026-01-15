#!/bin/bash

## DEPENDENCIES:
# conda install conda-build grayskull uv pip

VERSION=0.0.5

## BUILD
uv build .
chmod +r dist

# grayskull pypi dist/slkspec-${VERSION}.tar.gz --maintainers HaukeSchulzDMI
# mv slkspec/meta.yaml .
# sed -i '/PLEASE_ADD_LICENSE_FILE/d' meta.yaml

conda-build . -c local
mkdir -p conda
mv /opt/python/miniforge3/envs/slkspec/conda-bld/noarch/slkspec*.conda dist/

# {% set name = "slkspec" %}
# {% set version = "0.0.5" %}
# {% set PYTHON = "python" %}
#
# package:
#   name: {{ name|lower }}
#   version: {{ version }}
#
# source:
#   url: file:///home/k204221/Documents/55_StrongLink/37_Python/slkspec/dist/slkspec-0.0.5.tar.gz
#   sha256: b1f90afa61e71298e8aa96a023037a34900749de410766e2577314b40e53fde2
#
# build:
#   noarch: python
#   script: {{ PYTHON }} -m pip install . -vv --no-deps --no-build-isolation
#   number: 0
#
# about:
#   summary: fsspec implementation for StrongLink tape archive
#   license: MIT
#
# extra:
#   recipe-maintainers:
#     - HaukeSchulzDMI
