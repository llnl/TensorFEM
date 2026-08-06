#!/bin/bash

set -e

git submodule update --init
./install_glm.sh
./install_sdl2.sh
./install_mfem.sh
./install_glvis.sh
./install_metis.sh
cd BoBa
./boba_builder.py
