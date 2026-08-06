#!/bin/bash

git submodule update --init

./install_glm.sh
./install_sdl2.sh
./install_metis.sh
./install_mfem.sh
./install_glvis.sh
cd BoBa
./boba_builder.py
