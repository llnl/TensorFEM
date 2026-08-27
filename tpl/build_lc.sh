#!/bin/bash

#
# This is a script to help get GLVIS working on LC systems, such as Dane
# This script manually clones tarballs for GLEW and SDL2,
# which are not part of the normal setup instructions for other systems
# due to them being controlled by distribution managers such as Brew
#

# From https://github.com/GLVis/glvis/blob/master/INSTALL

git submodule update --init

./install_glew_manual.sh
./install_sdl2.sh

# Need metis for mfem
./install_metis.sh
./install_mfem.sh
./install_glvis.sh
cd BoBa
./boba_builder.py
