#!/bin/bash

pushd glvis

make clean
make MFEM_DIR=../mfem SDL_DIR=../SDL/SDL GLM_DIR=../glm CXXFLAGS="-std=c++17 -O3"

popd
