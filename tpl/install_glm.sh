#!/bin/bash

pushd glm

cmake \
    -DCXX_STANDARD=17 \
    -DGLM_BUILD_TESTS=OFF \
    -DBUILD_SHARED_LIBS=OFF \
    -DCMAKE_INSTALL_PREFIX=$(pwd)/install \
    -B build .
cmake --build build -- all
cmake --build build -- install

popd
