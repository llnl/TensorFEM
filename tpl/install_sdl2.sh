#!/bin/bash

cd SDL

mkdir -p build
cd build
../configure --prefix $(pwd)/../SDL
make
make install
