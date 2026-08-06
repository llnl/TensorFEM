#!/bin/bash

# GLEW
wget "https://github.com/nigels-com/glew/releases/download/glew-2.1.0/glew-2.1.0.tgz"
tar zxf glew-2.1.0.tgz
ln -s glew-2.1.0 glew

pushd glew

make -j 4

popd
