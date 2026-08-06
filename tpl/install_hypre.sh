#!/bin/bash

pushd ./hypre/src

./configure --disable-fortran
make install -j 10

popd
