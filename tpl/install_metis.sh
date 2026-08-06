#!/bin/bash

pushd mfem_tpls

rm -rf metis-5.1.0
tar -zvxf metis-5.1.0.tar.gz

cp ../metis_helper/Makefile ./metis-5.1.0/Makefile

cd metis-5.1.0

make BUILDDIR=lib config
make BUILDDIR=lib
cp lib/libmetis/libmetis.a lib

popd
