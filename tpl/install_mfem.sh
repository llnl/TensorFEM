#!/bin/bash

pushd mfem

make clean
make serial -j 20 MFEM_USE_METIS_5=YES METIS_DIR=@MFEM_DIR@/../mfem_tpls/metis-5.1.0 CXXFLAGS="-std=c++17 -O3"

#make config MFEM_USE_MPI=YES MFEM_DEBUG=YES MPICXX=mpic++
#make install -j

popd
