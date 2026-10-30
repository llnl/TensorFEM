// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file tensor_fem.cpp
 * @brief TensorFEM runtime initialization wrappers.
 */

#include "boba.hpp"

namespace tensor_fem
{

/**
 * @brief Initialize TensorFEM
 */
void init()
{
    boba::init();
}

/**
 * @brief Finalize TensorFEM.
 */
void finalize()
{
    boba::finalize();
}

}
