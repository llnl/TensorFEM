// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "../common.hpp"

/*
  This example demonstrates how to create 2d functions and communicate between boba and mfem
  through the TensorFEM interface
*/

struct parameters
{
  size_t refinement = 1;
  size_t order = 1;
  size_t visualize = 0;
};

static constexpr ::boba::execution_space space = ::boba::default_execution_space;

int main(int argc, char *argv[])
{
  bool check = true;
  checkpoint();
  boba::init();
  size_t dimension = 2;
  parameters input;
  boba::argparser args(argc, argv);
  args.add_optional_argument(input.refinement, "-n", "--refinement", "Element refinement factor in each dimension.");
  args.add_optional_argument(input.order, "-o", "--order", "Finite element order of accuracy");
  args.add_optional_argument(input.visualize, "-v", "--visualize", "Flag to turn on/off visualization");
  args.parse_check();
  auto refinement_factor = boba::pow(2, input.refinement);
  size_t number_points_1d = 3 * refinement_factor;
  auto feOrder = input.order;
  auto fesize = feOrder + 1;
  size_t N_EI = number_points_1d;
  size_t N_EJ = number_points_1d;
  auto mesh_2d = ::tensor_fem::make_cartesian_mesh_2d(N_EI, N_EJ, 1.0, 1.0);
  mfem::DG_FECollection fec_2d(input.order, dimension, mfem::BasisType::GaussLobatto);
  mfem::FiniteElementSpace pfes_2d(&mesh_2d, &fec_2d);
  auto blob = [=](double x, double y)
  {
    double x_center = 0.5;
    double blob_radius = 0.3;
    double xy2 = boba::pow(x - x_center, 2.0) + boba::pow(y - x_center, 2.0);
    double r2 = ::boba::pow(blob_radius, 2.0);
    double f = ( xy2 < r2 ) ? boba::exp(1./r2 + 1./(xy2 - r2)) : 0.0;
    return f;
  };
  auto gf = tensor_fem::make_gf_2d(pfes_2d, blob);
  if(input.visualize == 1)
    tensor_fem::glvis_visualize(mesh_2d, gf, "2D function demonstration");
  else
    boba_print("Visualization is off");
  auto vector_of_dofs = tensor_fem::gf_to_boba_vector_2d(mesh_2d, pfes_2d, gf);
  auto vector_of_dofs_cart = tensor_fem::gf_to_boba_vector_2d(mesh_2d, pfes_2d, gf, input.order);
  {
    boba::PermutationMatrix<space, size_t> index_reordering({vector_of_dofs_cart.size()});
    auto index_reordering_view = index_reordering.view();
    boba::Multiindexer<4> boba_ordering({fesize, N_EI, fesize, N_EJ});
    boba::Multiindexer<4> mfem_ordering({fesize, fesize, N_EI, N_EJ});
    for(size_t k = 0; k < index_reordering_view.size(); k++)
    {
      auto [li, lj, ei, ej] = mfem_ordering.multiindex(k);
      index_reordering_view(k) = boba_ordering.index({li, ei, lj, ej});
    }
    auto difference = boba::norm_frobenius(vector_of_dofs_cart - index_reordering*vector_of_dofs);
    pass_or_fail(check, difference, 1.0e-10);
  }
  boba::finalize();
  return final_check(check);
}
