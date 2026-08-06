
#include "../common.hpp"

static constexpr ::boba::execution_space space = ::boba::default_execution_space;

/**
 * \brief This example demonstrates "tensorized assembly".
 * Here we take the framework laid out in http://mfem.org/performance/,
 * and show how to meld this with tensor methods.
 *
 * Compile and run this example (from the top directory of this repo)
 * make example_assembly && ./example_assembly_cpu.out
 */

int main(int argc, char* argv[])
{
  bool check = true;
  checkpoint();
  boba::init();

  size_t spatial_resolution = 3;
  size_t order = 2;
  size_t refinements = 0;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(
    refinements,
    "-R",
    "--refinements",
    "Number of mesh refinements.");

  args.add_optional_argument(
    order,
    "M",
    "--order",
    "FEM order.");

  args.parse_check();

  //
  // Example in 1D
  // Demonstrates a simple case along a single dimension
  //
  {
    //
    // Generate a mesh and finite element space
    //
    mfem::Mesh mesh = tensor_fem::make_mesh_1d(spatial_resolution, 1.0);

    for (size_t refinement = 0; refinement < refinements; refinement++)
    {
      mesh.UniformRefinement();
    }

    int dim = mesh.Dimension();

    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLegendre);
    mfem::FiniteElementSpace fes(&mesh, &fec);

    //
    // Make the B and B^T matrices (full assembly)
    //
    auto B_matrix = tensor_fem::make_B_matrix_1d(fes);
    auto BT_matrix = B_matrix.transpose();

    //
    // Make the integration weights
    // These are usually wrapped into B^T, but we keep them separate here
    //
    auto W_vector = tensor_fem::make_W_vector_1d(fes);

    boba_print(B_matrix.sizes());
    boba_print(W_vector.sizes());
    boba_print(BT_matrix.sizes());

    //
    // Run some tests that show different ways of using this
    //
    boba::Vector<space, double> test_vector({B_matrix.cols()});
    test_vector.fill_with_random();

    checkpoint();
    boba::Vector<space, double> result_vector_PA;
    {
      auto Bv = B_matrix * test_vector;
      auto WBv = boba::elementwise_product(W_vector, Bv);
      result_vector_PA = BT_matrix * WBv;
    }
    checkpoint();
    boba::Vector<space, double> result_vector_full;
    {
      auto WB = B_matrix;
      boba::apply_as_diagonal_left_in_place(W_vector, WB);
      auto BtWB = BT_matrix * WB;
      result_vector_full = BtWB * test_vector;
    }
    checkpoint();
    boba::Vector<space, double> result_vector_mass;
    {
      auto mass_matrix = tensor_fem::Mass_matrix_1d(fes, [](double x)
      {
        return 1.0;
      });
      result_vector_mass = mass_matrix * test_vector;
    }
    checkpoint();

    // Verification

    auto diff_full_vs_PA = boba::norm_difference_inf(result_vector_full, result_vector_PA);
    auto diff_full_vs_mass = boba::norm_difference_inf(result_vector_full, result_vector_mass);
    auto diff_PA_vs_mass = boba::norm_difference_inf(result_vector_PA, result_vector_mass);

    auto tolerance = 1.0e-12;
    pass_or_fail(check, diff_full_vs_PA, tolerance);
    pass_or_fail(check, diff_full_vs_mass, tolerance);
    pass_or_fail(check, diff_PA_vs_mass, tolerance);
  }

  //
  // Example in 2D
  // Demonstrates the above plus how to use this with tensor trains and tt-matrices
  //
  {
    //
    // Generate a mesh and finite element space
    //
    mfem::Mesh mesh = tensor_fem::make_cartesian_mesh_2d(spatial_resolution, spatial_resolution, 1.0, 1.0);

    int mesh_order = 1;
    mesh.SetCurvature(mesh_order);

    for (size_t refinement = 0; refinement < refinements; refinement++)
    {
      mesh.UniformRefinement();
    }

    int dim = mesh.Dimension();
    mfem::DG_FECollection fec(order, dim, mfem::BasisType::GaussLegendre);
    mfem::FiniteElementSpace fes(&mesh, &fec);

    //
    // Make the B and B^T matrices (full assembly)
    //
    auto B_matrix = tensor_fem::make_B_matrix_2d(fes);
    auto BT_matrix = B_matrix.transpose();

    //
    // Make the integration weights
    // These are usually wrapped into B^T, but we keep them separate here
    //
    auto W_vector = tensor_fem::make_W_vector_2d(fes);

    boba_print(B_matrix.sizes());
    boba_print(W_vector.sizes());
    boba_print(BT_matrix.sizes());

    auto rows_1d = boba::sqrt(B_matrix.rows());
    auto cols_1d = boba::sqrt(B_matrix.cols());

    auto svd_tolerance_relative = 1.0e-08;
    auto svd_tolerance_absolute = 1.0e-08;

    checkpoint();
    auto B_ttm = boba::compress_to_TensorTrainMatrix<2>(B_matrix, {rows_1d, rows_1d}, {cols_1d, cols_1d}, svd_tolerance_relative, svd_tolerance_absolute);
    checkpoint();
    auto BT_ttm = B_ttm.transpose();
    checkpoint();
    auto W_tt = boba::compress_to_TensorTrain<2>(boba::reshape<2>(W_vector, {rows_1d, rows_1d}), svd_tolerance_relative, svd_tolerance_absolute);
    checkpoint();

    //
    // Run some tests that show different ways of using this
    //
    boba::Vector<space, double> test_vector({B_matrix.cols()});
    test_vector.fill_with_random();

    checkpoint();
    auto test_tt = boba::compress_to_TensorTrain<2>(boba::reshape<2>(test_vector, {cols_1d, cols_1d}), svd_tolerance_relative, svd_tolerance_absolute);

    checkpoint();
    boba::Vector<space, double> result_vector_PA;
    {
      auto Bv = B_matrix * test_vector;
      auto WBv = boba::elementwise_product(W_vector, Bv);
      result_vector_PA = BT_matrix * WBv;
    }
    checkpoint();
    boba::Vector<space, double> result_vector_full;
    {
      auto WB = B_matrix;
      boba::apply_as_diagonal_left_in_place(W_vector, WB);
      auto BtWB = BT_matrix * WB;
      result_vector_full = BtWB * test_vector;
    }
    checkpoint();
    boba::Vector<space, double> result_vector_mass;
    {
      auto mass_matrix = tensor_fem::Mass_matrix_2d(fes, [](double x, double y)
      {
        return 1.0;
      },
                                                    order,
                                                    spatial_resolution,
                                                    spatial_resolution);
      result_vector_mass = mass_matrix * test_vector;
    }
    checkpoint();
    boba::Vector<space, double> result_vector_tt;
    {
      auto Bv = B_ttm * test_tt;
      auto WBv = boba::elementwise_product(W_tt, Bv);
      auto result_tt = BT_ttm * WBv;
      result_vector_tt = boba::flatten(result_tt.decompress());
    }
    checkpoint();

    // Verification
    // We demonstrate that the results of using the tensorized PA provides the
    // same results as using a mass matrix

    auto diff_full_vs_PA = boba::norm_difference_inf(result_vector_full, result_vector_PA);
    auto diff_full_vs_mass = boba::norm_difference_inf(result_vector_full, result_vector_mass);
    auto diff_full_vs_tt = boba::norm_difference_inf(result_vector_full, result_vector_tt);
    auto diff_PA_vs_mass = boba::norm_difference_inf(result_vector_PA, result_vector_mass);
    auto diff_PA_vs_tt = boba::norm_difference_inf(result_vector_PA, result_vector_tt);
    auto diff_mass_vs_tt = boba::norm_difference_inf(result_vector_mass, result_vector_tt);

    auto tolerance = 1.0e-12;
    pass_or_fail(check, diff_full_vs_PA, tolerance);
    pass_or_fail(check, diff_full_vs_mass, tolerance);
    pass_or_fail(check, diff_full_vs_tt, tolerance);
    pass_or_fail(check, diff_PA_vs_mass, tolerance);
    pass_or_fail(check, diff_PA_vs_tt, tolerance);
    pass_or_fail(check, diff_mass_vs_tt, tolerance);
  }

  return final_check(check);
}