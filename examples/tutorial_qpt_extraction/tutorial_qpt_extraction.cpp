#include "../common.hpp"

/*
  This example demonstrates how to extract physical quadrature points
  from one- and two-dimensional MFEM meshes using helper functions
  available in TensorFEM.

  The correctness checks use Cartesian meshes, where the element
  transformations are known analytically.
*/

static constexpr ::boba::execution_space space = ::boba::default_execution_space;

int main(int argc, char* argv[])
{
  bool check = true;

  checkpoint();
  boba::init();

  size_t refinement = 1_z;
  size_t order = 1_z;

  boba::argparser args(argc, argv);

  args.add_optional_argument(
    refinement,
    "-n",
    "--refinement",
    "Number of element refinements to perform in each dimension.");

  args.add_optional_argument(
    order,
    "-o",
    "--order",
    "Finite element order of accuracy.");

  args.parse_check();

  const size_t elements_per_dimension = boba::pow(2, refinement);

  checkpoint();
  {
    boba_print("Checking the one-dimensional Cartesian mesh");

    // Make a line segment of length L with Ne elements.
    const size_t dimension = 1_z;
    const double L = 2.0;
    const size_t Ne = elements_per_dimension;

    auto mesh = tensor_fem::make_mesh_1d(Ne, L);

    // Setup the finite element space.
    // We demonstrate using DG finite elements, but others may be used.
    mfem::DG_FECollection fec(
      static_cast<int>(order),
      static_cast<int>(dimension),
      mfem::BasisType::GaussLobatto);

    mfem::FiniteElementSpace fes(&mesh, &fec);

    // Call the helper to extract the quadrature points from the mesh.
    // Since the mesh is 1-D, the quadrature points are stored in a
    // boba::Vector of size Nq * Ne.
    auto x_quad = tensor_fem::quad_points_to_boba_vector_1d(mesh, fes, order);

    // Reconstruct the same integration rule used by the helper.
    const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(0);
    const int ir_order = 2 * static_cast<int>(order) + 2;
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, ir_order);
    const size_t Nq = static_cast<size_t>(ir.GetNPoints());

    // Simple check: does the helper return data of the correct size?
    pass_or_fail_bool(check, x_quad.size() == Ne * Nq);

    // For a Cartesian mesh, the element transformation is
    //
    //   x = e*h + h*x_ref,
    //
    // where x_ref is the reference coordinate in [0, 1].
    const double h = L / static_cast<double>(Ne);
    const double tol = 1e-14;

    const auto x_quad_view = x_quad.const_view();

    for (size_t e = 0_z; e < Ne; ++e)
    {
      for (size_t q = 0_z; q < Nq; ++q)
      {
        const size_t idx = q + Nq * e;

        const double x_ref = ir.IntPoint(static_cast<int>(q)).x;
        const double x_true = static_cast<double>(e) * h + h * x_ref;

        pass_or_fail(check, boba::abs(x_quad_view({idx}) - x_true), tol);
      }
    }
  }

  checkpoint();
  {
    boba_print("Checking the two-dimensional Cartesian mesh");

    // Make a rectangle [0, Lx] x [0, Ly] with Ne * Ne elements.
    const size_t dimension = 2_z;
    const double Lx = 2.0;
    const double Ly = 1.0;
    const size_t Ne = elements_per_dimension;

    auto mesh = tensor_fem::make_cartesian_mesh_2d(Ne, Ne, Lx, Ly);

    // Setup the finite element space.
    // We demonstrate using DG finite elements, but others may be used.
    mfem::DG_FECollection fec(
      static_cast<int>(order),
      static_cast<int>(dimension),
      mfem::BasisType::GaussLobatto);

    mfem::FiniteElementSpace fes(&mesh, &fec);

    // Call the helper to extract the quadrature points from the mesh.
    // Since the mesh is 2-D, the quadrature points are stored in a pair
    // of boba::Matrix objects with sizes (Nq * Ne) x (Nq * Ne).
    auto [x_quad, y_quad] =
      tensor_fem::quad_points_to_boba_matrices_2d(mesh, fes, order);

    // Reconstruct the same integration rule used by the helper.
    const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(0);
    const int ir_order = 2 * static_cast<int>(order) + 2;
    const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, ir_order);

    const size_t Nq_total = static_cast<size_t>(ir.GetNPoints());
    const size_t Nq = static_cast<size_t>(boba::sqrt(Nq_total));

    boba_always_assert_equal(
      Nq * Nq,
      Nq_total,
      "Quadrature point count has non-square factorization.");

    // Simple check: do the helpers return data with the expected sizes?
    const boba::Array<size_t, 2> correct_sizes{Nq * Ne, Nq * Ne};

    pass_or_fail_bool(check, x_quad.sizes() == correct_sizes);
    pass_or_fail_bool(check, y_quad.sizes() == correct_sizes);

    // For a Cartesian mesh, the element transformation is
    //
    //   x = ei*hx + hx*x_ref,
    //   y = ej*hy + hy*y_ref,
    //
    // where (x_ref, y_ref) is the reference quadrature point.
    const double hx = Lx / static_cast<double>(Ne);
    const double hy = Ly / static_cast<double>(Ne);
    const double tol = 1e-14;

    boba::Multiindexer<2> quadrature_mider({Nq, Nq});

    const auto x_quad_view = x_quad.const_view();
    const auto y_quad_view = y_quad.const_view();

    for (size_t ej = 0_z; ej < Ne; ++ej)
    {
      for (size_t ei = 0_z; ei < Ne; ++ei)
      {
        for (size_t qj = 0_z; qj < Nq; ++qj)
        {
          for (size_t qi = 0_z; qi < Nq; ++qi)
          {
            const size_t row = qi + Nq * ei;
            const size_t col = qj + Nq * ej;

            const size_t q = quadrature_mider.index({qi, qj});

            const double x_ref = ir.IntPoint(static_cast<int>(q)).x;
            const double y_ref = ir.IntPoint(static_cast<int>(q)).y;

            const double x_true =
              static_cast<double>(ei) * hx + hx * x_ref;

            const double y_true =
              static_cast<double>(ej) * hy + hy * y_ref;

            pass_or_fail(check, boba::abs(x_quad_view({row, col}) - x_true), tol);
            pass_or_fail(check, boba::abs(y_quad_view({row, col}) - y_true), tol);
          }
        }
      }
    }
  }

  checkpoint();
  boba::finalize();

  return final_check(check);
}