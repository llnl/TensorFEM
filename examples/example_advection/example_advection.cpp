// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

/**
 * @file example_advection.cpp
 * @brief Explicit tensor train solution of advection with prescribed inflow.
 *
 * On Omega = (0,1)^2, solve
 *
 *   u_t + a u_x + b u_y = 0,  a = cos(theta),  b = sin(theta),
 *   u(x,y,0) = 0.
 *
 * Each side of the square is assigned the same bump in its transverse
 * coordinate, multiplied by a time pulse: u(x,0,t) = u(x,1,t) = g(x) p(t)
 * and u(0,y,t) = u(1,y,t) = g(y) p(t), where r = 0.3 and
 *
 *   g(s) = exp(1/r^2 - 1/(r^2 - (s - 0.5)^2))  if |s - 0.5| < r,
 *          0                                  otherwise.
 *
 * For pulse duration D (default 0.25), the smooth time envelope is
 *
 *   p(t) = exp(4 - 1/((t/D)(1-t/D)))  if 0 < t < D,
 *          0                         otherwise.
 *
 * Thus the initially zero boundary rises to one at t=D/2, falls, and
 * returns to zero at t=D. Only the inflow portions of the boundary enter
 * the numerical solution.
 *
 * The upwind DG flux uses prescribed data only where (a,b) dot n < 0;
 * outflow is determined by the interior solution. MFEM assembles the DG
 * advection, mass, and boundary operators. BoBa carries out SSP RK3 steps
 * with a tensor train matrix and tensor train vectors. Backward
 * characteristics give an exact solution: the value is zero until a
 * characteristic reaches an inflow side, then equals the spatial bump
 * times the pulse at the time that characteristic entered the domain.
 * The default Gauss-Legendre basis may be changed to Gauss-Lobatto to
 * exercise the general element-block mass inverse.
 *
 * Run:
 *     make example_advection && ./example_advection_cpu.out --theta 0.7853981633974483
 * To display the final solution, start GLVis with `glvis -server`, then add
 * `--visualize` to the example command:
 *     make example_advection && ./example_advection_cpu.out --theta 0.7853981633974483 --visualize
 * To check convergence at T=0.1, run
 * `python3 examples/example_advection/run_convergence.py`.
 */

#include "../common.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <vector>

namespace
{

constexpr boba::execution_space space = boba::default_execution_space;
constexpr double radius = 0.3;
constexpr double relative_tolerance = 1.0e-12;
constexpr double absolute_tolerance = 1.0e-14;

double bump(double s)
{
  const double distance = s - 0.5;
  constexpr double radius_squared = radius * radius;
  if (std::abs(distance) >= radius)
  {
    return 0.0;
  }
  return std::exp(1.0 / radius_squared -
                  1.0 / (radius_squared - distance * distance));
}

double time_pulse(double time, double duration)
{
  if (time <= 0.0 || time >= duration)
  {
    return 0.0;
  }
  const double phase = time / duration;
  return std::exp(4.0 - 1.0 / (phase * (1.0 - phase)));
}

double exact_solution(double x, double y, double time, double a, double b,
                      double pulse_duration)
{
  const double infinity = std::numeric_limits<double>::infinity();
  const double time_to_x_boundary = a > 0.0 ? x / a :
                                    a < 0.0 ? (1.0 - x) / -a : infinity;
  const double time_to_y_boundary = b > 0.0 ? y / b :
                                    b < 0.0 ? (1.0 - y) / -b : infinity;
  const double boundary_time = std::min(time_to_x_boundary, time_to_y_boundary);
  if (boundary_time > time)
  {
    return 0.0;
  }
  const double amplitude = time_pulse(time - boundary_time, pulse_duration);
  return amplitude * (time_to_x_boundary <= time_to_y_boundary ?
                      bump(y - b * boundary_time) :
                      bump(x - a * boundary_time));
}

struct Parameters
{
  size_t resolution = 8;
  size_t order = 1;
  double theta = std::acos(-1.0) / 4.0;
  double end_time = 0.5;
  double pulse_duration = 0.25;
  double cfl = 0.5;
  double max_l2_error = 0.0;
  bool gauss_lobatto = false;
  bool visualize = false;
};

using Train = boba::TensorTrain<2, space, double>;
using TrainMatrix = boba::TensorTrainMatrix<2, space, double>;

void apply_inverse_dg_mass_blocks(const mfem::FiniteElementSpace& fes,
                                  const mfem::SparseMatrix& mass,
                                  boba::Matrix<boba::host_space, double>& operator_matrix,
                                  mfem::Vector& source)
{
  const int total_dofs = fes.GetVSize();
  boba_always_assert_equal(operator_matrix.rows(), static_cast<size_t>(total_dofs),
                           "Mass and operator row counts differ.");
  boba_always_assert_equal(operator_matrix.cols(), static_cast<size_t>(total_dofs),
                           "Advection operator must be square.");
  boba_always_assert_equal(source.Size(), total_dofs,
                           "Mass and source sizes differ.");

  auto operator_view = operator_matrix.view();
  std::vector<bool> visited(static_cast<size_t>(total_dofs), false);
  mfem::Array<int> dofs;
  for (int element = 0; element < fes.GetNE(); ++element)
  {
    fes.GetElementDofs(element, dofs);
    const int block_size = dofs.Size();
    mfem::DenseMatrix local_mass(block_size);
    mfem::DenseMatrix local_operator(block_size, total_dofs);
    mfem::Vector local_source(block_size), updated_source(block_size);
    for (int i = 0; i < block_size; ++i)
    {
      const int row = dofs[i];
      boba_always_assert(row >= 0 && row < total_dofs,
                         "DG mass block contains an invalid DOF.");
      boba_always_assert(!visited[static_cast<size_t>(row)],
                         "DG elements must have disjoint DOFs.");
      visited[static_cast<size_t>(row)] = true;
      local_source(i) = source(row);
      for (int j = 0; j < block_size; ++j)
      {
        local_mass(i, j) = mass(row, dofs[j]);
      }
      for (int col = 0; col < total_dofs; ++col)
      {
        local_operator(i, col) = operator_view({static_cast<size_t>(row),
                                                static_cast<size_t>(col)});
      }
    }

    mfem::DenseMatrixInverse inverse_mass(local_mass, true);
    inverse_mass.Mult(local_operator);
    inverse_mass.Mult(local_source, updated_source);
    for (int i = 0; i < block_size; ++i)
    {
      const size_t row = static_cast<size_t>(dofs[i]);
      source(dofs[i]) = updated_source(i);
      for (int col = 0; col < total_dofs; ++col)
      {
        operator_view({row, static_cast<size_t>(col)}) = local_operator(i, col);
      }
    }
  }
  boba_always_assert(std::all_of(visited.begin(), visited.end(),
                                 [](bool value) { return value; }),
                     "Every DG DOF must belong to an element mass block.");
}

Train right_hand_side(const TrainMatrix& spatial_operator,
                      const Train& source,
                      const Train& state,
                      double time,
                      double pulse_duration)
{
  auto result = spatial_operator * state;
  auto inflow = source;
  inflow *= time_pulse(time, pulse_duration);
  result += inflow;
  result.round();
  return result;
}

void step_ssp_rk3(const TrainMatrix& spatial_operator,
                  const Train& source,
                  double time,
                  double dt,
                  double pulse_duration,
                  Train& state)
{
  auto update = right_hand_side(spatial_operator, source, state,
                                time, pulse_duration);
  update *= dt;
  auto stage_one = state + update;
  stage_one.round();

  update = right_hand_side(spatial_operator, source, stage_one,
                           time + dt, pulse_duration);
  update *= dt;
  auto stage_two = stage_one + update;
  stage_two *= 0.25;
  auto old_state = state;
  old_state *= 0.75;
  stage_two += old_state;
  stage_two.round();

  update = right_hand_side(spatial_operator, source, stage_two,
                           time + 0.5 * dt, pulse_duration);
  update *= dt;
  auto next_state = stage_two + update;
  next_state *= 2.0 / 3.0;
  old_state = state;
  old_state *= 1.0 / 3.0;
  next_state += old_state;
  next_state.round();
  state = std::move(next_state);
}

} // namespace

int main(int argc, char* argv[])
{
  boba::init();

  //
  // Inputs
  //
  Parameters input;
  boba::argparser args(argc, argv);
  args.add_optional_argument(input.resolution, "-n", "--resolution", "Elements in each direction.");
  args.add_optional_argument(input.order, "-p", "--order", "DG polynomial order.");
  args.add_optional_argument(input.theta, "-t", "--theta", "Advection angle in radians.");
  args.add_optional_argument(input.end_time, "-T", "--end_time", "Final time.");
  args.add_optional_argument(input.pulse_duration, "-D", "--pulse_duration",
                             "Duration of the smooth inflow pulse.");
  args.add_optional_argument(input.cfl, "-C", "--cfl", "CFL scaling in (0, 1].");
  args.add_optional_argument(input.max_l2_error, "-E", "--max_l2_error",
                             "Absolute L2 error limit for verification; 0 disables it.");
  args.add_optional_argument(input.gauss_lobatto, "-L", "--gauss_lobatto",
                             "Use Gauss-Lobatto DG nodes (full element mass blocks).");
  args.add_optional_argument(input.visualize, "-v", "--visualize",
                             "Show the final solution in GLVis on localhost:19916.");
  args.parse_check();

  boba_always_assert(input.resolution > 0, "Resolution must be positive.");
  boba_always_assert(input.order <= 4, "This small dense-assembly example supports orders 0 through 4.");
  boba_always_assert(std::isfinite(input.theta), "Theta must be finite.");
  boba_always_assert(std::isfinite(input.end_time) && input.end_time > 0.0,
                     "End time must be finite and positive.");
  boba_always_assert(std::isfinite(input.pulse_duration) && input.pulse_duration > 0.0,
                     "Pulse duration must be finite and positive.");
  boba_always_assert(std::isfinite(input.cfl) && input.cfl > 0.0 && input.cfl <= 1.0,
                     "CFL must lie in (0, 1].");
  boba_always_assert(std::isfinite(input.max_l2_error) && input.max_l2_error >= 0.0,
                     "The L2 error limit must be finite and nonnegative.");

  //
  // Derived parameters
  //
  const double a = std::cos(input.theta);
  const double b = std::sin(input.theta);
  const size_t n = input.resolution;
  const size_t mode_size = n * (input.order + 1);
  auto mesh = tensor_fem::make_cartesian_mesh_2d(n, n, 1.0, 1.0);
  mfem::DG_FECollection fec(input.order, 2,
    input.gauss_lobatto ? mfem::BasisType::GaussLobatto :
                          mfem::BasisType::GaussLegendre);
  mfem::FiniteElementSpace fes(&mesh, &fec);
  const long double dense_matrix_bytes = sizeof(double) *
    static_cast<long double>(fes.GetVSize()) * fes.GetVSize();
  boba_always_assert_le(dense_matrix_bytes, 8.0L * 1024.0L * 1024.0L * 1024.0L,
                        "Dense 2D matrix exceeds 8 GiB.");

  // On a horizontal side g(y)=0; on a vertical side g(x)=0.
  auto boundary_data = [](double x, double y) { return bump(x) + bump(y); };

  //
  // Construct operators
  //
  mfem::VectorFunctionCoefficient velocity(2,
    [a, b](const mfem::Vector&, mfem::Vector& value)
    {
      value(0) = a;
      value(1) = b;
    });
  mfem::BilinearForm advection_form(&fes);
  // Match MFEM's advection example: M du/dt = K u + boundary_rhs.
  // Assemble(0) retains the face entries needed by the DG flux.
  advection_form.AddDomainIntegrator(
    new mfem::ConvectionIntegrator(velocity, -1.0));
  advection_form.AddInteriorFaceIntegrator(
    new mfem::NonconservativeDGTraceIntegrator(velocity, -1.0));
  advection_form.AddBdrFaceIntegrator(
    new mfem::NonconservativeDGTraceIntegrator(velocity, -1.0));
  advection_form.Assemble(0);
  advection_form.Finalize(0);
  auto mass_matrix = tensor_fem::Mass_matrix_2d_mfem_sparse(
    fes, [](double, double) { return 1.0; });

  //
  // Generate source
  //
  mfem::FunctionCoefficient inflow_data([&boundary_data](const mfem::Vector& point)
  {
    return boundary_data(point(0), point(1));
  });
  auto source_mfem = tensor_fem::inflow_boundary_vector_core(fes, velocity,
                                                              inflow_data);

  //
  // Apply inverse mass and reorder
  //
  // DG mass matrices have independent element blocks; curved cells can make
  // those blocks non-diagonal.
  const size_t total_dofs = static_cast<size_t>(fes.GetVSize());
  boba::Matrix<boba::host_space, double> host_advection({total_dofs, total_dofs});
  auto host_advection_view = host_advection.view();
  std::unique_ptr<mfem::DenseMatrix> dense_advection(
    advection_form.SpMat().ToDenseMatrix());
  for (size_t row = 0; row < total_dofs; ++row)
  {
    for (size_t col = 0; col < total_dofs; ++col)
    {
      host_advection_view({row, col}) = (*dense_advection)(
        static_cast<int>(row), static_cast<int>(col));
    }
  }
  dense_advection.reset();
  apply_inverse_dg_mass_blocks(fes, mass_matrix, host_advection, source_mfem);

  boba::Matrix<space, double> advection_mfem = host_advection;
  auto advection_matrix = tensor_fem::mfem_ordering_to_tensorfem_ordering(
    std::move(advection_mfem), n, input.order + 1, n);
  auto source_mfem_boba = tensor_fem::vector_to_boba_vector(source_mfem);
  auto source_vector = tensor_fem::mfem_ordering_to_tensorfem_ordering(
    source_mfem_boba, n, input.order + 1, n);

  //
  // Form compressed objects
  //
  auto spatial_operator = boba::compress_to_TensorTrainMatrix<2>(
    advection_matrix, {mode_size, mode_size}, {mode_size, mode_size},
    relative_tolerance, absolute_tolerance);
  auto source = boba::compress_to_TensorTrain<2>(
    boba::reshape<2>(source_vector, {mode_size, mode_size}),
    relative_tolerance, absolute_tolerance);

  //
  // Verify compressed operator
  //
  boba::Vector<boba::host_space, double> host_probe({total_dofs});
  auto probe_view = host_probe.view();
  for (size_t i = 0; i < total_dofs; ++i)
  {
    probe_view(i) = std::sin(0.17 * static_cast<double>(i + 1));
  }
  boba::Vector<space, double> probe = host_probe;
  auto full_update = advection_matrix * probe;
  full_update += source_vector;
  auto probe_train = boba::compress_to_TensorTrain<2>(
    boba::reshape<2>(probe, {mode_size, mode_size}),
    relative_tolerance, absolute_tolerance);
  auto compressed_update = right_hand_side(spatial_operator, source, probe_train,
                                           0.5 * input.pulse_duration,
                                           input.pulse_duration);
  const double operator_error = boba::norm_difference_inf(
    full_update, boba::flatten(compressed_update.decompress()));
  boba_always_assert(operator_error < 1.0e-8,
                     "Tensor train update differs from the assembled MFEM update.");

  Train solution({mode_size, mode_size});
  solution.fill_with_zeros();

  //
  // Perform timestepping
  //
  const double speed_sum = std::abs(a) + std::abs(b);
  const double h = 1.0 / static_cast<double>(n);
  const double max_dt = input.cfl * h /
                        ((2.0 * static_cast<double>(input.order) + 1.0) * speed_sum);
  double time = 0.0;
  size_t steps = 0;

  while (time < input.end_time)
  {
    const double remaining = input.end_time - time;
    const double dt = std::min(max_dt, remaining);
    step_ssp_rk3(spatial_operator, source, time, dt,
                 input.pulse_duration, solution);
    time = (remaining <= max_dt) ? input.end_time : time + dt;
    ++steps;
  }

  //
  // Verification
  //
  auto final_tensor = solution.decompress();
  auto final_vector = boba::flatten(final_tensor);
  auto grid_function = tensor_fem::make_gf_2d(mesh, fes, final_vector, input.order);
  mfem::FunctionCoefficient exact_coefficient([=](const mfem::Vector& point)
  {
    return exact_solution(point(0), point(1), time, a, b,
                          input.pulse_duration);
  });
  const mfem::IntegrationRule* integration_rules[mfem::Geometry::NumGeom] = {};
  integration_rules[mfem::Geometry::SQUARE] =
    &mfem::IntRules.Get(mfem::Geometry::SQUARE, 40);

  const double l2_error = grid_function.ComputeL2Error(exact_coefficient, integration_rules);
  const double exact_l2_norm = mfem::ComputeLpNorm(2.0, exact_coefficient, mesh,
                                                   integration_rules);
  const double relative_l2_error = exact_l2_norm > 0.0 ?
                                   l2_error / exact_l2_norm :
                                   std::numeric_limits<double>::quiet_NaN();

  boba_always_assert(std::isfinite(l2_error) && std::isfinite(exact_l2_norm),
                     "The exact-solution error and norm must be finite.");
  if (input.max_l2_error > 0.0)
  {
    boba_always_assert(l2_error <= input.max_l2_error,
                       "L2 error exceeds the requested verification limit.");
  }
  if constexpr (not tensor_fem::is_tensorfem_ci())
  {
    std::ofstream mesh_file("example_advection.mesh");
    boba_always_assert(mesh_file.good(), "Cannot open example_advection.mesh.");
    mesh.Print(mesh_file);
    grid_function.Save("example_advection.gf");
  }

  const double final_norm = boba::norm_frobenius(final_tensor);
  boba_always_assert(std::isfinite(final_norm),
                     "Expected a finite solution driven by inflow.");
  if (input.visualize)
  {
    if constexpr (tensor_fem::is_tensorfem_ci())
    {
      std::cerr << "GLVis visualization is disabled in this CI build.\n";
    }
    else
    {
      mfem::socketstream visualization_socket;
      visualization_socket.open("localhost", 19916);
      if (visualization_socket.good())
      {
        tensor_fem::glvis_visualize(mesh, grid_function, visualization_socket,
                                   "Advection final solution");
      }
      else
      {
        std::cerr << "Could not connect to GLVis on localhost:19916. Start"
                  << " `glvis -server` before using --visualize. The solution"
                  << " was saved to example_advection.mesh and"
                  << " example_advection.gf.\n";
      }
    }
  }
  std::cout << "theta=" << input.theta << " a=" << a << " b=" << b
            << " steps=" << steps << " time=" << time
            << " pulse_duration=" << input.pulse_duration
            << " operator_error=" << operator_error
            << " solution_norm=" << final_norm
            << " l2_error=" << l2_error
            << " relative_l2_error=" << relative_l2_error << '\n';
  if constexpr (not tensor_fem::is_tensorfem_ci())
  {
    std::cout << "Wrote example_advection.mesh and example_advection.gf\n";
  }
  boba::finalize();
  return 0;
}
