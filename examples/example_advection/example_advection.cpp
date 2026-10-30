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
 *
 * Run: make example_advection && ./example_advection_cpu.out --theta 0.7853981633974483
 * To display the final solution, start GLVis with `glvis -server`, then add
 * `--visualize` to the example command.
 * To check convergence at T=0.1, run
 * `python3 examples/example_advection/run_convergence.py`.
 */

#include "../common.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>

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
  bool visualize = false;
};

using Train = boba::TensorTrain<2, space, double>;
using TrainMatrix = boba::TensorTrainMatrix<2, space, double>;

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
  Parameters input;
  boba::argparser args(argc, argv);
  args.add_optional_argument(input.resolution, "-n", "--resolution", "Elements in each direction.");
  args.add_optional_argument(input.order, "-p", "--order", "DG polynomial order.");
  args.add_optional_argument(input.theta, "-t", "--theta", "Advection angle in radians.");
  args.add_optional_argument(input.end_time, "-T", "--end_time", "Final time.");
  args.add_optional_argument(input.pulse_duration, "-D", "--pulse_duration",
                             "Duration of the smooth inflow pulse.");
  args.add_optional_argument(input.cfl, "-C", "--cfl", "CFL scaling in (0, 1].");
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

  const double a = std::cos(input.theta);
  const double b = std::sin(input.theta);
  const size_t n = input.resolution;
  const size_t mode_size = n * (input.order + 1);
  auto mesh = tensor_fem::make_cartesian_mesh_2d(n, n, 1.0, 1.0);
  mfem::DG_FECollection fec(input.order, 2, mfem::BasisType::GaussLegendre);
  mfem::FiniteElementSpace fes(&mesh, &fec);
  const long double dense_matrix_bytes = sizeof(double) *
    static_cast<long double>(fes.GetVSize()) * fes.GetVSize();
  if (dense_matrix_bytes > 256.0L * 1024.0L * 1024.0L)
  {
    std::cerr << "Resolution " << n << " would require more than 256 MiB for"
              << " one dense 2D matrix. Use a smaller mesh until 2D operator"
              << " compression is optimized.\n";
    boba::finalize();
    return 1;
  }

  auto velocity_x = [a](double, double) { return a; };
  auto velocity_y = [b](double, double) { return b; };
  // On a horizontal side g(y)=0; on a vertical side g(x)=0.
  auto boundary_data = [](double x, double y) { return bump(x) + bump(y); };

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
  auto advection_matrix = tensor_fem::mfem_ordering_to_tensorfem_ordering(
    tensor_fem::bilinear_form_to_matrix(advection_form),
    n, input.order + 1, n);
  auto mass_matrix = tensor_fem::Mass_matrix_2d(
    fes, [](double, double) { return 1.0; }, input.order, n, n);
  auto boundary_vector = tensor_fem::make_inflow_boundary_vector_2d(
    fes, boundary_data, velocity_x, velocity_y, input.order, n, n);

  // The affine Cartesian mesh and Gauss-Legendre nodal basis give a diagonal
  // exact mass matrix. Check that before using its inverse as a row scaling.
  boba::Matrix<boba::host_space, double> host_mass = mass_matrix;
  auto mass_view = host_mass.const_view();
  boba::Vector<boba::host_space, double> host_inverse_mass({host_mass.rows()});
  auto inverse_view = host_inverse_mass.view();
  double largest_diagonal = 0.0;
  double largest_off_diagonal = 0.0;
  for (size_t row = 0; row < host_mass.rows(); ++row)
  {
    const double diagonal = mass_view({row, row});
    boba_always_assert(diagonal > 0.0, "Mass matrix must be positive definite.");
    inverse_view(row) = 1.0 / diagonal;
    largest_diagonal = std::max(largest_diagonal, std::abs(diagonal));
    for (size_t col = 0; col < host_mass.cols(); ++col)
    {
      if (row != col)
      {
        largest_off_diagonal = std::max(largest_off_diagonal,
                                        std::abs(mass_view({row, col})));
      }
    }
  }
  boba_always_assert(largest_off_diagonal <= 1.0e-11 * largest_diagonal,
                     "Mass matrix is not diagonal at the requested order.");

  boba::Vector<space, double> inverse_mass = host_inverse_mass;
  boba::apply_as_diagonal_left_in_place(inverse_mass, advection_matrix);
  auto source_vector = boba::elementwise_product(inverse_mass, boundary_vector);
  if (n <= 8)
  {
    mfem::ConstantCoefficient one(1.0);
    mfem::GridFunction constant_gf(&fes);
    constant_gf.ProjectCoefficient(one);
    auto raw_boundary = tensor_fem::inflow_boundary_vector_core(
      fes, velocity, one);
    mfem::Vector raw_residual(fes.GetVSize());
    advection_form.SpMat().Mult(constant_gf, raw_residual);
    raw_residual += raw_boundary;
    boba_always_assert(raw_residual.Norml2() < 1.0e-10,
                       "DG advection assembly must preserve constant states.");
  }

  auto spatial_operator = boba::compress_to_TensorTrainMatrix<2>(
    advection_matrix, {mode_size, mode_size}, {mode_size, mode_size},
    relative_tolerance, absolute_tolerance);
  auto source = boba::compress_to_TensorTrain<2>(
    boba::reshape<2>(source_vector, {mode_size, mode_size}),
    relative_tolerance, absolute_tolerance);

  // Check a nonzero state as well as the inflow term against the assembled
  // full matrix. The zero initial state alone would only exercise the source.
  boba::Vector<boba::host_space, double> host_probe({static_cast<size_t>(fes.GetVSize())});
  auto probe_view = host_probe.view();
  for (size_t i = 0; i < host_probe.size(); ++i)
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
  std::ofstream mesh_file("example_advection.mesh");
  boba_always_assert(mesh_file.good(), "Cannot open example_advection.mesh.");
  mesh.Print(mesh_file);
  grid_function.Save("example_advection.gf");

  const double final_norm = boba::norm_frobenius(final_tensor);
  boba_always_assert(std::isfinite(final_norm),
                     "Expected a finite solution driven by inflow.");
  if (input.visualize)
  {
    tensor_fem::glvis_visualize(mesh, grid_function, "Advection final solution");
  }
  std::cout << "theta=" << input.theta << " a=" << a << " b=" << b
            << " steps=" << steps << " time=" << time
            << " pulse_duration=" << input.pulse_duration
            << " operator_error=" << operator_error
            << " solution_norm=" << final_norm
            << " l2_error=" << l2_error
            << " relative_l2_error=" << relative_l2_error << '\n';
  std::cout << "Wrote example_advection.mesh and example_advection.gf\n";
  boba::finalize();
  return 0;
}
