
#include "integration.hpp"

#include "../common.hpp"

struct parameters
{
  size_t order = 4;
};

/*
 * This tutorial demonstrates how the quadrature in make_spherical_harmonics_quadrature
 * exactly integrates spherical harmonics
 * We want to show that using numerical quadrature we can compute

    \int\int  Y^m_l * conj(Y^m_l') sin(polar) d\polar d\azimuth = delta(m - m')*delta(l - l')

  Note,
    real(Y^m_l(polar, azimuth)) = harmonic_coefficient(l, m)*P^m_l(cos(\polar))*sin(m \azimuth)
    imag(Y^m_l(polar, azimuth)) = harmonic_coefficient(l, m)*P^m_l(cos(\polar))*cos(m \azimuth)
  Also,
    (a + bi)*conj(c + di) = (a + bi)*(c - di) = a*c + b*d - (a*d - b*c)*i

  Given an integration order, we compute the associated numerical quadrature points and weights
  These are then used to verify the above identity using numerical integration of
  the spherical harmonics functions
 */

static constexpr ::boba::execution_space space = ::boba::default_execution_space;
static constexpr auto pi = ::boba::pi;

int main(int argc, char* argv[])
{
  bool check = true;
  checkpoint();
  boba::init();

  size_t dimension = 2;

  parameters input;

  ::boba::argparser args(argc, argv);

  args.add_optional_argument(
    input.order,
    "-p",
    "--order",
    "Integration order.");

  args.parse_check();

  tensor_fem::AngularDiscretization<> angular_discretization(input.order, 3);
  auto& polar_angles = angular_discretization.polar.points;
  auto& polar_weights = angular_discretization.polar.weights;
  auto& azimuth_angles = angular_discretization.azimuth.points;
  auto& azimuth_weights = angular_discretization.azimuth.weights;

  polar_angles.rename("polar_angles");
  polar_angles.print();

  polar_weights.rename("polar_weights");
  polar_weights.print();

  azimuth_angles.rename("azimuth_angles");
  azimuth_angles.print();

  azimuth_weights.rename("azimuth_weights");
  azimuth_weights.print();

  auto polar_angles_view = polar_angles.const_view();
  auto polar_weights_view = polar_weights.const_view();
  auto azimuth_angles_view = azimuth_angles.const_view();
  auto azimuth_weights_view = azimuth_weights.const_view();
  ::boba::Multiindexer<2> integration_mider({angular_discretization.num_polar_angles(), angular_discretization.num_azimuth_angles()});

  // Test spherical harmonics orthogonalization
  if (input.order > 0)
  {
    // Quadrature should be exact up to input.order
    int max_test_order = input.order;
    for (int l1 = 0; l1 < max_test_order; l1++)
    {
      for (int l2 = 0; l2 < max_test_order; l2++)
      {
        for (int m1 = -l1; m1 <= l1; m1++)
        {
          for (int m2 = -l2; m2 <= l2; m2++)
          {
            // Given m1, l1, m2, l2, compute
            // <Y^m1_l1, Y^m2_l2>_sin(polar)
            // (note that sin(polar) is implicitly computed in the quadrature weights)

            double expected_real_result = ((m1 == m2) and (l1 == l2)) ? 1.0 : 0.0;
            double expected_imag_result = 0.0;

            double real_integral = 0.0;
            double imag_integral = 0.0;
            for (size_t i = 0; i < integration_mider.size(); i++)
            {
              auto [polar_i, azimuth_j] = integration_mider.multiindex(i);

              double azimuth = azimuth_angles_view(azimuth_j);
              double polar = polar_angles_view(polar_i);
              double azimuth_weight = azimuth_weights_view(azimuth_j);
              double polar_weight = polar_weights_view(polar_i);
              double weight = azimuth_weight * polar_weight;

              double cos_polar = boba::cos(polar);
              double sin_polar = boba::sin(polar);

              // Compute Y^m1_l1
              // see integration.hpp for supporting functions
              double scale1 = harmonic_coefficient(m1, l1);
              double pm1l1 = associated_legendre(cos(polar), m1, l1);
              double cos_m1_azimuth = boba::cos(static_cast<double>(m1) * azimuth);
              double sin_m1_azimuth = boba::sin(static_cast<double>(m1) * azimuth);
              double real1 = scale1 * pm1l1 * sin_m1_azimuth;
              double imag1 = scale1 * pm1l1 * cos_m1_azimuth;

              // Compute Y^m2_l2
              double scale2 = harmonic_coefficient(m2, l2);
              double pm2l2 = associated_legendre(cos(polar), m2, l2);
              double cos_m2_azimuth = boba::cos(static_cast<double>(m2) * azimuth);
              double sin_m2_azimuth = boba::sin(static_cast<double>(m2) * azimuth);
              double real2 = scale2 * pm2l2 * sin_m2_azimuth;
              double imag2 = scale2 * pm2l2 * cos_m2_azimuth;

              // Compute real and imag parts of integrand
              double real_integrand = (real1 * real2 + imag1 * imag2);
              double imag_integrand = (real1 * imag2 - imag1 * real2);

              real_integral += real_integrand * weight;
              imag_integral += imag_integrand * weight;
            }

            double real_error = boba::abs(real_integral - expected_real_result);
            double imag_error = boba::abs(imag_integral - expected_imag_result);

            double error_tolerance = 1.0e-4;
            pass_or_fail(check, real_error, error_tolerance);
            pass_or_fail(check, imag_error, error_tolerance);
          }
        }
      }
    }
  }
  else
  {
    int l1 = 0;
    int l2 = 0;
    int m1 = 0;
    int m2 = 0;

    double expected_real_result = 1.0;
    double expected_imag_result = 0.0;

    double real_integral = 0.0;
    double imag_integral = 0.0;
    for (size_t i = 0; i < integration_mider.size(); i++)
    {
      auto [polar_i, azimuth_j] = integration_mider.multiindex(i);

      double azimuth = azimuth_angles_view(azimuth_j);
      double polar = polar_angles_view(polar_i);
      double azimuth_weight = azimuth_weights_view(azimuth_j);
      double polar_weight = polar_weights_view(polar_i);
      double weight = azimuth_weight * polar_weight;

      double cos_polar = boba::cos(polar);
      double sin_polar = boba::sin(polar);

      // Compute Y^m1_l1
      // see integration.hpp for supporting functions
      double scale1 = harmonic_coefficient(m1, l1);
      double pm1l1 = associated_legendre(cos(polar), m1, l1);
      double cos_m1_azimuth = boba::cos(static_cast<double>(m1) * azimuth);
      double sin_m1_azimuth = boba::sin(static_cast<double>(m1) * azimuth);
      double real1 = scale1 * pm1l1 * sin_m1_azimuth;
      double imag1 = scale1 * pm1l1 * cos_m1_azimuth;

      // Compute Y^m2_l2
      double scale2 = harmonic_coefficient(m2, l2);
      double pm2l2 = associated_legendre(cos(polar), m2, l2);
      double cos_m2_azimuth = boba::cos(static_cast<double>(m2) * azimuth);
      double sin_m2_azimuth = boba::sin(static_cast<double>(m2) * azimuth);
      double real2 = scale2 * pm2l2 * sin_m2_azimuth;
      double imag2 = scale2 * pm2l2 * cos_m2_azimuth;

      // Compute real and imag parts of integrand
      double real_integrand = (real1 * real2 + imag1 * imag2);
      double imag_integrand = (real1 * imag2 - imag1 * real2);

      real_integral += real_integrand * weight;
      imag_integral += imag_integrand * weight;
    }

    double real_error = boba::abs(real_integral - expected_real_result);
    double imag_error = boba::abs(imag_integral - expected_imag_result);

    double error_tolerance = 1.0e-4;
    pass_or_fail(check, real_error, error_tolerance);
    pass_or_fail(check, imag_error, error_tolerance);
  }

  boba::finalize();
  return final_check(check);
}
