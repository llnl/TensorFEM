#pragma once

#include "../../include/tensorfem/tensor_fem.hpp"

/*
 * Implements the coefficients associated with spherical harmonics
 * Consistent with https://mathworld.wolfram.com/SphericalHarmonic.html
 */

double harmonic_coefficient(int m, int l)
{
  double dm = static_cast<double>(m);
  double dl = static_cast<double>(l);
  double facdifflm = static_cast<double>(boba::factorial(l - m));
  double facsumlm = static_cast<double>(boba::factorial(l + m));
  double scale2 = (2.0 * dl + 1.0) * facdifflm / (4.0 * boba::pi * facsumlm);
  return boba::sqrt(scale2);
}

/*
 * Implements Legendre Polynomials
 * Consistent with https://mathworld.wolfram.com/LegendrePolynomial.html
 */

double legendre(double x, int l)
{
  boba_always_assert_le(boba::abs(x), 1.0, "-1 <= x <= 1");

  if (l == 0)
  {
    return 1.0; // P_0(x) = 1
  }
  if (l == 1)
  {
    return x; // P_1(x) = x
  }

  // Recurrence relation:
  // P_l(x) = ((2l-1)x P_(l-1)(x) - (l-1)P_(l-2)(x)) / l
  double p0 = 1.0; // P_0(x)
  double p1 = x;   // P_1(x)
  double pl = 0.0;

  for (size_t i = 2; i <= static_cast<size_t>(l); ++i)
  {
    double di = static_cast<double>(i);
    double a = 2.0 * di - 1.0;
    double b = di - 1.0;
    pl = (a * x * p1 - b * p0) / di;
    p0 = p1;
    p1 = pl;
  }

  return pl;
}

/*
 * Implements Associated Legendre Polynomials
 * Consistent with https://mathworld.wolfram.com/AssociatedLegendrePolynomial.html
 */

double associated_legendre(double x, int m, int l)
{
  if (m < 0)
  {
    double facdifflm = static_cast<double>(boba::factorial(l - boba::abs(m)));
    double facsumlm = static_cast<double>(boba::factorial(l + boba::abs(m)));
    double scale = boba::pow(-1.0, boba::abs(m)) * facdifflm / facsumlm;
    return scale * associated_legendre(x, boba::abs(m), l);
  }

  boba_assert_le(m, l, "0 <= m <= l");
  boba_assert_le(boba::abs(x), 1.0, "-1 <= x <= 1");

  // Compute the initial value P_m^m(x)
  double pmm = 1.0;
  if (m > 0)
  {
    double sign = (m % 2 == 0) ? 1.0 : -1.0; // (-1)^m
    double double_factorial = boba::double_factorial(2 * m - 1);
    pmm = sign * double_factorial * boba::pow(1.0 - x * x, m / 2.0);
  }

  if (l == m)
  {
    return pmm;
  }

  // Compute P_(m+1)^m(x)
  double d = 2.0 * m + 1.0;
  double pmp1m = x * d * pmm;

  // If l == m + 1, return P_(m+1)^m(x)
  if (l == m + 1)
  {
    return pmp1m;
  }

  // Use recurrence relation to compute P_l^m(x) for l > m + 1
  double plm = 0.0;
  for (size_t ll = m + 2; ll <= static_cast<size_t>(l); ++ll)
  {
    double dll = static_cast<double>(ll);
    double dm = static_cast<double>(m);
    double a = 2.0 * dll - 1.0;
    double b = dll + dm - 1.0;
    double c = dll - dm;
    plm = (a * x * pmp1m - b * pmm) / c;
    pmm = pmp1m;
    pmp1m = plm;
  }

  return plm;
}
