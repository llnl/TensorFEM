#pragma once

/**
 * @file tensor_fem.hpp
 * @brief TensorFEM utilities that bridge MFEM discretizations with BoBa tensors.
 *
 * This header contains conversion helpers, ordering transforms, finite-element
 * assembly wrappers, visualization utilities, mesh constructors, and angular
 * quadrature helpers.
 */

#include "boba.hpp"
#include "mfem.hpp"
#include <utility>

/**
 * @brief Import BoBa's size_t literal for pre-C++23 code.
 *
 * @see include/BOBA/abstractions/types.hpp
 */
using boba::operator""_z;

namespace tensor_fem
{

static constexpr ::boba::execution_space space = ::boba::default_execution_space;
static constexpr ::boba::execution_space host_space = ::boba::host_space;

/**
 * @brief Convert an MFEM sparse matrix to a BoBa dense matrix.
 *
 * @param sparse_matrix MFEM sparse matrix to copy.
 * @return BoBa matrix on TensorFEM's default execution space.
 */
::boba::Matrix<space, double> sparse_matrix_to_boba_matrix(
   const mfem::SparseMatrix& sparse_matrix)
{
   checkpoint();
   auto dense_matrix = sparse_matrix.ToDenseMatrix();
   size_t rows = dense_matrix->NumRows();
   size_t cols = dense_matrix->NumCols();

   checkpoint();
   ::boba::Matrix<host_space, double> boba_matrix({rows, cols});
   auto boba_matrix_view = boba_matrix.view();

   // Copy the matrix over
   for(size_t r = 0; r < rows; r++)
   {
      for(size_t c = 0; c < cols; c++)
      {
         boba_matrix_view({r, c}) = dense_matrix->Elem(r, c);
      }
   }

   checkpoint();
   return boba_matrix;
}

/**
 * @brief Convert an MFEM vector to a BoBa vector.
 *
 * @param mfem_vector MFEM vector to copy.
 * @return BoBa vector on TensorFEM's default execution space.
 */
::boba::Vector<space, double> vector_to_boba_vector(
   const mfem::Vector& mfem_vector)
{
   checkpoint();
   size_t num_elements = mfem_vector.Size();

   checkpoint();
   ::boba::Vector<host_space, double> boba_vector({num_elements});
   auto boba_vector_view = boba_vector.view();

   // Copy the entries of the vector over
   for(size_t i = 0; i < num_elements; ++i)
   {
      boba_vector_view(i) = mfem_vector.Elem(i);
   }

   return boba_vector;
}

/**
 * @brief Convert an MFEM bilinear form's sparse matrix to a BoBa matrix.
 *
 * @param blf Assembled MFEM bilinear form.
 * @return BoBa matrix containing the bilinear form entries.
 */
::boba::Matrix<space, double> bilinear_form_to_matrix(
   mfem::BilinearForm& blf)
{
   checkpoint();
   auto& sparse_matrix = blf.SpMat();

   return sparse_matrix_to_boba_matrix(sparse_matrix);
}

/**
 * @brief Convert a vector from MFEM 2D FES ordering to TensorFEM ordering.
 *
 * MFEM ordering is interpreted as `(li, lj, ei, ej)`. TensorFEM ordering is
 * `(li, ei, lj, ej)`, where `li` and `lj` are element-local indices and `ei`
 * and `ej` are element-global indices.
 *
 * @param mfem_vec Vector in MFEM ordering.
 * @param Nx Number of elements in the first logical direction.
 * @param basis_size Number of local basis functions in one direction.
 * @param Ny Number of elements in the second logical direction.
 * @return Vector flattened after permutation to TensorFEM ordering.
 */
boba::Vector<space, double> mfem_ordering_to_tensorfem_ordering(
   boba::Vector<space, double>& mfem_vec,
   size_t Nx,
   size_t basis_size,
   size_t Ny)
{
   auto mfem_tensor = reshape<4>(mfem_vec, {basis_size, basis_size, Nx, Ny});

   // move li lj ei ej to li ei lj ej
   ::boba::permute(mfem_tensor, {0, 2, 1, 3});
   return boba::flatten(mfem_tensor);
}

/**
 * @brief Convert a matrix from MFEM 2D FES ordering to TensorFEM ordering.
 *
 * Each matrix dimension is interpreted in MFEM ordering `(li, lj, ei, ej)` and
 * permuted to TensorFEM ordering `(li, ei, lj, ej)`.
 *
 * @param mfem_ordered Matrix in MFEM ordering.
 * @param Nx Number of elements in the first logical direction.
 * @param basis_size Number of local basis functions in one direction.
 * @param Ny Number of elements in the second logical direction.
 * @return Matrix reshaped after permutation to TensorFEM ordering.
 */
::boba::Matrix<space, double> mfem_ordering_to_tensorfem_ordering(
   ::boba::Matrix<space, double> mfem_ordered,
   size_t Nx,
   size_t basis_size,
   size_t Ny)
{
   // Reshape to tensor
   ::boba::Tensor<8, space, double> tensorfem_ordering_tensor({basis_size, basis_size, Nx, Ny, basis_size, basis_size, Nx, Ny});
   tensorfem_ordering_tensor.reshape(mfem_ordered);

   // permute so that element and cell indices are adjacent in each dimension
   ::boba::permute(tensorfem_ordering_tensor, {0, 2, 1, 3, 4, 6, 5, 7});

   // reshape to matrix
   ::boba::Matrix<space, double> tensorfem_ordered(mfem_ordered.sizes());
   tensorfem_ordered.reshape(tensorfem_ordering_tensor);
   return tensorfem_ordered;
}

/**
 * @brief Convert a vector from TensorFEM ordering to MFEM 2D FES ordering.
 *
 * TensorFEM ordering is interpreted as `(li, ei, lj, ej)` and permuted to MFEM
 * ordering `(li, lj, ei, ej)`.
 *
 * @param boba_vec Vector in TensorFEM ordering.
 * @param Nx Number of elements in the first logical direction.
 * @param basis_size Number of local basis functions in one direction.
 * @param Ny Number of elements in the second logical direction.
 * @return Vector flattened after permutation to MFEM ordering.
 */
boba::Vector<space, double> tensorfem_ordering_to_mfem_ordering(
   boba::Vector<space, double>& boba_vec,
   size_t Nx,
   size_t basis_size,
   size_t Ny)
{
   auto boba_tensor = reshape<4>(boba_vec, {basis_size, Nx, basis_size, Ny});

   // move li ei lj ej to li lj ei ej
   ::boba::permute(boba_tensor, {0, 2, 1, 3});
   return boba::flatten(boba_tensor);
}

/**
 * @brief Assemble the MFEM linear form for inflow boundary conditions.
 *
 * The direction field is supplied as an MFEM vector coefficient and the inflow
 * data as an MFEM scalar coefficient.
 *
 * MFEM's BoundaryFlowIntegrator assembles
 * `alpha / 2 * <(u . n) f, w> - beta * <abs(u . n) f, w>`, where `f` and `u`
 * are the supplied scalar and vector coefficients and `w` is the scalar test
 * function. The linear form for this DG discretization is obtained with
 * `alpha = -1` and `beta = -0.5`.
 *
 * @tparam velocity_coeff_t MFEM-compatible vector coefficient type.
 * @tparam inflow_coeff_t MFEM-compatible scalar coefficient type.
 * @param fes Finite element space for the linear form.
 * @param vfc Velocity or direction coefficient.
 * @param inflow_coeff Prescribed inflow boundary data.
 * @return Assembled MFEM right-hand-side vector.
 */
template<typename velocity_coeff_t, typename inflow_coeff_t>
mfem::Vector inflow_boundary_vector_core(
    mfem::FiniteElementSpace& fes,
    velocity_coeff_t& vfc,
    inflow_coeff_t& inflow_coeff)
{
   checkpoint();
   mfem::LinearForm inflow_rhs(&fes);

   // BoundaryFlowIntegrator coefficients for the DG inflow contribution.
   checkpoint();
   inflow_rhs.AddBdrFaceIntegrator(
      new mfem::BoundaryFlowIntegrator(
         inflow_coeff,
         vfc,
         -1.0,
         -0.5));

   checkpoint();
   inflow_rhs.Assemble();

   checkpoint();
   return inflow_rhs;
}

/**
 * @brief Build a 1D BoBa inflow boundary vector from callable coefficients.
 *
 * @param fes Finite element space for the boundary linear form.
 * @param inflow_function Callable evaluated as `inflow_function(x)`.
 * @param velocity_A Callable evaluated as the 1D velocity component `A(x)`.
 * @return BoBa vector containing the assembled inflow boundary contribution.
 */
template<typename inflow_functionlike_t, typename velocity_A_t>
::boba::Vector<space, double> make_inflow_boundary_vector_1d(
   mfem::FiniteElementSpace& fes,
   const inflow_functionlike_t& inflow_function,
   const velocity_A_t& velocity_A)
{
   checkpoint();
   auto inflow_lambda = [=](const mfem::Vector& x) -> double
   {
      return inflow_function(x(0));
   };

   mfem::FunctionCoefficient inflow_fc(inflow_lambda);

   checkpoint();
   auto velocity_lambda = [=](const mfem::Vector& x, mfem::Vector& y) -> void
   {
      y(0) = velocity_A(x(0));
   };

   mfem::VectorFunctionCoefficient vfc(1, velocity_lambda);

   checkpoint();
   auto mfem_rhs = inflow_boundary_vector_core(fes, vfc, inflow_fc);

   checkpoint();
   return vector_to_boba_vector(mfem_rhs);
}

/**
 * @brief Build a 2D BoBa inflow boundary vector in TensorFEM ordering.
 *
 * The MFEM linear form is assembled first, copied to BoBa, and then permuted
 * from MFEM ordering to TensorFEM ordering.
 *
 * @param fes Finite element space for the boundary linear form.
 * @param inflow_function Callable evaluated as `inflow_function(x, y)`.
 * @param velocity_A Callable evaluated as the x velocity component `A(x, y)`.
 * @param velocity_B Callable evaluated as the y velocity component `B(x, y)`.
 * @param feOrder Polynomial order of the finite element basis.
 * @param Nx Number of elements in the first logical direction.
 * @param Ny Number of elements in the second logical direction.
 * @return BoBa vector containing the assembled inflow boundary contribution.
 */
template<typename inflow_functionlike_t, typename velocity_A_t, typename velocity_B_t>
::boba::Vector<space, double> make_inflow_boundary_vector_2d(
   mfem::FiniteElementSpace& fes,
   const inflow_functionlike_t& inflow_function,
   const velocity_A_t& velocity_A,
   const velocity_B_t& velocity_B,
   size_t feOrder,
   size_t Nx,
   size_t Ny)
{
   checkpoint();
   auto inflow_lambda = [=](const mfem::Vector& x) -> double
   {
      return inflow_function(x(0), x(1));
   };
   mfem::FunctionCoefficient inflow_fc(inflow_lambda);

   checkpoint();
   auto velocity_lambda = [=](const mfem::Vector& x, mfem::Vector& y) -> void
   {
      y(0) = velocity_A(x(0), x(1));
      y(1) = velocity_B(x(0), x(1));
   };

   mfem::VectorFunctionCoefficient vfc(2, velocity_lambda);

   checkpoint();
   auto mfem_rhs = inflow_boundary_vector_core(fes, vfc, inflow_fc);

   checkpoint();
   auto boba_rhs_mfem_ordering = vector_to_boba_vector(mfem_rhs);

   checkpoint();
   auto basis_size = feOrder + 1_z;

   return mfem_ordering_to_tensorfem_ordering(
      boba_rhs_mfem_ordering,
      Nx,
      basis_size,
      Ny);
}

/**
 * @brief Assemble the MFEM first order derivative operator with domain and face terms.
 *
 * The operator includes the domain convection contribution plus interior and
 * boundary DG trace contributions using the supplied vector coefficient.
 *
 * MFEM's ConvectionIntegrator assembles terms of the form
 * `alpha * (Q . grad u, v)`, where `u` and `v` are the trial and test
 * variables and `Q = vfc`.
 *
 * MFEM's DGTraceIntegrator assembles terms of the form
 * `alpha * <rho_u (u . n) {v}, [w]> + beta * <rho_u abs(u . n) [v], [w]>`.
 * Here `u` and `v` are the trial and test variables and `rho` and `u` are the
 * supplied scalar and vector coefficients. The average value on a face is
 * `{v} = (v_1 + v_2) / 2`, and the jump is `[v] = v_1 - v_2` for the face
 * between elements 1 and 2. For boundary elements, `v_2 = 0`. The vector
 * coefficient `u` is assumed continuous across faces; when a scalar coefficient
 * `rho` is supplied, it is assumed discontinuous. The integrator uses the
 * upwind value `rho_u`, taken from the side into which the vector coefficient
 * `u` points. TensorFEM uses `rho_u = 1`.
 *
 * @tparam coeff_t MFEM-compatible vector coefficient type.
 * @param fes Finite element space for the bilinear form.
 * @param vfc Velocity or direction coefficient.
 * @return Sparse matrix owned by the finalized MFEM bilinear form result.
 */
template<typename coeff_t>
mfem::SparseMatrix L_matrix_core(
   mfem::FiniteElementSpace& fes,
   coeff_t& vfc)
{
   checkpoint();
   checkpoint();
   mfem::BilinearForm L_operator(&fes);
   if(fes.GetMesh()->Dimension() > 1)
   {
      L_operator.SetAssemblyLevel(mfem::AssemblyLevel::FULL);
   }
   checkpoint();

   // Domain term: alpha * (Q . grad u, v), with Q = vfc.
   {
      double alpha = -1.0;
      L_operator.AddDomainIntegrator(new mfem::TransposeIntegrator(new mfem::ConvectionIntegrator(vfc, alpha), 0));
   }
   checkpoint();

   // Interior-face term:
   // <(u . n) {v}, [w]> + 0.5 * <abs(u . n) [v], [w]>.
   {
      double alpha = 1.0;
      double beta = 0.5;
      L_operator.AddInteriorFaceIntegrator(new mfem::DGTraceIntegrator(vfc, alpha, beta));
   }
   checkpoint();

   // Boundary-face term uses the same upwind trace form as the interior faces.
   {
      double alpha = 1.0;
      double beta = 0.5;
      L_operator.AddBdrFaceIntegrator(new mfem::DGTraceIntegrator(vfc, alpha, beta));
   }

   checkpoint();
   L_operator.Assemble();
   checkpoint();
   L_operator.Finalize();
   checkpoint();

   return L_operator.SpMat();
}

/**
 * @brief Assemble a 1D first order derivative matrix with coefficient `A(x)`.
 *
 * The resulting operator represents the x-direction contribution of a first derivative operator
 * and is returned as a full BoBa matrix.
 *
 * @param fes Finite element space for the operator.
 * @param function_A Callable evaluated as `A(x)`.
 * @param feOrder Polynomial order of the finite element basis.
 * @param Nx Number of elements in the logical direction.
 * @return BoBa matrix in 1D TensorFEM ordering.
 */
template<typename functionlike_A_t>
::boba::Matrix<space, double> L_matrix_1d(
   mfem::FiniteElementSpace& fes,
   const functionlike_A_t& function_A,
   size_t feOrder,
   size_t Nx)
{
   // convert a function call like "y = f(x)"  into "f(x, y)"
   auto lambda_func = [=](const mfem::Vector &x, mfem::Vector &y)
   {
      y(0) = function_A(x(0));
   };

   mfem::VectorFunctionCoefficient vfc(1, lambda_func);

   // Get mfem matrix
   auto L_matrix_mfem_ordering = sparse_matrix_to_boba_matrix(L_matrix_core(fes, vfc));
   auto basis_size = feOrder + 1_z;

   // reshape to matrix
   ::boba::Matrix<space, double> L_matrix_full_ordering({basis_size*Nx, basis_size*Nx});

   L_matrix_full_ordering.reshape(L_matrix_mfem_ordering);

   return L_matrix_full_ordering;
}

/**
 * @brief Assemble a 2D first derivative operator sparse matrix in MFEM ordering.
 *
 * The velocity field is `A(x, y)` in the x direction and `B(x, y)` in the y
 * direction, corresponding to
 * `A(x, y) * partial / partial_x + B(x, y) * partial / partial_y`. This
 * function extracts the corresponding full matrix in MFEM degree-of-freedom
 * ordering. Reading `L_matrix_1d` first may make this function easier to parse.
 *
 * @param fes Finite element space for the operator.
 * @param function_2d_A Callable evaluated as the x velocity component.
 * @param function_2d_B Callable evaluated as the y velocity component.
 * @return MFEM sparse matrix using MFEM degree-of-freedom ordering.
 */
template<typename functionlike_A_t, typename functionlike_B_t>
mfem::SparseMatrix L_matrix_2d_mfem(mfem::FiniteElementSpace& fes, const functionlike_A_t& function_2d_A, const functionlike_B_t& function_2d_B)
{
   checkpoint();
   auto lambda_func = [=](const mfem::Vector &x, mfem::Vector &y)
   {
      y(0) = function_2d_A(x(0), x(1));
      y(1) = function_2d_B(x(0), x(1));
   };

   checkpoint();
   mfem::VectorFunctionCoefficient vfc(2, lambda_func);

   // Get mfem matrix
   checkpoint();
   return L_matrix_core(fes, vfc);
}

/**
 * @brief Assemble a 2D first derivative matrix in TensorFEM ordering.
 *
 * This wraps `L_matrix_2d_mfem`, copies the MFEM matrix to BoBa, and permutes it
 * with `mfem_ordering_to_tensorfem_ordering`.
 *
 * @see L_matrix_2d_mfem
 * @see mfem_ordering_to_tensorfem_ordering
 */
template<typename functionlike_A_t, typename functionlike_B_t>
::boba::Matrix<space, double> L_matrix_2d(mfem::FiniteElementSpace& fes, const functionlike_A_t& function_2d_A, const functionlike_B_t& function_2d_B, size_t feOrder, size_t Nx, size_t Ny)
{
   // Get mfem matrix
   auto L_matrix_mfem_ordering = sparse_matrix_to_boba_matrix(L_matrix_2d_mfem(fes, std::forward<const functionlike_A_t&>(function_2d_A), std::forward<const functionlike_B_t&>(function_2d_B)));
   auto basis_size = feOrder + 1_z;

   return mfem_ordering_to_tensorfem_ordering(L_matrix_mfem_ordering, Nx, basis_size, Ny);
}

/**
 * @brief Assemble an MFEM mass matrix with the supplied coefficient.
 *
 * @param fes Finite element space for the bilinear form.
 * @param coefficient MFEM-compatible scalar coefficient.
 * @return MFEM sparse mass matrix.
 */
template<typename coefficient_t>
mfem::SparseMatrix Mass_matrix_core(
   mfem::FiniteElementSpace& fes,
   coefficient_t coefficient)
{
   mfem::BilinearForm M(&fes);
   if(fes.GetMesh()->Dimension() > 1)
   {
      M.SetAssemblyLevel(mfem::AssemblyLevel::FULL);
   }
   M.AddDomainIntegrator(new mfem::MassIntegrator(coefficient));
   M.Assemble();
   M.Finalize();
   return M.SpMat();
}

/**
 * @brief Assemble a 1D BoBa mass matrix from a callable coefficient.
 *
 * @param fes Finite element space for the mass matrix.
 * @param function Callable evaluated as `function(x)`.
 * @return BoBa matrix containing the assembled mass matrix.
 */
template<typename func_t>
::boba::Matrix<space, double> Mass_matrix_1d(
   mfem::FiniteElementSpace& fes,
   const func_t& function)
{
   auto lambda_func = [=](const mfem::Vector &x)
   {
      return function(x(0));
   };

   mfem::FunctionCoefficient cf(lambda_func);

   return sparse_matrix_to_boba_matrix(Mass_matrix_core(fes, cf));
}

/**
 * @brief Assemble a 2D MFEM sparse mass matrix from a callable coefficient.
 *
 * @param fes Finite element space for the mass matrix.
 * @param function_2d Callable evaluated as `function_2d(x, y)`.
 * @return MFEM sparse matrix in MFEM ordering.
 */
template<typename functionlike_t>
mfem::SparseMatrix Mass_matrix_2d_mfem_sparse(mfem::FiniteElementSpace& fes, const functionlike_t& function_2d)
{
   auto lambda_func = [=](const mfem::Vector &x)
   {
      return function_2d(x(0), x(1));
   };

   mfem::FunctionCoefficient cf(lambda_func);

   // Get mfem matrix
   return Mass_matrix_core(fes, cf);
}

/**
 * @brief Assemble a 2D BoBa mass matrix in MFEM ordering.
 *
 * @param fes Finite element space for the mass matrix.
 * @param function_2d Callable evaluated as `function_2d(x, y)`.
 * @return BoBa matrix in MFEM ordering.
 */
template<typename functionlike_t>
::boba::Matrix<space, double> Mass_matrix_2d_mfem(mfem::FiniteElementSpace& fes, const functionlike_t& function_2d)
{
   auto sparse_matrix = Mass_matrix_2d_mfem_sparse(fes, std::forward<const functionlike_t>(function_2d));

   return sparse_matrix_to_boba_matrix(sparse_matrix);
}

/**
 * @brief Assemble a 2D BoBa mass matrix in TensorFEM ordering.
 *
 * @param fes Finite element space for the mass matrix.
 * @param function_2d Callable evaluated as `function_2d(x, y)`.
 * @param feOrder Polynomial order of the finite element basis.
 * @param Nx Number of elements in the first logical direction.
 * @param Ny Number of elements in the second logical direction.
 * @return BoBa matrix in TensorFEM ordering.
 */
template<typename functionlike_t>
::boba::Matrix<space, double> Mass_matrix_2d(mfem::FiniteElementSpace& fes, const functionlike_t& function_2d, size_t feOrder, size_t Nx, size_t Ny)
{
   // Get mfem matrix
   auto matrix_mfem_ordering =  Mass_matrix_2d_mfem(fes, std::forward<const functionlike_t&>(function_2d));
   auto basis_size = feOrder + 1_z;

   return mfem_ordering_to_tensorfem_ordering(matrix_mfem_ordering, Nx, basis_size, Ny);
}

/**
 * @brief Build the 1D interpolation matrix from element dofs to quadrature values.
 *
 * This is the `B` matrix described in MFEM's partial assembly performance notes.
 * Integration weights are not folded into `B^T` in TensorFEM.
 *
 * @param fes One-dimensional finite element space.
 * @return BoBa matrix mapping element dofs to element quadrature points.
 *
 * @see https://mfem.org/performance/
 */
::boba::Matrix<space, double> make_B_matrix_1d(
   mfem::FiniteElementSpace& fes)
{
   BOBA_CALI_MARK
   checkpoint();
   auto* mesh = fes.GetMesh();
   boba_always_assert_equal(mesh->Dimension(), 1, "fes is not expected dimensionality.");

   size_t num_elements = fes.GetNE();
   const mfem::FiniteElement &typical_fe = *fes.GetTypicalFE();
   mfem::Geometry::Type geom = mesh->GetElementBaseGeometry(0);

   checkpoint();
   //boba_always_assert_equal(mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::ElementDofOrdering(mfem::GetEVectorOrdering(pfes_xy)), "FES is not in tensor ordering!");

   auto feOrder = fes.GetFE(0)->GetOrder();
   const size_t nd = fes.GetFE(0)->GetDof();
   const size_t ne = mesh->GetNE();

   int ir_order = 2*feOrder + 2;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(geom, ir_order);
   const size_t nq = ir.GetNPoints();

   boba::Multiindexer<2> B_rows({nq, ne});
   boba::Multiindexer<2> B_cols({nd, ne});

   boba::Matrix<host_space, double> B_matrix({B_rows.size(), B_cols.size()});
   B_matrix.fill_with_zeros();

   auto B_matrix_view = B_matrix.view();

   checkpoint();
   for(size_t ie{0}; ie < ne; ie++)
   {
      const mfem::FiniteElement &fe = *fes.GetFE(ie);
      boba_always_assert_equal(nd, size_t(fe.GetDof()), "Elements are not all the same size!");
      mfem::Vector shape(nd);

      for(size_t q = 0; q < nq; q++)
      {
         const mfem::IntegrationPoint & ip = ir.IntPoint(q);
         fe.CalcShape(ip, shape);

         for (size_t lex_idx = 0; lex_idx < nd; lex_idx++)
         {
            auto row = B_rows.index({q, ie});
            auto col = B_cols.index({lex_idx, ie});
            B_matrix_view({row, col}) = shape[lex_idx];
         }
      }
   }

   checkpoint();
   return B_matrix;
}

/**
 * @brief Build the 1D vector of physical quadrature weights.
 *
 * These weights are often accumulated into `B^T`; TensorFEM keeps them as a
 * separate vector.
 *
 * @param fes One-dimensional finite element space.
 * @return BoBa vector containing integration weights and transformation values.
 *
 * @see https://mfem.org/performance/
 */
::boba::Vector<space, double> make_W_vector_1d(
   mfem::FiniteElementSpace& fes)
{
   BOBA_CALI_MARK
   checkpoint();
   auto* mesh = fes.GetMesh();
   boba_always_assert_equal(mesh->Dimension(), 1, "fes is not expected dimensionality.");

   size_t num_elements = fes.GetNE();
   const mfem::FiniteElement &typical_fe = *fes.GetTypicalFE();
   mfem::Geometry::Type geom = mesh->GetElementBaseGeometry(0);

   checkpoint();
   //boba_always_assert_equal(mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::ElementDofOrdering(mfem::GetEVectorOrdering(pfes_xy)), "FES is not in tensor ordering!");

   auto feOrder = fes.GetFE(0)->GetOrder();
   const size_t nd = fes.GetFE(0)->GetDof();
   const size_t ne = mesh->GetNE();

   int ir_order = 2*feOrder + 2;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(geom, ir_order);
   const size_t nq = ir.GetNPoints();

   boba::Multiindexer<2> W_mider({nq, ne});

   ::boba::Vector<host_space, double> W_vector({W_mider.size()});
   W_vector.fill_with_zeros();

   auto W_view = W_vector.view();

   checkpoint();
   for(size_t ie{0}; ie < ne; ie++)
   {
      const mfem::FiniteElement &fe = *fes.GetFE(ie);
      boba_always_assert_equal(nd, size_t(fe.GetDof()), "Elements are not all the same size!");
      mfem::ElementTransformation *T = fes.GetElementTransformation(ie);

      for(int q = 0; q < static_cast<int>(nq); q++)
      {
         const mfem::IntegrationPoint& ip = ir.IntPoint(q);
         T->SetIntPoint(&ip);
         const double w_phys = ip.weight * T->Weight();
         auto id = W_mider.index({size_t(q), size_t(ie)});
         W_view(id) = w_phys;
      }
   }

   checkpoint();
   return W_vector;
}

/**
 * @brief Build the 2D interpolation matrix from element dofs to quadrature values.
 *
 * @param fes Two-dimensional finite element space on a logically square mesh.
 * @return BoBa matrix mapping element dofs to element quadrature points.
 *
 * @see make_B_matrix_1d
 */
::boba::Matrix<space, double> make_B_matrix_2d(
   mfem::FiniteElementSpace& fes)
{
   BOBA_CALI_MARK
   checkpoint();
   auto* mesh = fes.GetMesh();
   boba_always_assert_equal(mesh->Dimension(), 2, "fes is not expected dimensionality.");

   size_t num_elements = fes.GetNE();
   const mfem::FiniteElement &typical_fe = *fes.GetTypicalFE();
   mfem::Geometry::Type geom = mesh->GetElementBaseGeometry(0);

   checkpoint();
   //boba_always_assert_equal(mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::ElementDofOrdering(mfem::GetEVectorOrdering(pfes_xy)), "FES is not in tensor ordering!");

   auto feOrder = fes.GetFE(0)->GetOrder();
   const size_t nd = fes.GetFE(0)->GetDof();
   const size_t ne = mesh->GetNE();

   int ir_order = 2*feOrder + 2;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(geom, ir_order);
   const size_t nq = ir.GetNPoints();

   auto ne_1d = boba::sqrt(ne);
   boba_always_assert_equal(ne_1d*ne_1d, ne, "Element count has non-square factorization.");
   auto nd_1d = boba::sqrt(nd);
   auto nq_1d = boba::sqrt(nq);

   boba::Multiindexer<2> elements_mider({ne_1d, ne_1d});
   boba::Multiindexer<2> quadrature_mider({nq_1d, nq_1d});
   boba::Multiindexer<2> dof_mider({nd_1d, nd_1d});

   boba::Multiindexer<4> B_rows({nq_1d, ne_1d, nq_1d, ne_1d});
   boba::Multiindexer<4> B_cols({nd_1d, ne_1d, nd_1d, ne_1d});

   boba::Matrix<host_space, double> B_matrix({B_rows.size(), B_cols.size()});
   B_matrix.fill_with_zeros();

   auto B_matrix_view = B_matrix.view();

   for(size_t ie{0}; ie < ne_1d; ie++)
   {
      for(size_t je{0}; je < ne_1d; je++)
      {
         auto e_id = elements_mider.index({ie, je});
         const mfem::FiniteElement &fe = *fes.GetFE(e_id);
         //boba_always_assert_equal(nd, size_t(fe.GetDof()), "Elements are not all the same size!");
         mfem::Vector shape(nd);
         for(size_t q = 0; q < nq; q++)
         {
            auto [iq, jq] = quadrature_mider.multiindex(q);
            const mfem::IntegrationPoint &ip = ir.IntPoint(q);
            fe.CalcShape(ip, shape);
            for (size_t lex_idx = 0; lex_idx < nd; lex_idx++)
            {
               auto [id, jd] = dof_mider.multiindex(lex_idx);
               auto row = B_rows.index({iq, ie, jq, je});
               auto col = B_cols.index({id, ie, jd, je});
               B_matrix_view({row, col}) = shape[lex_idx];
            }
         }
      }
   }

   checkpoint();
   return B_matrix;
}

/**
 * @brief Build the 2D vector of physical quadrature weights.
 *
 * @param fes Two-dimensional finite element space on a logically square mesh.
 * @return BoBa vector containing integration weights and transformation values.
 *
 * @see make_W_vector_1d
 */
::boba::Vector<space, double> make_W_vector_2d(
   mfem::FiniteElementSpace& fes)
{
   BOBA_CALI_MARK
   checkpoint();
   auto* mesh = fes.GetMesh();
   boba_always_assert_equal(mesh->Dimension(), 2, "fes is not expected dimensionality.");

   size_t num_elements = fes.GetNE();
   const mfem::FiniteElement &typical_fe = *fes.GetTypicalFE();
   mfem::Geometry::Type geom = mesh->GetElementBaseGeometry(0);

   checkpoint();
   //boba_always_assert_equal(mfem::ElementDofOrdering::LEXICOGRAPHIC, mfem::ElementDofOrdering(mfem::GetEVectorOrdering(pfes_xy)), "FES is not in tensor ordering!");

   auto feOrder = fes.GetFE(0)->GetOrder();
   const size_t nd = fes.GetFE(0)->GetDof();
   const size_t ne = mesh->GetNE();

   int ir_order = 2*feOrder + 2;
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(geom, ir_order);
   const size_t nq = ir.GetNPoints();

   auto ne_1d = boba::sqrt(ne);
   boba_always_assert_equal(ne_1d*ne_1d, ne, "Element count has non-square factorization.");
   auto nq_1d = boba::sqrt(nq);

   boba::Multiindexer<2> elements_mider({ne_1d, ne_1d});
   boba::Multiindexer<2> quadrature_mider({nq_1d, nq_1d});

   boba::Multiindexer<4> W_mider({nq_1d, ne_1d, nq_1d, ne_1d});

   ::boba::Vector<host_space, double> W_vector({W_mider.size()});
   W_vector.fill_with_zeros();

   auto W_view = W_vector.view();

   checkpoint();
   for(size_t ie{0}; ie < ne_1d; ie++)
   {
      for(size_t je{0}; je < ne_1d; je++)
      {
         auto e_id = elements_mider.index({ie, je});
         const mfem::FiniteElement &fe = *fes.GetFE(e_id);
         boba_always_assert_equal(nd, size_t(fe.GetDof()), "Elements are not all the same size!");
         mfem::ElementTransformation *T = fes.GetElementTransformation(e_id);
         for(int q = 0; q < static_cast<int>(nq); q++)
         {
            auto [iq, jq] = quadrature_mider.multiindex(q);
            const mfem::IntegrationPoint& ip = ir.IntPoint(q);
            T->SetIntPoint(&ip);
            const double w_phys = ip.weight * T->Weight();
            auto id = W_mider.index({size_t(iq), size_t(ie), size_t(jq), size_t(je)});
            W_view(id) = w_phys;
         }
      }
   }

   checkpoint();
   return W_vector;
}

/**
 * @brief Convert a 1D MFEM grid function to a BoBa vector.
 *
 * The output uses ordering `li + N * ei`, where `N` is the number of elements,
 * `li` is the element-local index, and `ei` is the element-global index.
 *
 * @param mesh MFEM mesh associated with the grid function.
 * @param fes Finite element space associated with the grid function.
 * @param gf MFEM grid function to copy.
 * @param feOrder Polynomial order of the finite element basis.
 * @return BoBa vector containing grid-function values.
 */
::boba::Vector<space, double> gf_to_boba_vector_1d(
   mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   mfem::GridFunction& gf,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   size_t N = mesh.GetNE();

   boba::Tensor<2, host_space, double> tensor_gf({feOrder+1, N});
   auto tensor_gf_view = tensor_gf.view();

   for (size_t ei = 0; ei < N; ++ei)
   {
      mfem::Array<int> vdofs;
      fes.GetElementVDofs(ei, vdofs);
      mfem::Vector func_loc(vdofs.Size());
      gf.GetSubVector(vdofs, func_loc);

      for (size_t li = 0; li < feOrder+1; ++li)
      {
         double fx = func_loc[li];
         tensor_gf_view({li, ei}) = fx;
      }
   }

   ::boba::Vector<host_space, double> gridfunction_vector({tensor_gf.size()});
   gridfunction_vector.reshape(tensor_gf);

   return gridfunction_vector;
}

/**
 * @brief Convert a 2D MFEM grid function on a square logical mesh to a BoBa vector.
 *
 * The output is permuted from `(li, lj, ei, ej)` to `(li, ei, lj, ej)`, where
 * `li` and `lj` are element-local indices and `ei` and `ej` are logical
 * element indices.
 *
 * @todo Support general `N * M` meshes instead of only `N * N` meshes.
 */
::boba::Vector<space, double> gf_to_boba_vector_2d(
   mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   mfem::GridFunction& gf,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();

   size_t N_EI = ::boba::sqrt(mesh.GetNE());
   size_t N_EJ = N_EI;
   boba_always_assert_equal(int(N_EI*N_EJ), mesh.GetNE(), "Mesh element deduction failed. Could this be a block mesh (e.g. patched NURBS)?");
   auto basis_size = feOrder+1;

   boba::Tensor<4, host_space, double> tensor_gf({basis_size, basis_size, N_EI, N_EJ});
   auto tensor_gf_view = tensor_gf.view();
   boba::Multiindexer<2> element_mider({N_EI, N_EJ});

   for (size_t ei = 0; ei < N_EI; ++ei)
   {
      for (size_t ej = 0; ej < N_EJ; ++ej)
      {
         const int e = element_mider.index({ei, ej});
         mfem::Array<int> vdofs;
         fes.GetElementVDofs(e, vdofs);
         mfem::Vector func_xy_loc(vdofs.Size());
         gf.GetSubVector(vdofs, func_xy_loc);

         for (size_t li = 0; li < basis_size; ++li)
         {
            for (size_t lj = 0; lj < basis_size; ++lj)
            {
               double f = func_xy_loc[lj * basis_size + li];
               tensor_gf_view({li, lj, ei, ej}) = f;
            }
         }
      }
   }

   // move li lj ei ej to li ei lj ej
   ::boba::permute(tensor_gf, {0, 2, 1, 3});

   ::boba::Vector<host_space, double> gridfunction_vector({tensor_gf.size()});
   gridfunction_vector.reshape(tensor_gf);

   return gridfunction_vector;
}

/**
 * @brief Convert a blockwise 2D MFEM grid function to BoBa block vectors.
 *
 * @param blocks Number of logical blocks in the mesh.
 * @param mesh MFEM mesh associated with the grid function.
 * @param fes Finite element space associated with the grid function.
 * @param gf MFEM grid function to copy.
 * @param feOrder Polynomial order of the finite element basis.
 * @return Block vector containing TensorFEM-ordered grid-function values.
 */
boba::BlockVector<::boba::Vector<space, double>> gf_to_block_boba_vector_2d(
   size_t blocks,
   mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   mfem::GridFunction& gf,
   size_t feOrder)
{
   size_t N_EI = ::boba::sqrt(mesh.GetNE()/blocks);
   size_t N_EJ = N_EI;
   boba_always_assert_equal(int(N_EI*N_EJ*blocks), mesh.GetNE(), "Mesh element deduction failed.");
   auto basis_size = feOrder+1;

   boba::BlockVector<boba::Vector<space, double>> block_vector(blocks);

   // Resize blocks for next time
   checkpoint();
   for(size_t r = 0; r < blocks; r++)
   {
      boba::Tensor<4, host_space, double> tensor_gf({basis_size, basis_size, N_EI, N_EJ});
      auto tensor_gf_view = tensor_gf.view();
      boba::Multiindexer<3> element_mider({N_EI, N_EJ, blocks});

      for (size_t ei = 0; ei < N_EI; ++ei)
      {
         for (size_t ej = 0; ej < N_EJ; ++ej)
         {
            const int e = element_mider.index({ei, ej, r});
            mfem::Array<int> vdofs;
            fes.GetElementVDofs(e, vdofs);
            mfem::Vector func_xy_loc(vdofs.Size());
            gf.GetSubVector(vdofs, func_xy_loc);

            for (size_t li = 0; li < basis_size; ++li)
            {
               for (size_t lj = 0; lj < basis_size; ++lj)
               {
                  double f = func_xy_loc[lj*basis_size + li];
                  tensor_gf_view({li, lj, ei, ej}) = f;
               }
            }
         }
      }

      // move li lj ei ej to li ei lj ej
      ::boba::permute(tensor_gf, {0, 2, 1, 3});

      block_vector(r) = boba::flatten(tensor_gf);
   }

   return block_vector;
}

/**
 * @brief Convert a general 2D MFEM grid function to a BoBa vector.
 *
 * This overload does not assume tensor-product element ordering and is
 * distinguished from the square-logical-mesh overload by the absence of
 * `feOrder`.
 */
::boba::Vector<space, double> gf_to_boba_vector_2d(
   mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   mfem::GridFunction& gf)
{
   BOBA_CALI_MARK
   checkpoint();
   ::boba::Vector<host_space, double> gridfunction_vector({static_cast<size_t>(gf.Size())});
   auto vec_view = gridfunction_vector.view();

   // Should this just be a memcpy?
   // boba::detail::memcpy<host_space, host_space>(gridfunction_vector.data(), gf.GetData(), gridfunction_vector.size());

   for(size_t long_id = 0; long_id < gridfunction_vector.size(); long_id++)
   {
      vec_view(long_id) = gf(long_id);
   }

   return gridfunction_vector;
}

/**
 * @brief Project a callable into a 1D MFEM grid function.
 *
 * @param fes Finite element space for the projected grid function.
 * @param function_1d Callable evaluated as `function_1d(x)`.
 * @return Projected MFEM grid function.
 */
template<typename functionlike>
mfem::GridFunction make_gf_1d(
   mfem::FiniteElementSpace& fes,
   const functionlike function_1d)
{
   BOBA_CALI_MARK
   checkpoint();
   mfem::FunctionCoefficient func_cf([=](const mfem::Vector &x){ return function_1d(x(0)); });
   mfem::GridFunction func_gf(&fes);
   func_gf.ProjectCoefficient(func_cf);
   return func_gf;
}

/**
 * @brief Project a callable into a 2D MFEM grid function.
 *
 * @param fes Finite element space for the projected grid function.
 * @param function_2d Callable evaluated as `function_2d(x, y)`.
 * @return Projected MFEM grid function.
 */
template<typename functionlike>
mfem::GridFunction make_gf_2d(
   mfem::FiniteElementSpace& fes,
   const functionlike function_2d)
{
   BOBA_CALI_MARK
   checkpoint();
   mfem::FunctionCoefficient func_cf([=](const mfem::Vector &x){ return function_2d(x(0), x(1)); });
   mfem::GridFunction func_gf(&fes);
   func_gf.ProjectCoefficient(func_cf);
   return func_gf;
}

/**
 * @brief Convert a BoBa vector in MFEM ordering to a 2D MFEM grid function.
 *
 * This overload makes no logical-mesh assumptions.
 */
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Vector<host_space, double>& vec)
{
   BOBA_CALI_MARK
   checkpoint();
   mfem::GridFunction gf(&fes);

   auto vec_view = vec.view();

   for(int long_id = 0; long_id < fes.GetNFDofs(); long_id++)
   {
      gf(long_id) = vec_view(long_id);
   }

   return gf;
}

template<boba::execution_space space>
   requires (space != host_space)
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Vector<space, double>& vec)
{
   BOBA_CALI_MARK
   checkpoint();
   ::boba::Vector<host_space, double> host_vec = vec;
   return make_gf_2d(mesh, fes, host_vec);
}

/**
 * @brief Convert a TensorFEM-ordered BoBa vector to a 2D MFEM grid function.
 *
 * The input corresponds to a logically Cartesian mesh of size `N * N`.
 *
 * @todo Support general `N * M` meshes instead of only `N * N` meshes.
 */
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Vector<host_space, double>& vec,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   size_t N_EI = ::boba::sqrt(mesh.GetNE());
   size_t N_EJ = N_EI;
   boba_always_assert_equal(int(N_EI*N_EJ), mesh.GetNE(), "Only square Cartesian meshes are supported.");

   mfem::GridFunction gf(&fes);

   auto vec_view = vec.view();

   ::boba::Multiindexer<4> vec_mider({feOrder+1, N_EI, feOrder+1, N_EJ});
   ::boba::Multiindexer<4> gf_mider({feOrder+1, feOrder+1, N_EI, N_EJ});

   for(size_t long_id = 0; long_id < gf_mider.size(); long_id++)
   {
      auto [li, lj, ei, ej] = gf_mider.multiindex(long_id);
      auto vec_id = vec_mider.index({li, ei, lj, ej});
      auto gf_id = gf_mider.index({li, lj, ei, ej});
      gf(gf_id) = vec_view(vec_id);
   }

   return gf;
}

/**
 * @brief Convert a device-space BoBa vector to a 2D MFEM grid function.
 *
 * This overload copies the vector to host space and forwards to the host-space
 * `make_gf_2d` overload.
 */
template<boba::execution_space space>
   requires (space != host_space)
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Vector<space, double>& vec,
   size_t feorder)
{
   BOBA_CALI_MARK
   checkpoint();
   ::boba::Vector<host_space, double> host_vec = vec;
   return make_gf_2d(mesh, fes, host_vec, feorder);
}

/**
 * @brief Convert a TensorFEM-ordered BoBa tensor to a 2D MFEM grid function.
 *
 * This is useful when the tensor corresponds to a 2D finite element space with
 * layout `tensor([li ei], [lj ej])`.
 */
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Tensor<2, host_space, double>& vec,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   size_t N_EI = ::boba::sqrt(mesh.GetNE());
   size_t N_EJ = N_EI;
   mfem::GridFunction gf(&fes);

   auto vec_view = vec.view();

   ::boba::Multiindexer<2> tensor_mider_I({feOrder+1, N_EI});
   ::boba::Multiindexer<2> tensor_mider_J({feOrder+1, N_EJ});
   ::boba::Multiindexer<4> gf_mider({feOrder+1, feOrder+1, N_EI, N_EJ});

   for(size_t long_id = 0; long_id < gf_mider.size(); long_id++)
   {
      auto [li, lj, ei, ej] = gf_mider.multiindex(long_id);
      auto id1 = tensor_mider_I.index({li, ei});
      auto id2 = tensor_mider_J.index({lj, ej});
      auto gf_id = gf_mider.index({li, lj, ei, ej});
      gf(gf_id) = vec_view({id1, id2});
   }

   checkpoint();
   return gf;
}

/**
 * @brief Convert blockwise TensorFEM-ordered BoBa tensors to a 2D MFEM grid function.
 *
 * Each block is interpreted with layout `tensor([li ei], [lj ej])`.
 */
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const boba::BlockVector<::boba::Tensor<2, host_space, double>>& vec,
   size_t feOrder)
{
   checkpoint();
   size_t blocks = vec.block_size;
   size_t elements_per_block = mesh.GetNE()/blocks;
   size_t N_EI = boba::sqrt(elements_per_block);
   size_t N_EJ = N_EI;
   boba_always_assert_equal(N_EI*N_EJ*blocks, size_t(mesh.GetNE()), "Element count deduction failed.");
   const size_t nd = fes.GetFE(0)->GetDof();
   auto nd_1d = boba::sqrt(nd);
   boba_always_assert_equal(nd_1d*nd_1d, size_t(fes.GetFE(0)->GetDof()), "Uneven number of elements per block!");

   checkpoint();
   mfem::GridFunction gf(&fes);

   checkpoint();
   ::boba::Multiindexer<2> tensor_mider_I({nd_1d, N_EI});
   ::boba::Multiindexer<2> tensor_mider_J({nd_1d, N_EJ});
   ::boba::Multiindexer<4> gf_block_mider({nd_1d, nd_1d, N_EI, N_EJ});
   ::boba::Multiindexer<5> gf_mider({nd_1d, nd_1d, N_EI, N_EJ, blocks});

   checkpoint();
   for(size_t block = 0; block < blocks; block++)
   {
      auto vec_view = vec(block).view();
      for(size_t long_id = 0; long_id < gf_block_mider.size(); long_id++)
      {
         auto [li, lj, ei, ej] = gf_block_mider.multiindex(long_id);
         auto id1 = tensor_mider_I.index({li, ei});
         auto id2 = tensor_mider_J.index({lj, ej});
         auto gf_id = gf_mider.index({li, lj, ei, ej, block});
         gf(gf_id) = vec_view({id1, id2});
      }
   }

   checkpoint();
   return gf;
}

/**
 * @brief Convert device-space block tensors to a 2D MFEM grid function.
 *
 * This overload uses an explicit finite element order and first copies each
 * block to host space.
 */
template<boba::execution_space space>
   requires (space != host_space)
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const boba::BlockVector<::boba::Tensor<2, space, double>>& vec,
   size_t feOrder)
{
   auto blocks = vec.block_size;
   boba::BlockVector<::boba::Tensor<2, host_space, double>> host_vec(blocks);
   for(size_t i = 0; i < blocks; i++)
   {
      host_vec(i) = vec(i);
   }

   return make_gf_2d(mesh, fes, host_vec, feOrder);
}

/**
 * @brief Convert a device-space tensor to a 2D MFEM grid function.
 *
 * This overload uses an explicit finite element order and first copies the
 * tensor to host space.
 */
template<boba::execution_space space>
   requires (space != host_space)
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Tensor<2, space, double>& vec,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   boba::Tensor<2, host_space, double> host_vec = vec;
   return make_gf_2d(mesh, fes, host_vec, feOrder);
}

/**
 * @brief Convert a host-space BoBa tensor train to a 2D MFEM grid function.
 *
 * The tensor train is unrolled and then handled by the tensor overload.
 */
template<size_t dimension>
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::TensorTrain<dimension, host_space, double>& tt,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   auto boba_unroll_tensor = tt.decompress();
   checkpoint();
   return make_gf_2d(mesh, fes, boba_unroll_tensor, feOrder);
}

/**
 * @brief Convert a device-space BoBa tensor train to a 2D MFEM grid function.
 *
 * This overload copies the tensor train to host space before forwarding to the
 * host-space tensor-train overload.
 */
template<size_t dimension, boba::execution_space space>
   requires (space != host_space)
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::TensorTrain<dimension, space, double>& vec,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   boba::TensorTrain<dimension, host_space, double> host_vec = vec;
   return make_gf_2d(mesh, fes, host_vec, feOrder);
}

/**
 * @brief Convert blockwise BoBa tensor trains to a 2D MFEM grid function.
 *
 * Each tensor train is unrolled and forwarded to the block tensor overload.
 */
template<size_t dimension, boba::execution_space space>
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const boba::BlockVector<::boba::TensorTrain<dimension, space, double>>& tt,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   size_t blocks = tt.block_size;
   boba::BlockVector<::boba::Tensor<dimension, space, double>> block_tensor(blocks);
   checkpoint();
   for(size_t block = 0; block < blocks; block++)
   {
      block_tensor(block) = tt(block).decompress();
   }
   checkpoint();
   return make_gf_2d(mesh, fes, block_tensor, feOrder);
}

/**
 * @brief Convert blockwise tensor trains to a log-scaled 2D MFEM grid function.
 *
 * Values are floored at `1.0e-07` before applying base-10 logarithmic scaling.
 */

template<size_t dimension, boba::execution_space space>
mfem::GridFunction make_gf_log_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const boba::BlockVector<::boba::TensorTrain<dimension, space, double>>& tt,
   size_t feOrder)
{
   checkpoint();
   size_t blocks = tt.block_size;
   auto floor_and_log = [=]__boba_host_device__(double x){ return ::boba::log10(::boba::max(x, 1.0e-07)); };
   checkpoint();
   boba::BlockVector<::boba::Tensor<dimension, space, double>> block_tensor(blocks);

   checkpoint();
   for(size_t block = 0; block < blocks; block++)
   {
      block_tensor(block) = ::boba::apply_function(tt(block).decompress(), floor_and_log);
   }

   checkpoint();
   return make_gf_2d(mesh, fes, block_tensor, feOrder);
}

/**
 * @brief Convert blockwise tensors to a log-scaled 2D MFEM grid function.
 *
 * Values are floored at `1.0e-07` before applying base-10 logarithmic scaling.
 */

template<size_t dimension, boba::execution_space space>
mfem::GridFunction make_gf_log_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const boba::BlockVector<::boba::Tensor<dimension, space, double>>& tensor,
   size_t feOrder)
{
   checkpoint();
   size_t blocks = tensor.block_size;
   auto floor_and_log = [=]__boba_host_device__(double x){ return ::boba::log10(::boba::max(x, 1.0e-07)); };
   checkpoint();
   boba::BlockVector<::boba::Tensor<dimension, space, double>> block_tensor(blocks);

   checkpoint();
   for(size_t block = 0; block < blocks; block++)
   {
      block_tensor(block) = ::boba::apply_function(tensor(block), floor_and_log);
   }

   checkpoint();
   return make_gf_2d(mesh, fes, block_tensor, feOrder);
}

/**
 * @brief Convert a TensorFEM-ordered BoBa vector to a log-scaled 2D MFEM grid function.
 *
 * Values are floored at `1.0e-07` before applying base-10 logarithmic scaling.
 */

 template<boba::execution_space space>
mfem::GridFunction make_gf_log_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Vector<space, double>& boba_unroll_tensor,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   auto floor_and_log = [=]__boba_host_device__(double x){ return ::boba::log10(::boba::max(x, 1.0e-07)); };
   auto boba_filtered_tensor = ::boba::apply_function(boba_unroll_tensor, floor_and_log);
   checkpoint();
   return make_gf_2d(mesh, fes, boba_filtered_tensor, feOrder);
}

/**
 * @brief Convert an MFEM-ordered BoBa vector to a log-scaled 2D MFEM grid function.
 *
 * Values are floored at `1.0e-07` before applying base-10 logarithmic scaling.
 */

template<boba::execution_space space>
mfem::GridFunction make_gf_log_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Vector<space, double>& boba_unroll_tensor)
{
   BOBA_CALI_MARK
   checkpoint();
   auto floor_and_log = [=]__boba_host_device__(double x){ return ::boba::log10(::boba::max(x, 1.0e-07)); };
   auto boba_filtered_tensor = ::boba::apply_function(boba_unroll_tensor, floor_and_log);
   checkpoint();
   return make_gf_2d(mesh, fes, boba_filtered_tensor);
}

/**
 * @brief Convert a BoBa tensor to a log-scaled 2D MFEM grid function.
 *
 * Values are floored at `1.0e-07` before applying base-10 logarithmic scaling.
 */

template<size_t split_dimensions, boba::execution_space space>
mfem::GridFunction make_gf_log_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Tensor<split_dimensions, space, double>& boba_unroll_tensor,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   return make_gf_log_2d(mesh, fes, boba::flatten(boba_unroll_tensor), feOrder);
}

/**
 * @brief Convert a tensor to a log-scaled 2D MFEM grid function.
 *
 * Values are floored at `1.0e-07` before applying base-10 logarithmic scaling.
 */

 template<size_t split_dimensions, boba::execution_space space>
mfem::GridFunction make_gf_log_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::Tensor<split_dimensions, space, double>& boba_unroll_tensor)
{
   BOBA_CALI_MARK
   checkpoint();
   return make_gf_log_2d(mesh, fes, boba::flatten(boba_unroll_tensor));
}

/**
 * @brief Convert a BoBa tensor train to a log-scaled 2D MFEM grid function.
 *
 * Values are floored at `1.0e-07` before applying base-10 logarithmic scaling.
 */

template<size_t dimension, boba::execution_space space>
mfem::GridFunction make_gf_log_2d(
   const mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   const ::boba::TensorTrain<dimension, space, double>& tt,
   const size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   auto boba_unroll_tensor = tt.decompress();
   return make_gf_log_2d(mesh, fes, boba_unroll_tensor, feOrder);
}

/**
 * @brief Convert a BoBa Tucker decomposition to a 2D MFEM grid function.
 */

template<size_t dimension, boba::execution_space space>
mfem::GridFunction make_gf_2d(
   const mfem::Mesh& mesh,
   const mfem::FiniteElementSpace& fes,
   const ::boba::Tucker<dimension, space, double>& tuck,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();
   auto boba_unroll_tensor = tuck.decompress();
   return make_gf_2d(mesh, fes, boba_unroll_tensor, feOrder);
}

/**
 * @brief Build a BoBa vector of physical quadrature coordinates for a 1D MFEM mesh.
 *
 * The output is ordered by element-local quadrature point index and element index.
 */

::boba::Vector<host_space, double> quad_points_to_boba_vector_1d(
   mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();

   boba_always_assert_equal(mesh.Dimension(), 1, "fes is not expected dimensionality.");

   const size_t ne = mesh.GetNE();

   // Make sure that the order of the quadrature rule is the same as in B and W
   const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(0);
   const int ir_order = 2 * static_cast<int>(feOrder) + 2;
   const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, ir_order);
   const size_t nq = ir.GetNPoints();

   boba::Tensor<2, host_space, double> tensor_x({nq, ne});
   auto tensor_x_view = tensor_x.view();

   for (size_t ei = 0; ei < ne; ++ei)
   {
      mfem::ElementTransformation* T = mesh.GetElementTransformation(static_cast<int>(ei));

      for (size_t q = 0; q < nq; ++q)
      {
         const mfem::IntegrationPoint& ip = ir.IntPoint(static_cast<int>(q));

         mfem::Vector x(mesh.SpaceDimension());
         T->Transform(ip, x);

         tensor_x_view({q, ei}) = x(0);
      }
   }

   ::boba::Vector<host_space, double> x_vector({tensor_x.size()});
   x_vector.reshape(tensor_x);

   return x_vector;
}

/**
 * @brief Build BoBa matrices of physical quadrature coordinates for a 2D MFEM mesh.
 *
 * Rows are indexed by `(iq, ie)` and columns by `(jq, je)`, where `iq` and
 * `jq` are element-local quadrature point indices and `ie` and `je` are
 * logical element indices.
 */

std::pair<
   ::boba::Matrix<host_space, double>,
   ::boba::Matrix<host_space, double>>
quad_points_to_boba_matrices_2d(
   mfem::Mesh& mesh,
   mfem::FiniteElementSpace& fes,
   size_t feOrder)
{
   BOBA_CALI_MARK
   checkpoint();

   boba_always_assert_equal(mesh.Dimension(), 2, "fes is not expected dimensionality.");

   const size_t ne = mesh.GetNE();

   const size_t N_EI = ::boba::sqrt(ne);
   const size_t N_EJ = N_EI;

   boba_always_assert_equal(int(N_EI * N_EJ), mesh.GetNE(),
   "Total number of mesh elements must be a perfect square.");

   // Make sure that the order of the quadrature rule is the same as in B and W
   const mfem::Geometry::Type geom = mesh.GetElementBaseGeometry(0);
   const int ir_order = 2 * static_cast<int>(feOrder) + 2;
   const mfem::IntegrationRule& ir = mfem::IntRules.Get(geom, ir_order);

   const size_t nq = ir.GetNPoints();
   const size_t nq_1d = ::boba::sqrt(nq);

   boba_always_assert_equal(nq_1d * nq_1d, nq,
      "Quadrature point count has non-square factorization.");

   boba::Multiindexer<2> element_mider({N_EI, N_EJ});
   boba::Multiindexer<2> quadrature_mider({nq_1d, nq_1d});

   boba::Multiindexer<2> row_mider({nq_1d, N_EI});
   boba::Multiindexer<2> col_mider({nq_1d, N_EJ});

   boba::Matrix<host_space, double> x_matrix({row_mider.size(), col_mider.size()});
   boba::Matrix<host_space, double> y_matrix({row_mider.size(), col_mider.size()});

   auto x_matrix_view = x_matrix.view();
   auto y_matrix_view = y_matrix.view();

   for (size_t ei = 0; ei < N_EI; ++ei)
   {
      for (size_t ej = 0; ej < N_EJ; ++ej)
      {
         const int e_id = static_cast<int>(element_mider.index({ei, ej}));

         mfem::ElementTransformation* T = mesh.GetElementTransformation(e_id);

         for (size_t q = 0; q < nq; ++q)
         {
            const auto [iq, jq] = quadrature_mider.multiindex(q);

            const mfem::IntegrationPoint& ip = ir.IntPoint(static_cast<int>(q));

            mfem::Vector x_phys(mesh.SpaceDimension());
            T->Transform(ip, x_phys);

            const size_t row = row_mider.index({iq, ei});
            const size_t col = col_mider.index({jq, ej});

            x_matrix_view({row, col}) = x_phys(0);
            y_matrix_view({row, col}) = x_phys(1);
         }
      }
   }

   return {std::move(x_matrix), std::move(y_matrix)};
}

/**
 * @brief Save a grid function to files for later GLVis usage.
 *
 * @see https://glvis.org/options-and-use/
 */

void glvis_save(
   const mfem::FiniteElementSpace& fespace,
   const mfem::Mesh& mesh,
   const mfem::GridFunction& gf,
   std::string name)
{
#ifndef TENSORFEM_CI
   // Save the fes
   std::ofstream fes_ofs(name + ".fes");
   fes_ofs.precision(8);
   fespace.Save(fes_ofs);
   // Save the mesh
   std::ofstream mesh_ofs(name + ".mesh");
   mesh_ofs.precision(8);
   mesh.Print(mesh_ofs);
   // Save the gridfunction
   std::ofstream sol_ofs(name + ".gf");
   sol_ofs.precision(8);
   gf.Save(sol_ofs);
#else
   boba_warn("CI mode is on, not saving!");
#endif
}

/**
 * @brief Send a grid function to GLVis through an existing socket stream.
 *
 * The socket stream is opened if needed. GLVis must already be running.
 */

void glvis_visualize(
   const mfem::Mesh& mesh,
   const mfem::GridFunction& gf,
   mfem::socketstream& sol_sock,
   std::string_view title = "")
{
#ifndef TENSORFEM_CI
   if(not(sol_sock.good()))
   {
      std::cout << "socket not good, making" << std::endl;
      sol_sock.open("localhost", 19916);
   }

   sol_sock.precision(8);
   sol_sock << "solution\n" << mesh << gf << " view 0 0 viewcenter 0 0 ";
   if(not(title.empty()))
   {
      sol_sock << " window_title '" << title << "'";
   }
   //auto color_option = std::string(36, 'p');
   //sol_sock << " keys cff" + color_option << "\n" << std::flush;
   sol_sock << std::flush;

   // TODO<feature> figure out how to use plot_caption
   // << " plot_caption '" << caption << "'"
#else
   boba_warn("CI mode is on, not plotting!");
#endif
}

/**
 * @brief Send a grid function to GLVis through a new socket stream.
 */

void glvis_visualize(
   mfem::Mesh& mesh,
   const mfem::GridFunction& gf,
   std::string_view title = "")
{
#ifndef TENSORFEM_CI
   mfem::socketstream stream;
   stream.open("localhost", 19916);
   glvis_visualize(mesh, gf, stream, title);
#else
   boba_warn("CI mode is on, not plotting!");
#endif
}

/**
 * @brief Create a 1D Cartesian mesh.
 *
 * This is a shortcut for making a unit-interval-style MFEM mesh with `N`
 * elements and explicit physical length.
 */

mfem::Mesh make_mesh_1d(size_t N, double length)
{
  mfem::Mesh mesh_x = mfem::Mesh::MakeCartesian1D(N, length);
  return mesh_x;
}

/**
 * @brief Create a 2D Cartesian quadrilateral mesh.
 *
 * This is a shortcut for making a unit-square-style MFEM mesh with logical
 * extents `N_EI x N_EJ` and explicit physical lengths in each direction.
 */

mfem::Mesh make_cartesian_mesh_2d(size_t N_EI, size_t N_EJ, double length_x, double length_y)
{
  // want LEXICOGRAPHIC ordering for comparison to 1D tensorized version
  bool space_filling_curve_ordering = false;
  bool generate_edges = true;
  mfem::Mesh mesh_xy = mfem::Mesh::MakeCartesian2D(N_EI, N_EJ, mfem::Element::QUADRILATERAL,
                                      generate_edges, length_x, length_y, space_filling_curve_ordering);
  return mesh_xy;
}

/**
 * @brief Return midpoint-rule quadrature points and weights on `[0, 1)`.
 */

auto make_midpoint_integration_1d(size_t integration_points)
{
   boba::Vector<boba::host_space, double> points({integration_points});
   boba::Vector<boba::host_space, double> weights({integration_points});

   auto points_view = points.view();
   auto weights_view = weights.view();

   auto delta = 1.0/static_cast<double>(integration_points);

   for(size_t i = 0; i < integration_points; i++)
   {
      points_view(i) = 0.5 * delta + delta * static_cast<double>(i);
      weights_view(i) = delta;
   }

   return std::tuple(points, weights);
}

/**
 * @brief Return MFEM Legendre quadrature points and weights as BoBa vectors.
 */

auto make_legendre_quadrature_1d(size_t order)
{
   const mfem::IntegrationRule &ir = mfem::IntRules.Get(mfem::Geometry::SEGMENT, order);

   size_t integration_points = ir.GetNPoints();
   boba::Vector<boba::host_space, double> points({integration_points});
   boba::Vector<boba::host_space, double> weights({integration_points});

   auto points_view = points.view();
   auto weights_view = weights.view();

   for(size_t p = 0; p < integration_points; p++)
   {
      const mfem::IntegrationPoint &ip = ir.IntPoint(p);
      double adjustedX, adjustedWeight;
      points_view(p) = ip.x;
      weights_view(p) = ip.weight;
   }

   boba::Vector<space, double> device_points = points;
   boba::Vector<space, double> device_weights = weights;
   return std::tuple(device_points, device_weights);
}

/**
 * @brief Return quadrature points and weights for spherical harmonics.
 *
 * The polar coordinate is in `[0, pi]`; the azimuthal coordinate is in
 * `[0, 2*pi]`.
 */

auto make_spherical_harmonics_quadrature(size_t polar_order, size_t azimuth_order, size_t dimension)
{
   double pi = ::boba::pi;

   boba::Vector<space, double> azimuth_angles, azimuth_weights, polar_angles, polar_weights;

   if(polar_order == 0)
   {
      polar_angles.resize(1);
      polar_angles.fill_with(boba::pi/2.0);
      polar_weights.resize(1);
      polar_weights.fill_with(2.0);
   }
   else
   {
      auto num_polar_angles = 2*polar_order;
      // Polar points in (0, pi).
      auto [unit_points, _polar_weights] = make_legendre_quadrature_1d(2 * num_polar_angles - 1);
      // Scale (0, 1) to (0, pi)
      polar_angles = boba::apply_function(unit_points, [](double x){ return boba::acos(2.0*x - 1.0); });
      polar_weights = 2.0 * _polar_weights;
   }

   if(azimuth_order == 0)
   {
      azimuth_angles.resize(1);
      azimuth_angles.fill_with(pi/2.0);
      azimuth_weights.resize(1);
      azimuth_weights.fill_with(2.0*pi);
   }
   else
   {
      auto num_azimuthal_angles  = ::boba::pow(2, dimension - 1)*azimuth_order;

      // Azimuthal points in (0, 2*pi).
      std::tie(azimuth_angles, azimuth_weights) = make_midpoint_integration_1d(num_azimuthal_angles);
      // Scale (0, 1) to (0, 2*pi)
      azimuth_angles *= 2.0*pi;
      azimuth_weights *= 2.0*pi;
   }

   return std::tuple(polar_angles, polar_weights, azimuth_angles, azimuth_weights);
}

/**
 * @brief Store points and weights for one quadrature dimension.
 */

template<::boba::execution_space execution_space = tensor_fem::space, typename data_type = double>
struct QuadratureHelper
{
  ::boba::Vector<execution_space, data_type> points;
  ::boba::Vector<execution_space, data_type> weights;

  QuadratureHelper() = default;

  QuadratureHelper(
    ::boba::Vector<execution_space, data_type> points_in,
    ::boba::Vector<execution_space, data_type> weights_in)
    : points(std::move(points_in)),
      weights(std::move(weights_in))
  {
  }

  boba::index_t size() const
  {
    return points.size();
  }

  ::boba::Matrix<execution_space, data_type> make_identity_matrix() const
  {
    ::boba::Matrix<execution_space, data_type> identity({size(), size()});
    identity.set_to_identity_matrix();
    return identity;
  }

  ::boba::Matrix<execution_space, data_type> make_weight_matrix() const
  {
    ::boba::Matrix<execution_space, data_type> weight_matrix({size(), size()});
    auto weight_matrix_view = weight_matrix.view();
    auto weights_view = weights.const_view();

    ::boba::loop<execution_space, 2>(weight_matrix.sizes(),
      [=]__boba_host_device__(::boba::Array<boba::index_t, 2> ij)
      {
        weight_matrix_view(ij) = weights_view(ij[1]);
      });

    return weight_matrix;
  }
};

/**
 * @brief Store a spherical-harmonics angular quadrature scheme.
 */

template<::boba::execution_space execution_space = tensor_fem::space, typename data_type = double>
struct AngularDiscretization
{
  using dimension_t = QuadratureHelper<execution_space, data_type>;
  using index_t = boba::index_t;

  index_t polar_order = 0;
  index_t azimuth_order = 0;
  index_t dimension = 3;

  dimension_t polar;
  dimension_t azimuth;

  AngularDiscretization() = default;

  explicit AngularDiscretization(index_t angular_order, index_t dimension_in = 3)
    : AngularDiscretization(angular_order, angular_order, dimension_in)
  {
  }

  AngularDiscretization(index_t polar_order_in, index_t azimuth_order_in, index_t dimension_in)
    : polar_order(polar_order_in),
      azimuth_order(azimuth_order_in),
      dimension(dimension_in)
  {
    std::tie(polar.points, polar.weights, azimuth.points, azimuth.weights)
      = make_spherical_harmonics_quadrature(polar_order, azimuth_order, dimension);
  }

  index_t num_polar_angles() const
  {
    return polar.size();
  }

  index_t num_azimuth_angles() const
  {
    return azimuth.size();
  }

  auto views()
  {
    return std::make_pair(azimuth.points.view(), polar.points.view());
  }

  auto const_views() const
  {
    return std::make_pair(azimuth.points.const_view(), polar.points.const_view());
  }

  auto make_identity_matrices() const
  {
    return std::make_pair(polar.make_identity_matrix(), azimuth.make_identity_matrix());
  }

  auto make_weight_matrices() const
  {
    return std::make_pair(polar.make_weight_matrix(), azimuth.make_weight_matrix());
  }

  auto make_scattering_matrices() const
  {
    return make_weight_matrices();
  }
};

/**
 * @brief Generate logarithmically spaced points, similar to MATLAB's `logspace`.
 *
 * This produces the same result as MATLAB's
 * `logspace(log10(start), log10(end), number_points)`.
 *
 * @param start First point in the sequence.
 * @param end Last point in the sequence.
 * @param number_points Number of points to generate.
 * @return Host-space BoBa vector containing logarithmically spaced points.
 */

::boba::Vector<host_space, double> logspace(double start, double end, size_t number_points)
{
  ::boba::Vector<host_space, double> output({number_points});

  auto output_view = output.view();
  double log_start = boba::log10(start);
  double log_end = boba::log10(end);
  double step = (log_end - log_start) / static_cast<double>(number_points - 1);

  for (size_t i = 0; i < number_points; ++i) {
    output_view(i) = boba::pow(10.0, log_start + static_cast<double>(i) * step);
  }

  return output;
}

}
