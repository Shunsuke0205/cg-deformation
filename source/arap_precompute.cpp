#include "../include/arap_precompute.h"
#include <igl/min_quad_with_fixed.h>
#include <igl/arap_linear_block.h>
#include <igl/cotmatrix.h>

void arap_precompute(
  const Eigen::MatrixXd & V,
  const Eigen::MatrixXi & F,
  const Eigen::VectorXi & b,
  igl::min_quad_with_fixed_data<double> & data,
  Eigen::SparseMatrix<double> & K)
{
  Eigen::SparseMatrix<double> L, Aeq;
  igl::cotmatrix(V, F, L);
  igl::min_quad_with_fixed_precompute(L, b, Aeq, false, data);

  Eigen::MatrixXd cotangent_matrix;
  igl::cotmatrix_entries(V, F, cotangent_matrix);
  typedef Eigen::Triplet<double> Tri;
  std::vector<Tri> triplet_list;
  triplet_list.reserve(F.rows() * 3 * 6);

  K.resize(V.rows(), 3 * V.rows()); // K is a (n, 3n) matrix
  for (int i_face = 0; i_face < F.rows(); ++i_face) {
    for (int kk = 0; kk < 3; ++kk) {
      int j = (kk + 1) % 3;
      int i = (kk + 2) % 3;

      Eigen::RowVector3d constant_vector = cotangent_matrix(i_face, kk) * (V.row(F(i_face, i)) - V.row(F(i_face, j)));
      for (int k = 0; k < 3; ++k) {
        triplet_list.push_back(Tri(F(i_face, i), 3 * F(i_face, k) + 0,  constant_vector(0)));
        triplet_list.push_back(Tri(F(i_face, i), 3 * F(i_face, k) + 1,  constant_vector(1)));
        triplet_list.push_back(Tri(F(i_face, i), 3 * F(i_face, k) + 2,  constant_vector(2)));

        triplet_list.push_back(Tri(F(i_face, j), 3 * F(i_face, k) + 0, -constant_vector(0)));
        triplet_list.push_back(Tri(F(i_face, j), 3 * F(i_face, k) + 1, -constant_vector(1)));
        triplet_list.push_back(Tri(F(i_face, j), 3 * F(i_face, k) + 2, -constant_vector(2)));
      }
    }
  }
  K.setFromTriplets(triplet_list.begin(), triplet_list.end());
  K /= 3.0;
}
