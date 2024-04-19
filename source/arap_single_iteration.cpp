#include "../include/arap_single_iteration.h"
#include <igl/polar_svd3x3.h>
#include <igl/min_quad_with_fixed.h>

void arap_single_iteration(
  const igl::min_quad_with_fixed_data<double> & data,
  const Eigen::SparseMatrix<double> & K,
  const Eigen::MatrixXd & bc,
  Eigen::MatrixXd & U)
{
  //local step
  Eigen::MatrixXd C = K.transpose() * U;
  Eigen::MatrixXd R;
  R.resize(3 * data.n, 3); // R is a (3n, 3) matrix

  for (int k = 0; k < data.n; ++k) {
    Eigen::Matrix3d C_k = C.block<3, 3>(3 * k, 0);
    Eigen::Matrix3d R_k;
    igl::polar_svd3x3(C_k, R_k);
    R.block<3, 3>(3 * k, 0) = R_k;
  }

  //global step
  Eigen::VectorXd Beq;
  Eigen::MatrixXd B = K * R;
  igl::min_quad_with_fixed_solve(data, B, bc, Beq, U);
}
