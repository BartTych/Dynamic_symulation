#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>

Eigen::SparseMatrix<double> build_sparse_from_triplets(const Eigen::MatrixXd& triplets, int nrows, int ncols);
void excitation_sweep(double t, double& x, double& v,
                      double f0, double f1, double T, double A);