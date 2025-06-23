#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>

Eigen::SparseMatrix<double> build_sparse_from_triplets(const Eigen::MatrixXd& triplets, int nrows, int ncols);
