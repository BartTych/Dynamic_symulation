#include "s_loader.hpp"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>


Eigen::SparseMatrix<double> build_sparse_from_triplets(const Eigen::MatrixXd& triplets, int nrows, int ncols) {
    using namespace Eigen;
    using T = Triplet<double>;

    std::vector<T> tlist;
    tlist.reserve(triplets.rows());

    for (int i = 0; i < triplets.rows(); ++i) {
        int r = static_cast<int>(triplets(i, 0));
        int c = static_cast<int>(triplets(i, 1));
        double val = triplets(i, 2);
        tlist.emplace_back(r, c, val);
    }

    SparseMatrix<double> K(nrows, ncols);
    K.setFromTriplets(tlist.begin(), tlist.end());
    return K;
}

PYBIND11_MODULE(s_loader, m) {
    m.def("build_sparse_from_triplets", &build_sparse_from_triplets);
}