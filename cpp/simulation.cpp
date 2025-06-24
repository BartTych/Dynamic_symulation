#include "simulation.hpp"
#include <vector>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <Eigen/Dense>
#include <cmath>
#include <iostream>
#include <vector>
#include <Eigen/Sparse>


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

std::vector<int> get_fixed_dofs(const std::vector<int>& BC_nodes) {
    std::vector<int> fixed_dofs;
    fixed_dofs.reserve(BC_nodes.size());  // optional but efficient
    for (int node : BC_nodes) {
        fixed_dofs.push_back(2 * node);  // assuming 2 DoFs per node
    }
    return fixed_dofs;
}

std::vector<int> get_excitation_dofs(const std::vector<int>& BC_nodes) {
    std::vector<int> excitation_dofs;
    excitation_dofs.reserve(BC_nodes.size());
    for (int node : BC_nodes) {
        excitation_dofs.push_back(2 * node + 1);
    }
    return excitation_dofs;
}


void excitation_sweep(double t, double& x, double& v,
                      double f0 = 1.0, double f1 = 10.0, double T = 5.0, double A = 1.0) {
    double k = (f1 - f0) / T;
    double phi = 2.0 * M_PI * (f0 * t + 0.5 * k * t * t);
    x = A * std::sin(phi);
    v = A * 2.0 * M_PI * (f0 + k * t) * std::cos(phi);
}

Eigen::VectorXd build_inv_mass_vector(std::size_t number_of_nodes, double mass_per_dof = 0.03) {
    std::size_t ndofs = number_of_nodes * 2;
    Eigen::VectorXd inv_M(ndofs);
    inv_M.setConstant(1.0 / mass_per_dof);
    return inv_M;
}

PYBIND11_MODULE(simulation, m) {
    m.def("build_sparse_from_triplets", &build_sparse_from_triplets);
    m.def("excitation_sweep", &excitation_sweep, 
          pybind11::arg("t"), 
          pybind11::arg("x"), 
          pybind11::arg("v"), 
          pybind11::arg("f0") = 1.0, 
          pybind11::arg("f1") = 10.0, 
          pybind11::arg("T") = 5.0, 
          pybind11::arg("A") = 1.0);
    m.def("build_inv_mass_vector", &build_inv_mass_vector, 
          pybind11::arg("number_of_nodes"), 
          pybind11::arg("mass_per_dof") = 0.03);
}