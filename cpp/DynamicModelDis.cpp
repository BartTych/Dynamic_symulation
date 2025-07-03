
#include "DynamicModelDis.hpp"
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <vector>
#include <cmath>
#include <iostream>
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <chrono> 


DynamicModelForce::DynamicModelForce(
        const Eigen::MatrixXd& triplet_matrix,
        int nrows,
        int ncols,
        const std::vector<int>& BC_nodes,
        const std::vector<int>& response_nodes,
        std::size_t number_of_nodes,
        double damping,
        double mass_per_dof,
        double C_stiffness
    )
    {
        //std::cout << "damping_div = " << damping_div << std::endl;
        K = build_sparse_from_triplets(triplet_matrix, nrows, ncols);
        K.makeCompressed();
        inv_M = build_inv_mass_vector(number_of_nodes, mass_per_dof);
        this -> mass_per_dof = mass_per_dof;
        fixed_dofs = get_fixed_dofs(BC_nodes);
        excitation_dofs = get_excitation_dofs(BC_nodes);
        response_dofs = get_response_dofs(response_nodes);
        this->C_stiffness = C_stiffness;
        this->damping_coefficient = damping;
        //std::cout << "damping_coefficient = " << damping_coefficient << std::endl;
        std::size_t ndofs = number_of_nodes * 2;
        u = Eigen::VectorXd::Zero(ndofs);
        v = Eigen::VectorXd::Zero(ndofs);
        a = Eigen::VectorXd::Zero(ndofs);
        f_damp = Eigen::VectorXd::Zero(ndofs);
        f_int = Eigen::VectorXd::Zero(ndofs);
    }

std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
DynamicModelDis::run_simulation(int n_steps, double dt, int start_f, int end_f,int log_interval)
{
    //std::cout << "K: " << K.rows() << " x " << K.cols() << ", u: " << u.size() << std::endl;
    std::vector<double> exc_log;
    std::vector<double> end_log;
    std::vector<double> steps_log;

    int n_log_steps = n_steps / log_interval + 1;
    exc_log.reserve(n_log_steps);
    end_log.reserve(n_log_steps);
    steps_log.reserve(n_log_steps);

    double length = dt * n_steps;
    //auto excitation = precompute_excitation(n_steps, dt, start_f, end_f, length, 0.0001);
    std::chrono::high_resolution_clock::time_point last_time = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < n_steps; ++step) {
        double T = step * dt;
        
        // Enforce fixed boundary conditions
        for (int idx : fixed_dofs)
            u.coeffRef(idx) = v.coeffRef(idx) = 0.0;

        // Apply dynamic excitation
        for (std::size_t i = 0; i < excitation_dofs.size(); ++i) {
            excitation_sweep(T,u[excitation_dofs[i]], v[excitation_dofs[i]],
                          start_f, end_f, length, 0.0001);
        
        //u[excitation_dofs[i]] = excitation[step].first;
        //v[excitation_dofs[i]] = excitation[step].second;
        }


        // Compute internal forces
        f_int = K * u;
        f_damp = damping_coefficient * v;
        a.setZero();

        a = (-f_damp - f_int).cwiseProduct(inv_M);

        // Time integration
        v += a * dt;
        u += v * dt;

        
        // Logging
        if (step % log_interval == 0) {
            exc_log.push_back(u[excitation_dofs[0]]);
            steps_log.push_back(T);
            if (!response_dofs.empty()) {
                end_log.push_back(u[response_dofs[0]]);
            }
        }
        /*
        if (step % 1000 == 0) {
          u_log.push_back(u);  // this makes a deep copy of the vector
        }
        */

        /*
        if (step % 1000 == 0) {
        int nnz = 0;
        for (int i = 0; i < u.size(); ++i) {
            if (std::abs(u[i]) > 1e-12)  // consider as nonzero
                ++nnz;
        }
        std::cout << "step " << step
                << ", nnz(u) = " << nnz
                << ", u[exc] = " << u[excitation_dofs[0]]
                << ", u[resp] = " << u[response_dofs[0]] << std::endl;
        }
        */

        if (step % 10000 == 0) {
        auto now = std::chrono::high_resolution_clock::now();
        double elapsed = std::chrono::duration<double>(now - last_time).count();
        std::cout << "Step " << step << ", elapsed: " << elapsed << " s" << std::endl;
        
        last_time = now;
        
    }
        
        
    }
    return std::make_tuple(exc_log, end_log, steps_log);
}

std::vector<std::pair<double, double>> DynamicModelDis::precompute_excitation(int n_steps, double dt, 
    double f0, double f1, double T, double A) const
{
    std::vector<std::pair<double, double>> excitation(n_steps);
    for (int step = 0; step < n_steps; ++step) {
        double t = step * dt;
        double k = (f1 - f0) / T;
        double phi = 2.0 * M_PI * (f0 * t + 0.5 * k * t * t);
        double x = A * std::sin(phi);
        double v = A * 2.0 * M_PI * (f0 + k * t) * std::cos(phi);
        excitation[step] = {x, v};
    }
    return excitation;
}

void DynamicModelDis::excitation_sweep(double t, double& x, double& v,
                          double f0 = 1.0, double f1 = 10.0, double T = 5.0, double A = 1.0) const {
        double k = (f1 - f0) / T;
        double phi = 2.0 * M_PI * (f0 * t + 0.5 * k * t * t);
        x = A * std::sin(phi);
        v = A * 2.0 * M_PI * (f0 + k * t) * std::cos(phi);
    }

    /*
    Eigen::SparseMatrix<double> K;
    Eigen::VectorXd inv_M;
    Eigen::VectorXd u, v, a, f_damp, f_int;
    std::vector<int> fixed_dofs;
    std::vector<int> excitation_dofs;
    double C_stiffness;
    */


Eigen::SparseMatrix<double> DynamicModelDis::build_sparse_from_triplets(const Eigen::MatrixXd& triplets, int nrows, int ncols) {
        using T = Eigen::Triplet<double>;
        std::vector<T> tlist;
        tlist.reserve(triplets.rows());

        for (int i = 0; i < triplets.rows(); ++i) {
            int r = static_cast<int>(triplets(i, 0));
            int c = static_cast<int>(triplets(i, 1));
            double val = triplets(i, 2);
            tlist.emplace_back(r, c, val);
        }

        Eigen::SparseMatrix<double> K(nrows, ncols);
        K.setFromTriplets(tlist.begin(), tlist.end());
        return K;
    }

std::vector<int> DynamicModelDis::get_fixed_dofs(const std::vector<int>& BC_nodes) {
        std::vector<int> fixed_dofs;
        fixed_dofs.reserve(BC_nodes.size());
        for (int node : BC_nodes)
            fixed_dofs.push_back(2 * node);
        return fixed_dofs;
    }

std::vector<int> DynamicModelDis::get_excitation_dofs(const std::vector<int>& BC_nodes) {
        std::vector<int> excitation_dofs;
        excitation_dofs.reserve(BC_nodes.size());
        for (int node : BC_nodes)
            excitation_dofs.push_back(2 * node + 1);
        return excitation_dofs;
    }

std::vector<int> DynamicModelDis::get_response_dofs(const std::vector<int>& end_nodes) {
        std::vector<int> response_dofs;
        response_dofs.reserve(end_nodes.size());
        for (int node : end_nodes)
            response_dofs.push_back(2 * node + 1);
        return response_dofs;
    }

Eigen::VectorXd DynamicModelDis::build_inv_mass_vector(std::size_t number_of_nodes, double mass_per_dof) {
        std::size_t ndofs = number_of_nodes * 2;
        Eigen::VectorXd inv_M(ndofs);
        inv_M.setConstant(1.0 / mass_per_dof);
        return inv_M;
    }


namespace py = pybind11;
PYBIND11_MODULE(DynamicModelForce, m) {
    py::class_<DynamicModelDis>(m, "DynamicModelDis")
        .def(py::init<const Eigen::MatrixXd&, int, int, const std::vector<int>&, const std::vector<int>&, std::size_t, double, double, double>(),
             py::arg("triplet_matrix"),
             py::arg("nrows"),
             py::arg("ncols"),
             py::arg("BC_nodes"),
             py::arg("response_nodes"),
             py::arg("number_of_nodes"),
             py::arg("damping_div"),
             py::arg("mass_per_dof") = 0.03 * 10e-5,
             py::arg("C_stiffness") = 1e8)
        .def("run_simulation", &DynamicModelDis::run_simulation,  py::arg("n_steps"), py::arg("dt"),py::arg("start_f"),py::arg("end_f"),py::arg("log_interval"), py::call_guard<py::gil_scoped_release>())
        .def("excitation_sweep", &DynamicModelDis::excitation_sweep)
        .def_readwrite("K", &DynamicModelDis::K)
        .def_readwrite("u", &DynamicModelDis::u)
        .def_readwrite("u_log", &DynamicModelDis::u_log)
        .def_readwrite("v", &DynamicModelDis::v)
        .def_readwrite("a", &DynamicModelDis::a)
        .def_readwrite("inv_M", &DynamicModelDis::inv_M)
        .def_readwrite("mass_per_dof",&DynamicModelDis::mass_per_dof)
        .def_readwrite("fixed_dofs", &DynamicModelDis::fixed_dofs)
        .def_readwrite("excitation_dofs", &DynamicModelDis::excitation_dofs)
        .def_readwrite("response_dofs", &DynamicModelDis::response_dofs)
        .def_readwrite("C_stiffness", &DynamicModelDis::C_stiffness)
        .def_readwrite("damping_coefficient", &DynamicModelDis::damping_coefficient);
}
