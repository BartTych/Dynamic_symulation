#pragma once
#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <vector>

class DynamicModelForce {
public:

    // Public members
    Eigen::SparseMatrix<double> K;
    Eigen::VectorXd inv_M;
    double mass_per_dof;
    Eigen::VectorXd u,v, a, f_ext ,f_damp, f_int;
    std::vector<Eigen::VectorXd> u_log;
    std::vector<int> fixed_dofs;
    std::vector<int> excitation_dofs;
    std::vector<int> response_dofs;
    double C_stiffness;
    double damping_coefficient;
    double damping_ratio;

    DynamicModelForce(
        const Eigen::MatrixXd& triplet_matrix,
        int nrows,
        int ncols,
        const std::vector<int>& BC_nodes,
        const std::vector<int>& response_nodes,
        std::size_t number_of_nodes,
        double damping_div,
        double mass_per_dof = 0.03 * 10e-5,
        double C_stiffness =1e8
    );

    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>> run_simulation(int n_steps, double dt,int start_f ,int end_f, int log_interval);
    
    void force_excitation_sweep(double t, double& force,
                          double f0, double f1,
                          double T, double A) const;

    std::vector<std::pair<double, double>> precompute_excitation(int n_steps, double dt, 
    double f0, double f1, double T, double A) const;

    

private:
    static Eigen::SparseMatrix<double> build_sparse_from_triplets(
        const Eigen::MatrixXd& triplets, int nrows, int ncols
    );

    static std::vector<int> get_fixed_dofs(const std::vector<int>& BC_nodes);
    static std::vector<int> get_excitation_dofs(const std::vector<int>& BC_nodes);
    static std::vector<int> get_response_dofs(const std::vector<int>& end_nodes);
    static Eigen::VectorXd build_inv_mass_vector(std::size_t number_of_nodes, double mass_per_dof);
};
