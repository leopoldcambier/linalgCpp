#ifndef __MFCHOL_HPP__
#define __MFCHOL_HPP__

#include <Eigen/Core>
#include <Eigen/Cholesky>
#include <Eigen/Sparse>
#include <Eigen/SparseLU> 

#include <scotch.h>

#include <fstream>
#include <iostream>
#include <set>
#include <array>
#include <random>
#include <vector>
#include <algorithm>
#include <stdio.h>

#include <cblas.h>
#include <lapacke.h>

#include "mmio.hpp"

typedef Eigen::SparseMatrix<double> SpMat;

struct Front {
    int id;
    int start;
    int self_size;
    int nbr_size;
    std::vector<int> rows;
    std::vector<Front*> childrens;
    Front* parent;
    Eigen::MatrixXd* front;
    Front(int id_, int start_, int size_);
    void extend_add();
    void factor();
    void fwd(Eigen::VectorXd& xglob);
    void bwd(Eigen::VectorXd& xglob);
    void extract(Eigen::MatrixXd& A);
};

struct MF {
    std::vector<Front*> fronts;
    Eigen::VectorXi perm;
    Eigen::VectorXi invperm;
    MF(std::vector<Front*> fronts_, Eigen::VectorXi perm_, Eigen::VectorXi invperm_);
    void factorize();
    Eigen::VectorXd solve(Eigen::VectorXd& b);
    void extract(Eigen::MatrixXd& A);
    void print_ordering(std::string ordering_file, std::string neighbors_file, std::string tree_file);
};

#endif
