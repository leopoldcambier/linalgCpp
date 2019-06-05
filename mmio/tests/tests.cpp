/** 
 * BUILD
 * g++ tests.cpp -o tests -I ../ -I (Eigen) --std=c++11 -Wall -O3
 */

#include "mmio.hpp"

using namespace mmio;

int main() {
    
    std::cout << "---------------------- Reading test_1.mm" << std::endl;
    auto E = sp_mmread<std::complex<double>,int>("test_1.mm");
    std::cout << E << std::endl;

    std::cout << "---------------------- Reading test_2.mm" << std::endl;
    auto A = sp_mmread<int,int>("test_2.mm");
    std::cout << A << std::endl;

    std::cout << "---------------------- Reading test_4.mm and writing to test_out_2.mm" << std::endl;
    auto B = dense_mmread<double>("test_4.mm");
    std::cout << B << std::endl;
    dense_mmwrite("test_out_2.mm", B);
    auto B2 = dense_mmread<double>("test_out_2.mm");
    std::cout << "ERROR ? " << (B - B2).norm() << std::endl;

    std::cout << "---------------------- Reading neglapl_2_3.mm and writing to test_out_1.mm" << std::endl;
    auto C = sp_mmread<double,int>("neglapl_2_3.mm");
    std::cout << C << std::endl;
    sp_mmwrite("test_out_1.mm", C);
    auto C2 = sp_mmread<double,int>("test_out_1.mm");
    std::cout << "ERROR ? " << (C - C2).norm() << std::endl;

    std::cout << "---------------------- Writing to test_out_3.mm" << std::endl;
    Eigen::MatrixXd D = B * B.transpose();
    std::cout << D << std::endl;
    dense_mmwrite("test_out_3.mm", D, mmio::property::symmetric);

    std::cout << "---------------------- Reading to test_5.mm and writing to test_out_4.mm" << std::endl;
    auto F = sp_mmread<double,int>("test_5.mm");
    std::cout << std::setprecision(20) << F << std::endl;
    sp_mmwrite("test_out_4.mm", F);
    auto F2 = sp_mmread<double,int>("test_out_4.mm");
    std::cout << "ERROR ? " << (F - F2).norm() << std::endl;
}