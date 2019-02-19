/** 
 * BUILD
 * g++ tests.cpp -o tests -I ../ -I (Eigen) --std=c++11 -Wall -O3
 */

#include "mmio.hpp"

using namespace mmio;

int main() {
    auto E = sp_mmread<std::complex<double>,int>("test_1.mm");
    std::cout << E << std::endl;

    auto A = sp_mmread<int,int>("test_2.mm");
    std::cout << A << std::endl;

    auto B = dense_mmread<double>("test_4.mm");
    std::cout << B << std::endl;

    auto C = sp_mmread<double,int>("neglapl_2_3.mm");
    std::cout << C << std::endl;
    sp_mmwrite("test_out_1.mm", C);

    dense_mmwrite("test_out_2.mm", B);

    Eigen::MatrixXd D = B * B.transpose();
    std::cout << D << std::endl;
    dense_mmwrite("test_out_3.mm", D, mmio::property::symmetric);
}