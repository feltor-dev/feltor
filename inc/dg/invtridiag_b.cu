#define __STDCPP_WANT_MATH_SPEC_FUNCS__ 1
#include <boost/math/special_functions/jacobi_elliptic.hpp>

#undef DG_BENCHMARK


#include <iostream>
#include <iomanip>

#include "matrixsqrt.h"

#include "backend/timer.h"
#include <cusp/print.h>

using coo_type =  cusp::coo_matrix<int, double, cusp::host_memory>;

int main()
{
    dg::Timer t;
    unsigned size = 5;
//     std::vector<double> a(size,1.);
//     std::vector<double> b(size,2.);
//     std::vector<double> c(size,3.);
    std::vector<double> a = {1.98242, 4.45423, 5.31867, 7.48144, 7.11534};
    std::vector<double> b = {-0.00710891, -0.054661, -0.0554193, -0.0172191, -0.297645};
    std::vector<double> c = {-1.98242, -4.44712, -5.26401, -7.42602, -7.09812}; 

    InvTridiag<dg::HVec> invtridiag(a);
    coo_type Tinv; 
//     invtridiag.resize(size);

    t.tic();
    Tinv = invtridiag(a,b,c);
    t.toc();

    cusp::print(Tinv);
    std::cout <<  "time: "<< t.diff()<<"s \n";
    return 0;
}
