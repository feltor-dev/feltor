#undef DG_BENCHMARK


#include <iostream>
#include <iomanip>

#include "lanczos.h"

#include "backend/timer.h"
#include <cusp/print.h>

using CooMatrix =  cusp::coo_matrix<int, double, cusp::device_memory>;
using DiaMatrix =  cusp::dia_matrix<int, double, cusp::device_memory>;

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
    DiaMatrix T; 
    
    T.resize(size, size, 3*size-2, 3);
    T.diagonal_offsets[0] = -1;
    T.diagonal_offsets[1] =  0;
    T.diagonal_offsets[2] =  1;
    for( unsigned i=0; i<size-1; i++)
    {
        T.values(i,1) =  a[i];  // 0 diagonal
        T.values(i+1,0) =  c[i];  // -1 diagonal
        T.values(i,2) =  b[i];  // +1 diagonal //dia_rows entry works since its outside of matrix
    }
    T.values(size-1,1) =  a[size-1];
    cusp::print(T);
    
    dg::InvTridiag<dg::DVec, DiaMatrix, CooMatrix> invtridiag(a);
    CooMatrix Tinv; 

    t.tic();
    Tinv = invtridiag(a,b,c);
    t.toc();

    cusp::print(Tinv);
    std::cout <<  "time: "<< t.diff()<<"s \n";
    
    t.tic();
    Tinv = invtridiag(T);
    t.toc();

    cusp::print(Tinv);
    std::cout <<  "time: "<< t.diff()<<"s \n";
    return 0;
}
