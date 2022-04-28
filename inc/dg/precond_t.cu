#include <iostream>
#include <cusp/csr_matrix.h>
#include <cusp/print.h>
#include "precond.h"

int main()
{
    std::cout << "Test SAINV preconditioner\n";
    unsigned num = 3;
    cusp::csr_matrix<int, double, cusp::host_memory> a(num,num,num*num), z;
    thrust::host_vector<double> d, weights;
    for( unsigned i=0; i<num*num; i++)
    {
        a.column_indices[i] = i%num;
    }
    for( unsigned k=0; k<num; k++)
    {
        a.row_offsets[k] = k*num;
    }
    a.row_offsets[num] = num*num;
    //std::vector<double> values = {0.2, 0.0, 0.0,  0., 0.3, -3./35.,  0.0, -3./35., 8./245.};
    //std::vector<double> values = {193./125., -87./125., 6./125.,
    //                              -87./125, 91./250., -4./125,
    //                              6./125,-4./125, 2./125.};
    std::vector<double> values = {2., -4., 6./5.,
                                  -4., 11., -18./5.,
                                  6./5.,-18./5., 34./25.};
    a.values = values;
    weights.resize( 3, 1.);

    std::cout << "Matrix A\n";
    cusp::print( a);

    dg::create::sainv_precond( a, z, d, weights, 3, 0.01);
    std::cout << "Matrix Z\n";
    cusp::print( z);
    std::cout << "Diagonal\n";
    for( unsigned u=0; u<num; u++)
        std::cout << d[u]<<"\n";

    return 0;
}

