#include <iostream>

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include "dlt.h"
#include "functions.h"
#include "operator_matrix.cuh"

using namespace std;

int main()
{

    cout << "Test of operator Matrix creation\n";
    dg::Operator<double, 3> Op1( dg::pipj);

    dg::Operator<double, 3> Op2( dg::DLT<3>::forward);
    cout << "1st Operator is:\n" << Op1<<"\n";
    cout << "2nd Operator is:\n" << Op2<<"\n";
    cout << "Tensor Product is: \n" << dg::tensor( Op1, Op2);
    cusp::coo_matrix<int, double, cusp::host_memory> test1 = dg::tensor<double, 3>( 2, dg::delta);
    cusp::coo_matrix<int, double, cusp::host_memory> test2 = dg::tensor<double, 3>( 2, dg::DLT<3>::forward);
    cusp::coo_matrix<int, double, cusp::host_memory> test3 = dg::tensor(2, dg::tensor( Op1, Op2));
    cusp::print(test1);
    cusp::print(test2);
    cusp::print(test3);
    return 0;
}
