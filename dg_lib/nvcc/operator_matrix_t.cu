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
    dg::Operator<double, 3> Op( dg::create::detail::pipj);

    dg::Operator<double, 3> Op1( dg::DLT<3>::forward);
    cout << "1st Operator is:\n" << Op<<"\n";
    cout << "2nd Operator is:\n" << Op1<<"\n";
    cout << "Tensor Product is: \n" << dg::tensorProduct(Op, Op1);
    cusp::coo_matrix<int, double, cusp::host_memory> test1 = dg::create::operatorMatrix( Op, 2);
    cusp::coo_matrix<int, double, cusp::host_memory> test2 = dg::create::operatorMatrix( Op1, 2);
    cusp::coo_matrix<int, double, cusp::host_memory> test3 = dg::create::operatorMatrix(dg::tensorProduct( Op, Op1), 2);
    cusp::print(test1);
    cusp::print(test2);
    cusp::print(test3);
    return 0;
}
