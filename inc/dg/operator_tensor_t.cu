#include <iostream>

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include "dlt.cuh"
#include "functions.h"
#include "operator_tensor.cuh"

using namespace std;

int main()
{

    cout << "Test of operator Matrix creation\n";
    dg::DLT<double> dlt(3);
    dg::Operator<double> Op1 = dg::create::pipj( 3);

    dg::Operator<double> Op2 = dlt.forward( );
    cout << "1st Operator is:\n" << Op1<<"\n";
    cout << "2nd Operator is:\n" << Op2<<"\n";
    cout << "Tensor Product is: \n" << dg::tensor( Op1, Op2);
    cusp::coo_matrix<int, double, cusp::host_memory> test1 = dg::tensor( 2, dg::create::delta(3));
    cusp::coo_matrix<int, double, cusp::host_memory> test2 = dg::tensor( 2, Op2);
    cusp::coo_matrix<int, double, cusp::host_memory> test3 = dg::tensor( 2, dg::tensor( Op1, Op2));
    cusp::print(test1);
    cusp::print(test2);
    cusp::print(test3);
    cout << "You should see the above tensor product in the sparse matrix!\n";
    return 0;
}
