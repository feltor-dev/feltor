#include <iostream>

#include <cusp/coo_matrix.h>
#include <cusp/print.h>

#include "dlt.h"
#include "functions.h"
#include "operator_tensor.cuh"

int main()
{

    std::cout << "Test of operator Matrix creation\n";
    dg::DLT<double> dlt(3);
    dg::Operator<double> Op1 = dg::create::pipj( 3);

    dg::Operator<double> Op2 = dlt.forward( );
    std::cout << "1st Operator is:\n" << Op1<<"\n";
    std::cout << "2nd Operator is:\n" << Op2<<"\n";
    std::cout << "Tensor Product is: \n" << dg::tensorproduct( Op1, Op2);
    cusp::coo_matrix<int, double, cusp::host_memory> test1 = dg::tensorproduct( 2, dg::create::delta(3));
    cusp::coo_matrix<int, double, cusp::host_memory> test2 = dg::tensorproduct( 2, Op2);
    cusp::coo_matrix<int, double, cusp::host_memory> test3 = dg::tensorproduct( 2, dg::tensorproduct( Op1, Op2));
    cusp::print(test1);
    cusp::print(test2);
    cusp::print(test3);
    std::cout << "You should see the above tensor product twice in the sparse matrix!\n";
    return 0;
}
