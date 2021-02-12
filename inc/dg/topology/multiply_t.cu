#include <iostream>
#include <cmath>

#include "tensor.h"
#include "multiply.h"

template<class container>
void print( const dg::SparseTensor<container >& t)
{
    for( unsigned i=0; i<3; i++)
    {
        for( unsigned j=0; j<3; j++)
            std::cout << t.value(i,j)[0]<<" ";
        std::cout << "\n";
    }
}

const thrust::host_vector<double> one(1,1), two(1,2), three(1,3), four(1,4), five(1,5), six(1,6), seven(1,7), eight(1,8), nine(1,9), zero(1,0);

int main()
{
    std::cout << "Test dg::Sparse Tensor class \n";
    dg::SparseTensor<thrust::host_vector<double> > t;
    t.idx(0,0) = 2, t.idx(0,1) = 3, t.idx(0,2) = 0;
    t.idx(1,0) = 3, t.idx(1,1) = 4, t.idx(1,2) = 0;
    t.idx(2,0) = 0, t.idx(2,1) = 0, t.idx(2,2) = 1;
    t.values().resize(5);
    t.values()[0] = zero, t.values()[1] = one, t.values()[2]= two, t.values()[3] = three, t.values()[4]=four;
    thrust::host_vector<double> inout0=eight, inout1=nine, inout2=two, work0(inout0), work1(inout1), work2(inout2);
    thrust::host_vector<double> mu(five), nu;
    std::cout << "Begin T\n"; print(t);
    dg::tensor::scal(t,mu);
    std::cout<< "Scale with 5 \n";print(t);
    dg::tensor::scal(t,1);
    std::cout << "Scale with empty element \n";print(t);
    dg::tensor::scal(t,1./5.);
    std::cout << "Scale with 1/5 \n";print(t);
    dg::tensor::scal(t,two);
    std::cout << "Scale with container(2) \n";print(t);
    std::cout << "explicit dense Tensor \n";
    dg::SparseTensor<thrust::host_vector<double> > dense3d = t;
    print( dense3d);

    std::cout << "Test Tensor multiplies \n";
    print(t);
    std::cout << "Multiply T with [8,9]\n";
    dg::tensor::multiply2d( t, eight, nine, work0, work1);
    std::cout << "Result         is ["<<work0[0]<<" "<<work1[0]<<"] ([86 120])\n";
    std::cout << "Scalar product 2d\n";
    inout0 = eight;
    dg::tensor::scalar_product2d( 1., 2., one, two, t, 2., eight, nine, 1., inout0);
    std::cout << "Result         is "<<inout0[0]<<" (1312)\n";
    std::cout << "Multiply T^{-1} with [86,120]\n";
    dg::tensor::inv_multiply2d(1., t, work0, work1, 0., work0, work1);
    std::cout << "Result         is ["<<work0[0]<<" "<<work1[0]<<"] ([8 9])\n";
    inout0=eight, inout1=nine, inout2=two;
    dg::tensor::multiply2d(1., t, inout0, inout1, 0., work0, inout1);
    std::cout << "Result inplace is ["<<work0[0]<<" "<<inout1[0]<<"] ([86 120])\n T is \n";
    t.idx(0,2) = 4; std::swap( t.idx(1,1), t.idx(2,1)); print(t);
    std::cout << "Multiply T with [8,9,2]\n";
    dg::tensor::multiply3d(t, eight, nine,two, work0, work1, work2);
    std::cout << "Result         is ["<<work0[0]<<" "<<work1[0]<<" "<<work2[0]<<"] ([102 48 76])\n";
    std::cout << "Scalar product 3d\n";
    inout0 = eight;
    dg::tensor::scalar_product3d( 1., 3., one, two,three, t, 3., 8.,9.,2., -100., inout0);
    std::cout << "Result         is "<<inout0[0]<<" (3034)\n";
    std::cout << "Multiply T^{-1} with [102,48,76]\n";
    dg::tensor::inv_multiply3d(1., t, work0, work1, work2, 0., work0, work1, work2);
    std::cout << "Result         is ["<<work0[0]<<" "<<work1[0]<<" "<<work2[0]<<"] ([8 9 2])\n";
    inout0=eight, inout1=nine, inout2=two;
    dg::tensor::multiply3d(1., t, inout0, inout1, inout2, 0., inout0, inout1, inout2);
    std::cout << "Result inplace is ["<<inout0[0]<<" "<<inout1[0]<<" "<<inout2[0]<<"] ([102 48 76])\n";
    std::cout << "Determinant3d of T: "<<dg::tensor::determinant(t)[0]<<" (312)\n";
    std::cout << "Determinant2d of T: "<<dg::tensor::determinant2d(t)[0]<<" (-36)\n";
    return 0;


}
