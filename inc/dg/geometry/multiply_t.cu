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
        {
            if(t.isSet(i,j)) std::cout << t.value(i,j)[0]<<" ";
            else std::cout <<"xx ";
        }
        std::cout << "\n";
    }

}

template<class container>
std::ostream& operator<<(std::ostream& os, const dg::SparseElement<container >& t)
{
    if(t.isSet()) os << t.value()[0]<<" ";
    else os <<"XX ";
    return os;
}

const thrust::host_vector<double> one(1,1), two(1,2), three(1,3), four(1,4), five(1,5), six(1,6), seven(1,7), eight(1,8), nine(1,9), zero(1,0);

int main()
{
    std::cout << "Test dg::Sparse Tensor class \n";
    dg::SparseTensor<thrust::host_vector<double> > t(3);
    t.idx(0,0) = 0, t.idx(0,1)=t.idx(1,0) = 1;
    t.idx(1,1) = 2;
    t.value(0)= two, t.value(1) = three, t.value(2)=four;
    thrust::host_vector<double> inout0=eight, inout1=nine, inout2=two, work0(inout0), work1(inout1), work2(inout2);
    dg::SparseElement<thrust::host_vector<double> >mu(five), nu;
    std::cout << "Begin T\n"; print(t);
    dg::tensor::scal(t,mu); 
    std::cout<< "Scale with 5 \n";print(t);
    dg::tensor::scal(t,nu); 
    std::cout << "Scale with empty element \n";print(t);
    dg::tensor::invert(mu);
    dg::tensor::scal(t,mu); 
    std::cout << "Scale with 1/5 \n";print(t);
    dg::tensor::scal(t,two); 
    std::cout << "Scale with container(2) \n";print(t);
    std::cout << "explicit dense Tensor \n";
    dg::SparseTensor<thrust::host_vector<double> > dense3d = dg::tensor::dense(t);
    if(dense3d.isDense())print( dense3d);

    std::cout << "Test Element multiplies \n";
    dg::SparseElement<thrust::host_vector<double> > sqr(nine);
    dg::tensor::sqrt(sqr);
    std::cout<<"sqrt(): "<<sqr<<" ("<<std::sqrt(9)<<")\n";
    dg::tensor::invert(sqr);
    std::cout<<"invert(): "<<sqr<<"\n";
    dg::tensor::pointwiseDot(mu, eight, inout0); 
    std::cout<< "8*5 = "<<inout0[0]<<"\n";
    dg::tensor::pointwiseDivide(inout0,nu,inout0); 
    dg::tensor::pointwiseDivide(inout0,mu,inout0); 
    std::cout << "Restore 8 = "<<inout0[0]<<"\n";
    std::cout << "Test Tensor multiplies \n";
    print(t);
    std::cout << "Multiply T with [8,9]\n";
    dg::tensor::multiply2d(t, eight, nine, work0, work1);
    std::cout << "Result         is ["<<work0[0]<<" "<<work1[0]<<"] ([86 120])\n";
    dg::tensor::multiply2d(t, inout0, inout1, work0, inout1);
    std::cout << "Result inplace is ["<<work0[0]<<" "<<inout1[0]<<"] ([86 120])\n T is \n";
    t.idx(0,2) = 2; std::swap( t.idx(1,1), t.idx(2,1));  print(t);
    std::cout << "Multiply T with [8,9,2]\n";
    dg::tensor::multiply3d(t, eight, nine,two, work0, work1, work2);
    std::cout << "Result         is ["<<work0[0]<<" "<<work1[0]<<" "<<work2[0]<<"] ([102 57 76])\n";
    inout0=eight, inout1=nine, inout2=two;
    dg::tensor::multiply3d(t, inout0, inout1, inout2, work0, work1, inout2);
    std::cout << "Result inplace is ["<<work0[0]<<" "<<work1[0]<<" "<<inout2[0]<<"] ([102 57 76])\n";
    std::cout << "Determinant of T: "<<dg::tensor::determinant(t).value()[0]<<" (320)\n";
    std::cout << "Perp Determinant of T: "<<dg::tensor::determinant(t.perp()).value()[0]<<" (-32)\n";
    std::swap(t.idx(2,1), t.idx(2,0)); 
    t.value(0) = five;
    t.idx(1,1) = 0;
    t.idx(2,2) = 0;
    std::cout<<"Make a LDL^T decomposition\n";
    dg::CholeskyTensor<thrust::host_vector<double> > ch(t);
    std::cout << "origin\n"; print(t);
    std::cout << "lower \n"; print(ch.lower());
    std::cout << "diag  \n"; 
    if(ch.diagonal().isDiagonal())print(ch.diagonal());
    std::cout << "upper \n"; print(ch.upper());
    std::cout << "Multiply T with [8,9]\n";
    inout0=eight, inout1=nine, inout2=two;
    dg::tensor::multiply2d(ch, inout0, inout1, inout0, inout1);
    std::cout << "Result         is ["<<inout0[0]<<", "<<inout1[0]<<"] ([94, 93])\n";
    std::cout << "Multiply T with [8,9,2]\n";
    dg::tensor::multiply3d(ch, eight,nine,two, work0, work1, work2);
    std::cout << "Result         is ["<<work0[0]<<" "<<work1[0]<<" "<<work2[0]<<"] (110, 93, 74)\n";
    return 0;


}
