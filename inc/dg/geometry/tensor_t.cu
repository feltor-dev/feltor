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

thrust::host_vector<double> one(1,10), two(1,20), three(1,30), four(1,40), five(1,50), six(1,60), seven(1,70), eight(1,80), nine(1,90);

int main()
{
    std::cout << "Test dg::Sparse Tensor class \n";
    std::cout << "construct empty Tensor \n";
    dg::SparseTensor<thrust::host_vector<double> > empty;
    if( empty.isEmpty())print( empty);

    dg::SparseTensor<thrust::host_vector<double> > dense2d(3);
    dense2d.idx(0,0) = 0, dense2d.idx(0,1) = 1;
    dense2d.idx(1,0) = 1, dense2d.idx(1,1) = 2;
    dense2d.value(0) = eight; dense2d.value(1) = two; dense2d.value(2) = nine; 
    std::cout << "Dense 2d Tensor \n";
    if( !dense2d.isEmpty())print( dense2d);
    std::vector<thrust::host_vector<double> > values(4);
    values[0] = seven; values[1] = three; values[2] = nine; values[3] = one; 
    dg::SparseTensor<thrust::host_vector<double> > sparse3d(values);
    sparse3d.idx(0,0) = 0, sparse3d.idx(0,1) = 1                       ;
    sparse3d.idx(1,0) = 1                       , sparse3d.idx(1,2) = 3;
    sparse3d.idx(2,0) = 1                       , sparse3d.idx(2,2) = 3;
    dg::SparseTensor<thrust::device_vector<double> > sparse3d_D=sparse3d;

    std::cout << "Sparse 3d Tensor \n";
    print( sparse3d);
    std::cout << "unset an element \n";
    sparse3d.unset( 2,0);
    print( sparse3d);
    std::cout << "clear unused values \n";
    std::cout << "Size before "<<sparse3d.values().size()<<"\n";
    sparse3d.clear_unused_values();
    std::cout << "Size after  "<<sparse3d.values().size()<<"\n";
    std::cout << "tensor after \n";
    if(!sparse3d.isDiagonal()) print( sparse3d);
    std::cout << "empty Tensor \n";
    if(sparse3d.empty().isDiagonal()) print( sparse3d.empty());
    sparse3d = sparse3d_D;
    std::cout << "original stored on device \n";
    if(!sparse3d.isDense())print( sparse3d);
    std::cout << "perp \n";
    if( sparse3d.perp().isPerp()) print( sparse3d.perp());
    std::cout << "transpose \n";
    print( sparse3d.transpose());

    sparse3d.idx(2,0)=sparse3d.idx(0,2);
    sparse3d.idx(1,2)=sparse3d.idx(2,1);
    sparse3d.idx(1,1)=3;
    sparse3d.value(3)=nine;
    std::cout<<"Make a LDL^T decomposition\n";
    dg::CholeskyTensor<thrust::host_vector<double> > ch(sparse3d);
    std::cout << "origin\n"; print(sparse3d);
    std::cout << "lower \n"; print(ch.lower());
    std::cout << "diag  \n"; print(ch.diagonal());
    std::cout << "upper \n"; print(ch.upper());



    std::cout<< "Test dg::SparseElement";
    dg::SparseElement<thrust::host_vector<double> > e(eight);
    dg::SparseElement<thrust::device_vector<double> > ee;
    std::cout<<"\n construct: " <<e<<" "<<ee<<"\n";
    ee = e;
    e.value()=nine;
    std::cout << "Assignment and set : "<<ee<<" (80) "<<e<<"(90)\n";
    e.clear();
    std::cout<<"clear(): "<<e<<"\n";
    std::cout <<std::flush;
    return 0;


}
