#include <iostream>

#include "tensor.h"

void print( const dg::SparseTensor<thrust::host_vector<double> >& t)
{
    for( unsigned i=0; i<3; i++)
    {
        for( unsigned j=0; j<3; j++)
        {
            if(t.isSet(i,j)) std::cout << t.value(i,j)[0]<<" ";
            else std::cout <<"XX ";
        }
        std::cout << "\n";
    }

}
std::ostream& operator<<(std::ostream& os, const dg::SparseElement<thrust::host_vector<double> >& t)
{
    if(t.isSet()) os << t.value()[0]<<" ";
    else os <<"XX ";
    return os;
}

int main()
{
    thrust::host_vector<double> one(1,11), two(1,22), three(1,33), four(1,44), five(1,55), six(1,66), seven(1,77), eight(1,88), nine(1,99);

    dg::SparseTensor<thrust::host_vector<double> > dense2d(3);
    dense2d.idx(0,0) = 0, dense2d.idx(0,1) = 1;
    dense2d.idx(1,0) = 1, dense2d.idx(1,1) = 2;
    dense2d.value(0) = eight; dense2d.value(1) = two; dense2d.value(2) = nine; 
    dg::SparseTensor<thrust::host_vector<double> > sparse3d(4);
    sparse3d.idx(0,0) = 0, sparse3d.idx(0,1) = 1                       ;
    sparse3d.idx(1,0) = 1                       , sparse3d.idx(1,2) = 3;
    sparse3d.idx(2,0) = 1                       , sparse3d.idx(2,2) = 3;
    sparse3d.value(0) = seven; sparse3d.value(1) = three; sparse3d.value(2) = nine, sparse3d.value(3) = one; 

    dg::SparseTensor<thrust::host_vector<double> > empty;

    std::cout << "Test dg::Sparse Tensor class \n";
    std::cout << "Dense 2d Tensor \n";
    print( dense2d);
    std::cout << "dg::Sparse 3d Tensor \n";
    print( sparse3d);
    std::cout << "empty Tensor \n";
    print( empty);
    std::cout<< "Test dg::SparseElement";
    dg::SparseElement<thrust::host_vector<double> > e(eight);
    dg::SparseElement<thrust::host_vector<double> > ee;
    std::cout<<"\n construct: " <<e<<" "<<ee<<"\n";
    ee = e;
    e.set(nine);
    std::cout << "Assignment and set : "<<e<<" "<<ee<<"\n";
    dg::SparseElement<thrust::host_vector<double> > sqrt = e.sqrt();
    std::cout<<"\n sqrt(): "<<sqrt;
    sqrt = sqrt.invert();
    std::cout<<"\n invert(): "<<sqrt;
    sqrt.clear();
    std::cout<<"\n clear(): "<<sqrt;
    std::cout <<std::endl;
    return 0;


}
