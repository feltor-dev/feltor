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
            std::cout << t.value(i,j)[0]<<" ";
        }
        std::cout << "\n";
    }

}

thrust::host_vector<double> zero(0,10), one(1,10), two(1,20), three(1,30), four(1,40), five(1,50), six(1,60), seven(1,70), eight(1,80), nine(1,90);

int main()
{
    std::cout << "Test dg::Sparse Tensor class \n";
    dg::Grid2d grid( 0,1,0,1, 1, 10, 1); //grid with 10 elements
    dg::SparseTensor<thrust::host_vector<double> > dense2d(grid);
    dense2d.idx(0,0) = 0, dense2d.idx(0,1) = 1;
    dense2d.idx(1,0) = 1, dense2d.idx(1,1) = 2;
    dense2d.values().resize(3);
    dense2d.values()[0] = eight; dense2d.values()[1] = two; dense2d.values()[2] = nine;
    std::cout << "Dense 2d Tensor \n";
    print( dense2d);
    std::vector<thrust::host_vector<double> > values(4);
    values[0] = seven; values[1] = three; values[2] = nine; values[3] = one;
    dg::SparseTensor<thrust::host_vector<double> > sparse3d(zero);
    sparse3d.values() = values;
    sparse3d.idx(0,0) = 0, sparse3d.idx(0,1) = 1                       ;
    sparse3d.idx(1,0) = 1                       , sparse3d.idx(1,2) = 3;
    sparse3d.idx(2,0) = 1                       , sparse3d.idx(2,2) = 3;
    dg::SparseTensor<thrust::device_vector<double> > sparse3d_D=sparse3d;

    std::cout << "Sparse 3d Tensor \n";
    print( sparse3d);
    sparse3d = sparse3d_D;
    std::cout << "original stored on device \n";
    print( sparse3d);
    std::cout << "transpose \n";
    print( sparse3d.transpose());

    sparse3d.idx(2,0)=sparse3d.idx(0,2);
    sparse3d.idx(1,2)=sparse3d.idx(2,1);
    sparse3d.idx(1,1)=3;
    sparse3d.values()[3]=nine;

    std::cout <<std::flush;
    return 0;


}
