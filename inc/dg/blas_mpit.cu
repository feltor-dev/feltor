#include <iostream>
#include <iomanip>
#include <mpi.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "blas.h"

struct Expression{
   DG_DEVICE
   void operator() ( double& v, double w, double param){
       v = param*v*v + w;
   }
};

int main( int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    MPI_Comm comm = MPI_COMM_WORLD;
    int rank,size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    //mpi_init2d( dg::PER, dg::PER, comm);
    if(rank==0)std::cout << "This program tests the many different possibilities to call blas1 and blas2 functions.\n";
    if(rank==0)std::cout << "Test passed? \n";
    double x1 = 1., x2 = 2., x3 = 3.;
    std::vector<double> vec1(3, x3), vec2;
    vec1[0] = 10., vec1[1] = 20., vec1[2] = 30;
    vec2 = vec1;
    std::array<double, 3>  arr1{2,3,4}, arr2(arr1);
    dg::MHVec hvec1 ( dg::HVec(vec1.begin(), vec1.end()), comm);
    dg::MDVec dvec1 ( dg::DVec(vec1.begin(), vec1.end()), comm);
    std::vector<dg::MDVec  > arrdvec1( 3, dvec1);

    if(rank==0)std::cout << "Test trivial parallel functions:\n"<<std::boolalpha;
    dg::blas1::axpby( 2., x1, 3., x2);
    if(rank==0)std::cout << "Scalar addition                   "<< (x2 == 8.)<<std::endl;
    if(rank==0)std::cout << "Reduction                         " << (dg::blas1::reduce( arr1, 0, thrust::maximum<float>()) == 4) <<std::endl;
    dg::blas1::axpby( 2., arr1, 3., vec2);
    if(rank==0)std::cout << "Recursive Vec Scalar addition     "<< (vec2[0] == 34.)<<std::endl;
    dg::blas1::axpby( 2., vec1, 3., arr2);
    if(rank==0)std::cout << "Recursive Arr Scalar addition     "<< (arr2[0] == 26.)<<std::endl;
    double max = dg::blas1::reduce( dvec1, 0, thrust::maximum<double>());
    if(rank==0)std::cout << "Recursive DVec reduction          " << (max == 30) <<std::endl;
    dg::blas1::copy( 2., arrdvec1);
    if(rank==0)std::cout << "Recursive DVec Copy Scalar to     "<< (arrdvec1[0].data()[0] == 2 && arrdvec1[1].data()[0]==2)<<std::endl;
    dg::blas1::axpby( 2., vec1 , 3, arrdvec1);
    if(rank==0)std::cout << "Recursive Scalar/Vetor addition   "<< (arrdvec1[0].data()[0] == 26 && arrdvec1[1].data()[0]==46.)<<std::endl;
    // test the examples in the documentation
    dg::blas1::plus( dvec1, 1);
    std::array<dg::MDVec, 3> array_v{ dvec1, dvec1, dvec1}, array_w(array_v);
    std::array<double, 3> array_p{ 1,2,3};
    if(rank==0)std::cout << dvec1.data()[0]<< " "<<array_w[2].data()[0]<<"\n";
    dg::blas1::subroutine( Expression(), dvec1, array_w[2], 3);
    if(rank==0)std::cout << "Example in documentation          "<< (dvec1.data()[0] ==374)<<std::endl;
    dg::blas1::subroutine( Expression(), array_v, array_w, array_p);
    if(rank==0)std::cout << "Example in documentation      	  "<< (array_v[0].data()[0] == 132 && array_v[1].data()[1] == 903)<<std::endl;
    if(rank==0)std::cout << "Test DOT functions:\n"<<std::boolalpha;
    double result = dg::blas1::dot( 1., array_p);
    if(rank==0)std::cout << "blas1 dot recursive Scalar          "<< (result == 6) <<"\n";
    result = dg::blas1::dot( 1., arrdvec1 )/(double)size;
    if(rank==0)std::cout << "blas1 dot recursive Scalar Vector   "<< (result == 414) <<"\n";
    result = dg::blas2::dot( 1., 4., arrdvec1 )/(double)size;
    if(rank==0)std::cout << "blas2 dot recursive Scalar Vector   "<< (result == 1656) <<"\n";
    result = dg::blas2::dot( 1., arrdvec1, arrdvec1 )/(double)size;
    double result1 = dg::blas1::dot( arrdvec1, arrdvec1)/(double)size;
    if(rank==0)std::cout << "blas1/2 dot recursive Scalar Vector "<< (result == result1) <<"\n";
    if(rank==0)std::cout << "Test SYMV functions:\n";
    dg::blas2::symv( 2., arrdvec1, arrdvec1);
    if(rank==0)std::cout << "symv Scalar times Vector          "<<( arrdvec1[0].data()[0] == 52) << std::endl;
    std::array<std::vector<dg::MDVec>,1> recursive{ arrdvec1};
    dg::blas2::symv( arrdvec1[0], recursive, recursive);
    if(rank==0)std::cout << "symv deep Recursion               "<<( recursive[0][0].data()[0] == 52*52) << std::endl;
    dg::blas2::symv( 0.5, dg::asDenseMatrix(dg::asPointers(arrdvec1)),
            std::array<double,3>({0.1, 10, 1000}), 0.001, dvec1);
    if(rank==0)std::cout << "symv as DenseMatrix               "<<( dvec1.data()[0] == 66462.974) << std::endl;
    MPI_Finalize();

    return 0;
}
