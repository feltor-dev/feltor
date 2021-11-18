#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "blas.h"

struct Expression{
   DG_DEVICE
   void operator() ( double& v, double w, double param){
       v = param*v*v + w;
   }
};

struct Functor{
    template<class T>
    void operator()( const T& in, T&  out){
        dg::blas1::axpby( 1., in, 0., out);
    }
};

struct NoFunctor{

};

int main()
{
    std::cout << "This program tests the many different possibilities to call blas1 and blas2 functions.\n";
    std::cout << "Test passed? \n";
    double x1 = 1., x2 = 2., x3 = 3.;
    std::vector<double> vec1(3, x3), vec2;
    vec1[0] = 10., vec1[1] = 20., vec1[2] = 30;
    vec2 = vec1;
    std::array<float, 3>  arr1{2,3,4}, arr2(arr1);
    dg::HVec  hvec1 ( vec1.begin(), vec1.end());
    dg::DVec  dvec1 ( vec1.begin(), vec1.end());
    std::vector<dg::DVec  > arrdvec1( 3, dvec1);

    std::cout << "Test trivial parallel functions:\n"<<std::boolalpha;
    dg::blas1::axpby( 2., x1, 3., x2);
    std::cout << "Scalar addition                   "<< (x2 == 8.)<<std::endl;
    std::cout << "Reduction                         " << (dg::blas1::reduce( arr1, 0, thrust::maximum<float>()) == 4) <<std::endl;
    dg::blas1::axpby( 2., arr1, 3., vec2);
    std::cout << "Recursive Vec Scalar addition     "<< (vec2[0] == 34.)<<std::endl;
    dg::blas1::axpby( 2., vec1, 3., arr2);
    std::cout << "Recursive Arr Scalar addition     "<< (arr2[0] == 26.)<<std::endl;
    std::cout << "Recursive DVec reduction          " << (dg::blas1::reduce( dvec1, 0, thrust::maximum<double>()) == 30) <<std::endl;
    dg::blas1::copy( 2., arrdvec1);
    std::cout << "Recursive DVec Copy Scalar to     "<< (arrdvec1[0][0] == 2 && arrdvec1[1][0]==2)<<std::endl;
    dg::blas1::axpby( 2., vec1 , 3, arrdvec1);
    std::cout << "Recursive Scalar/Vetor addition   "<< (arrdvec1[0][0] == 26 && arrdvec1[1][0]==46.)<<std::endl;
    // test the examples in the documentation
    // dg::blas1::subroutine( []__host__ __device__(double& v){ v+=1.;}, dvec1);
    dg::blas1::plus( dvec1, 1);
    std::array<dg::DVec, 3> array_v{ dvec1, dvec1, dvec1}, array_w(array_v);
    std::array<double, 3> array_p{ 1,2,3};
    std::cout << dvec1[0]<< " "<<array_w[2][0]<<"\n";
    dg::blas1::subroutine( Expression(), dvec1, array_w[2], 3);
    std::cout << "Example in documentation          "<< (dvec1[0] ==374)<<std::endl;
    dg::blas1::subroutine( Expression(), array_v, array_w, array_p);
    std::cout << "Example in documentation          "<< (array_v[0][0] == 132 && array_v[1][1] == 903)<<std::endl;
    std::cout << "Test DOT functions:\n"<<std::boolalpha;
    double result = dg::blas1::dot( 1., array_p);
    std::cout << "blas1 dot recursive Scalar          "<< (result == 6) <<"\n";
    result = dg::blas1::dot( 1., arrdvec1 );
    std::cout << "blas1 dot recursive Scalar Vector   "<< (result == 414) <<"\n";
    result = dg::blas2::dot( 1., 4., arrdvec1 );
    std::cout << "blas2 dot recursive Scalar Vector   "<< (result == 1656) <<"\n";
    result = dg::blas2::dot( 1., arrdvec1, arrdvec1 );
    double result1 = dg::blas1::dot( arrdvec1, arrdvec1);
    std::cout << "blas1/2 dot recursive Scalar Vector "<< (result == result1) <<"\n";
    std::cout << "Test SYMV functions:\n";
    dg::blas2::symv( 2., arrdvec1, arrdvec1);
    std::cout << "symv Scalar times Vector          "<<( arrdvec1[0][0] == 52) << std::endl;
    std::array<std::vector<dg::DVec>,1> recursive{ arrdvec1};
    dg::blas2::symv( arrdvec1[0], recursive, recursive);
    std::cout << "symv deep Recursion               "<<( recursive[0][0][0] == 52*52) << std::endl;
    dg::blas2::symv( 0.5, dg::asDenseMatrix<dg::DVec>({&arrdvec1[0], &arrdvec1[1], &arrdvec1[2]}),
            std::array<double,3>({0.1, 10, 1000}), 0.001, dvec1);
    std::cout << "symv as DenseMatrix               "<<( dvec1[0] == 66462.974) << std::endl;
    Functor f;
    dg::blas2::symv( f, arrdvec1[0], dvec1);
    std::cout << "symv with functor "<< ( dvec1[0] == 52) << std::endl;
    //Check compiler error:
    //NoFunctor nof;
    //dg::blas2::symv( nof, arrdvec1[0], dvec1);

    return 0;
}
