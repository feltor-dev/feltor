//#define CUSP_DEVICE_BLAS_SYSTEM CUSP_DEVICE_BLAS_CUBLAS
#include <iostream>
#include <vector>
#include <array>

#include "blas1.h"
#include "backend/blas1_array.h"

struct EXP{ __host__ __device__ double operator()(double x){return exp(x);}};


//test program that (should ) call every blas1 function for every specialization

//typedef thrust::device_vector<double>  Vector;
typedef cusp::array1d<double, cusp::device_memory>  Vector;
int main()
{
    Vector v1( 5, 2), v2( 5, 3), v3(5), v4(5,4);
    double temp = dg::blas1::dot(v1,v2);
    std::cout << "5*(2*3) = "<<temp << " (30)\n"; 
    dg::blas1::axpby( 2., v1, 3., v2, v3);
    std::cout << "2*2+ 3*3 = " << v3[0] <<" (13)\n";
    dg::blas1::axpby( 0., v1, 3., v2, v3);
    std::cout << "0*2+ 3*3 = " << v3[0] <<" (9)\n";
    dg::blas1::axpby( 2., v1, 0., v2, v3);
    std::cout << "2*2+ 0*3 = " << v3[0] <<" (4)\n";
    dg::blas1::pointwiseDot( v1, v2, v3);
    std::cout << "2*3 = "<<v3[0]<<" (6)\n";
    dg::blas1::pointwiseDot( 2., v1, v2, -4., v3);
    std::cout << "2*2*3 -4*6 = "<<v3[0]<<" (-12)\n";
    dg::blas1::pointwiseDot( 2., v1, v2,v4, -4., v3);
    std::cout << "2*2*3*4 -4*(-12) = "<<v3[0]<<" (96)\n";
    dg::blas1::axpby( 2., v1, 3., v2);
    std::cout << "2*2+ 3*3 = " << v2[0] <<" (13)\n";
    dg::blas1::axpby( 2.5, v1, 0., v2);
    std::cout << "2.5*2+ 0 = " << v2[0] <<" (5)\n";
    dg::blas1::axpbypgz( 2.5, v1, 2., v2, 3., v3);
    std::cout << "2.5*2+ 2.*5-3*12 = " << v3[0] <<" (-21)\n";
    dg::blas1::pointwiseDivide( 5.,v1,v2,-1,v3);
    std::cout << "5*2/5-1*21 = " << v3[0] <<" (-19)\n";
    dg::blas1::copy( v2, v1);
    std::cout << "5 = " << v1[0] <<" (5)"<< std::endl;
    dg::blas1::scal( v1, 0.4);
    std::cout << "5*0.4 = " << v1[0] <<" (2)"<< std::endl;
    dg::blas1::transform( v1, v3, EXP());
    std::cout << "e^2 = " << v3[0] <<" (7.389056...)"<< std::endl;
    dg::blas1::scal( v2, 0.6);
    dg::blas1::plus( v3, -7.0);
    std::cout << "e^2-7 = " << v3[0] <<" (0.389056...)"<< std::endl;

    //v1 = 2, v2 = 3

    std::cout << "Test std::array \n";
    std::array<Vector, 2> w1( {v1,v1}), w2({v2,v2}), w3({v3,v3}), w4({v4,v4});
    temp = dg::blas1::dot( w1, w2);
    std::cout << "2*5*(2*3) = "<<temp << " (60)\n"; 
    dg::blas1::axpby( 2., w1, 3., w2, w3);
    std::cout << "2*2+ 3*3 = " << w3[0][0] <<" (13)\n";
    dg::blas1::axpby( 0., w1, 3., w2, w3);
    std::cout << "0*2+ 3*3 = " << w3[0][0] <<" (9)\n";
    dg::blas1::axpby( 2., w1, 0., w2, w3);
    std::cout << "2*2+ 0*3 = " << w3[0][0] <<" (4)\n";
    dg::blas1::pointwiseDot( w1, w2, w3);
    std::cout << "2*3 = "<<w3[0][0]<<" (6)\n";
    dg::blas1::pointwiseDot( 2., w1, w2, -4., w3);
    std::cout << "2*2*3 -4*6 = "<<w3[0][0]<<" (-12)\n";
    dg::blas1::pointwiseDot( 2., w1, w2,w4, -4., w3);
    std::cout << "2*2*3*4 -4*(-12) = "<<w3[0][0]<<" (96)\n";
    dg::blas1::pointwiseDot( 2., w1[0], w2[0], -4., v1, v2, 0., v2);
    std::cout << "2*2*3 -4*2*3 = "<<v2[0]<<" (-12)\n";
    dg::blas1::axpby( 2., w1, 3., w2);
    std::cout << "2*2+ 3*3 = " << w2[0][0] <<" (13)\n";
    dg::blas1::axpby( 2.5, w1, 0., w2);
    std::cout << "2.5*2+ 0 = " << w2[0][0] <<" (5)\n";
    dg::blas1::copy( w2, w1);
    std::cout << "5 = " << w1[0][0] <<" (5)"<< std::endl;
    dg::blas1::scal( w1, 0.4);
    std::cout << "5*0.5 = " << w1[0][0] <<" (2)"<< std::endl;
    dg::blas1::transform( w1, w3, EXP());
    std::cout << "e^2 = " << w3[0][0] <<" (7.389056...)"<< std::endl;
    dg::blas1::scal( w2, 0.6);
    dg::blas1::plus( w3, -7.0);
    std::cout << "e^2-7 = " << w3[0][0] <<" (0.389056...)"<< std::endl;
    std::cout << "FINISHED\n\n";


    return 0;

}
