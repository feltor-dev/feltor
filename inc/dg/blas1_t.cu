//#define CUSP_DEVICE_BLAS_SYSTEM CUSP_DEVICE_BLAS_CUBLAS
#include <iostream>
#include <vector>
#include <array>

#include "blas1.h"
#include "functors.h"


//test program that (should ) call every blas1 function for every specialization

//using Vector = std::array<double,2>;
//using Vector = thrust::host_vector<double>;
using Vector = thrust::device_vector<double>;
//using Vector = cusp::array1d<double, cusp::device_memory>;
int main()
{
    {
    std::cout << "This program tests the blas1 functions up to binary reproducibility with the exception of the dot function, which is tested in the dg/geometry/evaluation_t program\n";
    std::cout << "A TEST IS PASSED IF THE RESULT IS ZERO.\n";
    //Vector v1( 5, 2.0002), v2( 5, 3.00003), v3(5,5.0005), v4(5,4.00004), v5(v4);
    //Vector v1( {2,2.0002}), v2({3,3.00003}), v3({5,5.0005}), v4({4,4.00004}), v5(v4); //std::array
    thrust::device_vector<double> v1p( 500, 2.0002), v2p( 500, 3.00003), v3p(500,5.0005), v4p(500,4.00004);
    Vector v1(v1p), v2(v2p), v3(v3p), v4(v4p), v5(v4p);
    exblas::udouble ud;
    dg::blas1::scal( v3, 3e-10); ud.d = v3[0];
    std::cout << "scal (x=ax)           "<<ud.i-4474825110624711575<<std::endl;
    dg::blas1::plus( v3, 3e-10); ud.d = v3[0];
    std::cout << "plus (x=x+a)          "<<ud.i-4476275821608249130<<std::endl;
    dg::blas1::axpby( 3e+10, v3, 1 , v4); ud.d = v4[0];
    std::cout << "fma (y=ax+y)          "<<ud.i-4633360230582305548<<std::endl;
    dg::blas1::axpby( 3e-10, v1, -2e-10 , v2); ud.d = v2[0];
    std::cout << "axpby (y=ax+by)       "<<ud.i-4408573477492505937<<std::endl;
    dg::blas1::axpbypgz( 2.5, v1, 7.e+10, v2, -0.125, v3); ud.d = v3[0];
    std::cout << "axpbypgz (y=ax+by+gz) "<<ud.i-4617320336812948958<<std::endl;
    dg::blas1::pointwiseDot( v1, v2, v3); ud.d = v3[0];
    std::cout << "pDot (z=xy)           "<<ud.i-4413077932784031586<<std::endl;
    dg::blas1::pointwiseDot( 0.2, v1, v2, +0.4e10, v3); ud.d = v3[0];
    std::cout << "pDot ( z=axy+bz)      "<<ud.i-4556605413983777388<<std::endl;
    dg::blas1::pointwiseDot( -0.2, v1, v2, 0.4, v3, v4, 0.1, v5); ud.d = v5[0];
    std::cout << "pDot (z=axy+bfh+gz)   "<<ud.i-4601058031075598447<<std::endl;
    dg::blas1::pointwiseDot( 0.2, v1, v2,v4, 0.4, v3); ud.d = v3[0];
    std::cout << "pDot (z=awxy + bz)    "<<ud.i-4550507856334720009<<std::endl;
    dg::blas1::pointwiseDivide( 5.,v1,v2,-1,v3); ud.d = v3[0];
    std::cout << "pDivide (z=ax/y+bz)   "<<ud.i-4820274520177585116<<std::endl;
    dg::blas1::transform( v1, v3, dg::EXP<>()); ud.d = v3[0];
    std::cout << "transform y=exp(x)    "<<ud.i-4620007020034741378<<std::endl;
    }

    //v1 = 2, v2 = 3

    std::cout << "Human readable test RecursiveVector (passed if ouput equals value in brackets) \n";
    Vector v1( 5, 2.), v2( 5, 3.), v3(5,5.), v4(5,4.), v5(v4);
    std::array<Vector, 2> w1( dg::transfer<std::array<Vector,2>>(v1)), w2({v2,v2}), w3({v3,v3}), w4({v4,v4});
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
    dg::blas1::pointwiseDot( 2., w1, w2, -4., w1, w2, 0., w2);
    std::cout << "2*2*3 -4*2*3 = "<<w2[0][0]<<" (-12)\n";
    dg::blas1::axpby( 2., w1, 3., w2);
    std::cout << "2*2+ 3*3 = " << w2[0][0] <<" (13)\n";
    dg::blas1::axpby( 2.5, w1, 0., w2);
    std::cout << "2.5*2+ 0 = " << w2[0][0] <<" (5)\n";
    dg::blas1::axpbypgz( 2.5, w1, 2., w2, -0.125, w3);
    std::cout << "2.5*2+ 2.*5-0.125*96 = " << w3[0][0] <<" (3)\n";
    dg::blas1::pointwiseDivide( 5.,w1,5.,-1,w3);
    std::cout << "5*2/5-1*3 = " << w3[0][0] <<" (-1)\n";
    dg::blas1::copy( w2, w1);
    std::cout << "5 = " << w1[0][0] <<" (5)"<< std::endl;
    dg::blas1::scal( w1, 0.4);
    std::cout << "5*0.5 = " << w1[0][0] <<" (2)"<< std::endl;
    dg::blas1::evaluate( w4, dg::equals(),dg::AbsMax<>(), w1, w2);
    std::cout << "absMax( 2, 5) = " << w4[0][0] <<" (5)"<< std::endl;
    dg::blas1::transform( w1, w3, dg::EXP<>());
    std::cout << "e^2 = " << w3[0][0] <<" (7.389056...)"<< std::endl;
    dg::blas1::scal( w2, 0.6);
    dg::blas1::plus( w3, -7.0);
    std::cout << "e^2-7 = " << w3[0][0] <<" (0.389056...)"<< std::endl;
    std::cout << "\nFINISHED! Continue with geometry/evaluation_t.cu !\n\n";

    std::cout << "Human readable test mixed vector classes (passed if ouput equals value in brackets) \n";
    double x1 = 2., x2 = 3., x3 = 4.;
    std::vector<double> vv3(3, x3);
    dg::blas1::axpby( 2., x1, 3., x2);
    dg::blas1::axpby( 2., vv3, 3., vv3);
    std::array<double, 3>  arr1{2,3,4}, arr2(arr1);
    dg::blas1::axpby( 2., arr1, 3., arr2);
    dg::blas1::axpby( 2., vv3, 3., arr2);
    dg::blas1::copy( 1., vv3);
    dg::blas1::dot( vv3, vv3);
    dg::blas1::dot(arr1, arr1);


    return 0;

}

