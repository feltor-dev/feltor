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
    std::cout << "This program tests the blas1 functions up to binary reproducibility with the exception of the dot function, which is tested in the dg/topology/evaluation_t program\n";
    std::cout << "A TEST IS PASSED IF THE RESULT IS ZERO.\n";
    //Vector v1( 5, 2.0002), v2( 5, 3.00003), v3(5,5.0005), v4(5,4.00004), v5(v4);
    //Vector v1( {2,2.0002}), v2({3,3.00003}), v3({5,5.0005}), v4({4,4.00004}), v5(v4); //std::array
    thrust::device_vector<double> v1p( 500, 2.0002), v2p( 500, 3.00003), v3p(500,5.0005), v4p(500,4.00004);
    Vector v1(v1p), v2(v2p), v3(v3p), v4(v4p), v5(v4p);
    dg::exblas::udouble ud;
    v3[0] = 1./0.; //we test here if nan breaks code
    dg::blas1::copy( v3p, v3); ud.d = v3[0];
    std::cout << "copy (x=x)            ";
    if ( !std::isfinite( v3[0]))
        std::cerr << "Error: Result has NaN!\n";
    else
        std::cout <<ud.i-4617316080911554445<<std::endl;
    dg::blas1::scal( v3, 3e-10); ud.d = v3[0];
    std::cout << "scal (x=ax)           "<<ud.i-4474825110624711575<<std::endl;
    dg::blas1::plus( v3, 3e-10); ud.d = v3[0];
    std::cout << "plus (x=x+a)          "<<ud.i-4476275821608249130<<std::endl;
    dg::blas1::axpby( 3e+10, v3, 1 , v4); ud.d = v4[0];
    std::cout << "fma (y=ax+y)          "<<ud.i-4633360230582305548<<std::endl;
    dg::blas1::axpby( 3e-10, v1, -2e-10 , v2); ud.d = v2[0];
    std::cout << "axpby (y=ax+by)       "<<ud.i-4408573477492505937<<std::endl;
    v5[0] = 1./0.; //we test here if nan breaks code
    dg::blas1::axpby( 3e-10, v1, -2. , v2, v5); ud.d = v5[0];
    std::cout << "axpby (z=ax+by)       ";
    if ( !std::isfinite( v5[0]))
        std::cerr << "Error: Result has NaN!\n";
    else
        std::cout <<ud.i-4468869610430797025<<std::endl;
    dg::blas1::axpbypgz( 2.5, v1, 7.e+10, v2, -0.125, v3); ud.d = v3[0];
    std::cout << "axpbypgz (y=ax+by+gz) "<<ud.i-4617320336812948958<<std::endl;
    v3[0] = 1./0.; //we test here if nan breaks code
    dg::blas1::pointwiseDot( v1, v2, v3); ud.d = v3[0];
    std::cout << "pDot (z=xy)           ";
    if ( !std::isfinite( v3[0]))
        std::cerr << "Error: Result has NaN!\n";
    else
        std::cout <<ud.i-4413077932784031586<<std::endl;
    dg::blas1::pointwiseDot( 0.2, v1, v2, +0.4e10, v3); ud.d = v3[0];
    std::cout << "pDot ( z=axy+bz)      "<<ud.i-4556605413983777388<<std::endl;
    v5 = v4p;
    dg::blas1::pointwiseDot( -0.2, v1, v2, 0.4, v3, v4, 0.1, v5); ud.d = v5[0];
    std::cout << "pDot (z=axy+bfh+gz)   "<<ud.i-4601058031075598447<<std::endl;
    dg::blas1::pointwiseDot( 0.2, v1, v2,v4, 0.4, v3); ud.d = v3[0];
    std::cout << "pDot (z=awxy + bz)    "<<ud.i-4550507856334720009<<std::endl;
    v5[0] = 1./0.; //we test here if nan breaks code
    dg::blas1::pointwiseDivide( v1,v2,v5); ud.d = v5[0];
    std::cout << "pDivide (z=x/y)       ";
    if ( !std::isfinite( v5[0]))
        std::cerr << "Error: result has NaN!\n";
    else
        std::cout <<ud.i-4810082017219139146<<std::endl;
    dg::blas1::pointwiseDivide( 5.,v1,v2,-1,v3); ud.d = v3[0];
    std::cout << "pDivide (z=ax/y+bz)   "<<ud.i-4820274520177585116<<std::endl;
    v3[0] = 1./0.; //we test here if nan breaks code
    dg::blas1::transform( v1, v3, dg::EXP<>()); ud.d = v3[0];
    std::cout << "transform y=exp(x)    ";
    if ( !std::isfinite( v3[0]))
        std::cerr << "Error: result has NaN!\n";
    else
        std::cout <<ud.i-4620007020034741378<<std::endl;;
    // host transform (checks if nvcc generates warnings, it should be suppressed)
    thrust::host_vector<double> v1h ( v1), v3h(v3);
    dg::blas1::transform( v1h, v3h, [](double x){return exp(x);});
    std::cout << "h_transform y=exp(x)    ";
    if ( !std::isfinite( v3[0]))
        std::cerr << "Error: result has NaN!\n";
    else
        std::cout <<ud.i-4620007020034741378<<std::endl;;
    }

    //v1 = 2, v2 = 3

    std::cout << "Human readable test RecursiveVector (passed if ouput equals value in brackets) \n";
    Vector v1( 5, 2.), v2( 5, 3.), v3(5,5.), v4(5,4.), v5(v4);
    std::array<Vector, 2> w1( dg::construct<std::array<Vector,2>>(v1)), w2({v2,v2}), w3({v3,v3}), w4({v4,v4});
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
    std::cout << "2*2+ 3*(-12) = " << w2[0][0] <<" (-32)\n";
    dg::blas1::axpby( 2.5, w1, 0., w2);
    std::cout << "2.5*2+ 0 = " << w2[0][0] <<" (5)\n";
    dg::blas1::axpbypgz( 2.5, w1, 2., w2, -0.125, w3);
    std::cout << "2.5*2+ 2.*5-0.125*96 = " << w3[0][0] <<" (3)\n";
    dg::blas1::pointwiseDivide( 5.,w1,5.,-1,w3);
    std::cout << "5*2/5-1*3 = " << w3[0][0] <<" (-1)\n";
    dg::blas1::pointwiseDivide( w1,5.,w3);
    std::cout << "2/5 = " << w3[0][0] <<" (0.4)\n";
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

    std::cout << "Human readable test RecursiveRecursiveVector (passed if ouput equals value in brackets) \n";
    w1 = dg::construct<std::array<Vector,2>>(v1), w2 = {v2,v2}, w3 = {v3,v3}, w4 = {v4,v4};
    std::array<std::array<Vector, 2>,2> w11( dg::construct<std::array<std::array<Vector,2>,2>>(v1)), w22({w2,w2}), w33({w3,w3}), w44({w4,w4});
    dg::blas1::axpby( 2., w11, 3., w22);
    std::cout << "2*2+ 3*3 = " << w22[1][1][0] <<" (13)\n";
    std::cout << "\nFINISHED! Continue with topology/evaluation_t.cu !\n\n";

    return 0;

}

