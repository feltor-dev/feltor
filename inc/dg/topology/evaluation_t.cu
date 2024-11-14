#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dg/blas.h"
#include "dg/functors.h"

#include "evaluation.h"
#include "weights.h"

template<class T>
T function(T x, T y)
{
    T rho = 0.20943951023931953; //pi/15
    T delta = 0.050000000000000003;
    if( y<= M_PI)
        return delta*cos(x) - 1./rho/cosh( (y-M_PI/2.)/rho)/cosh( (y-M_PI/2.)/rho);
    return delta*cos(x) + 1./rho/cosh( (3.*M_PI/2.-y)/rho)/cosh( (3.*M_PI/2.-y)/rho);
}
double function3d( double x, double y, double z)
{
    return exp(x)*exp(y)*exp(z);
}


int main()
{
    std::cout << "This program tests the exblas::dot function. The tests succeed only if the evaluation and grid functions but also the weights and especially the exblas::dot function are correctly implemented and compiled. Furthermore, the compiler implementation of the exp function in the math library must be consistent across platforms to get reproducible results\n";
    std::cout << "A TEST is PASSED if the number in the second column shows EXACTLY 0 \\pm 1!\n";
    unsigned n = 3, Nx = 12, Ny = 28, Nz = 100;
    std::cout << "On Grid "<<n<<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<"\n";

    dg::Grid1d g1d( 1, 2, n, 12);
    dg::Grid2d g2d( 0.0, 6.2831853071795862, 0.0, 6.2831853071795862, 3, 48, 48);
    //dg::Grid2d g2d( {0.0, 6.2831853071795862, 3, 48}, {0.0, 6.2831853071795862, 5, 28});
    dg::RealGrid<float,2> gf2d( 0.0, 6.2831853071795862, 0.0, 6.2831853071795862, 3, 48, 48);
    dg::Grid3d g3d( 1, 2, 3, 4, 5, 6, n, Nx, Ny, Nz,dg::PER,dg::PER,dg::PER);
    //dg::Grid3d g3d( {1, 2, n, Nx,},{ 3, 4, 7, Ny},{ 5, 6, 4, Nx});

    //test evaluation functions
    const dg::DVec func1d = dg::construct<dg::DVec>( dg::evaluate( exp, g1d));
    const dg::DVec func2d = dg::construct<dg::DVec>( dg::evaluate( function<double>, g2d));
    const dg::fDVec funcf2d = dg::construct<dg::fDVec>( dg::evaluate( function<float>, gf2d));
    const dg::DVec func3d = dg::construct<dg::DVec>( dg::evaluate( function3d, g3d));
    const dg::DVec w1d = dg::construct<dg::DVec>( dg::create::weights( g1d));
    const dg::DVec w2d = dg::construct<dg::DVec>( dg::create::weights( g2d));
    const dg::fDVec wf2d = dg::construct<dg::fDVec>( dg::create::weights( gf2d));
    const dg::DVec w3d = dg::construct<dg::DVec>( dg::create::weights( g3d));
    dg::exblas::udouble res;
    dg::exblas::ufloat resf;

    double integral = dg::blas1::dot( w1d, func1d); res.d = integral;
    std::cout << "1D integral               "<<std::setw(6)<<integral <<"\t" << res.i - 4616944842743393935  << "\n";
    double sol = (exp(2.) -exp(1));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol<<std::endl;
    std::cout << "Relative 1d error is      "<<(integral-sol)/sol<<"\n\n";

    unsigned size = dg::blas1::vdot( []DG_DEVICE(double x){ return 1u;}, func3d);
    std::cout << "Size of vector test       "<<size<<"\t"<<(int)size - (int)g3d.size()<<"\n\n";

    double integral2d = dg::blas1::dot( w2d, func2d); res.d = integral2d;
    std::cout << "2D integral               "<<std::setw(6)<<integral2d <<"\t" << res.i + 4823286950217646080<< "\n";
    double sol2d = 0;
    std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    std::cout << "2d error is               "<<(integral2d-sol2d)<<"\n\n";

    float integralf2d = dg::blas1::dot( wf2d, funcf2d); resf.f = integralf2d;
    std::cout << "2D integral (float)       "<<std::setw(6)<<integralf2d <<"\t" << resf.i - 913405508<<"\n";
    std::cout << "(Remark: in floating precision the function to integrate may already be different on different compilers)\n";
    float solf2d = 0;
    std::cout << "Correct integral is       "<<std::setw(6)<<solf2d<<std::endl;
    std::cout << "2d error (float)          "<<(integralf2d-solf2d)<<"\n\n";

    double integral3d = dg::blas1::dot( w3d, func3d); res.d = integral3d;
    std::cout << "3D integral               "<<std::setw(6)<<integral3d <<"\t" << res.i - 4675882723962622631<< "\n";
    double sol3d = (exp(2.)-exp(1))*(exp(4.)-exp(3))*(exp(6.)-exp(5));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol3d<<std::endl;
    std::cout << "Relative 3d error is      "<<(integral3d-sol3d)/sol3d<<"\n\n";

    double norm = dg::blas2::dot( func1d, w1d, func1d); res.d = norm;
    std::cout << "Square normalized 1D norm "<<std::setw(6)<<norm<<"\t" << res.i - 4627337306989890294 <<"\n";
    double solution = (exp(4.) -exp(2))/2.;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution<<std::endl;
    std::cout << "Relative 1d error is      "<<(norm-solution)/solution<<"\n\n";

    double norm2d = dg::blas2::dot( w2d, func2d); res.d = norm2d;
    std::cout << "Square normalized 2D norm "<<std::setw(6)<<norm2d<<"\t" << res.i - 4635333359953759707<<"\n";
    double solution2d = 80.0489;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution2d<<std::endl;
    std::cout << "Relative 2d error is      "<<(norm2d-solution2d)/solution2d<<"\n\n";

    double norm3d = dg::blas2::dot( func3d, w3d, func3d); res.d = norm3d;
    std::cout << "Square normalized 3D norm "<<std::setw(6)<<norm3d<<"\t" << res.i - 4746764681002108278<<"\n";
    double solution3d = (exp(4.)-exp(2))/2.*(exp(8.)-exp(6.))/2.*(exp(12.)-exp(10))/2.;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution3d<<std::endl;
    std::cout << "Relative 3d error is      "<<(norm3d-solution3d)/solution3d<<"\n\n";

    std::cout << "TEST result of a sin and exp function to compare compiler specific math libraries:\n";
    dg::DVec x(100, 6.12610567450009658);
    dg::blas1::transform( x, x, [] DG_DEVICE ( double x){ return sin(x);} );
    res.d = x[0];
    std::cout << "Result of sin:    "<<res.i<<"\n"
              << "          GCC:    -4628567870976535683 (correct)"<<std::endl;
    dg::DVec y(1, 5.9126151457310376);
    dg::blas1::transform( y, y,[] DG_DEVICE ( double x){ return exp(x);} );
    res.d = y[0];
    std::cout << "Result of exp:     "<<res.i<<"\n"
              << "          GCC:     4645210948416067678 (correct)"<<std::endl;

    //TEST OF INTEGRAL
    dg::HVec integral_num = dg::integrate( cos, g1d, dg::forward);
    dg::HVec integral_ana = dg::evaluate( sin, g1d);
    dg::blas1::plus( integral_ana, -sin(g1d.x0()));
    dg::blas1::axpby( 1., integral_ana, -1., integral_num);
    norm = dg::blas2::dot( integral_num, dg::create::weights( g1d), integral_num);
    std::cout << " Error norm of  1d integral function (forward) "<<norm<<"\n";
    integral_num = dg::integrate( cos, g1d, dg::backward);
    integral_ana = dg::evaluate( sin, g1d);
    dg::blas1::plus( integral_ana, -sin(g1d.x1()));
    dg::blas1::axpby( 1., integral_ana, -1., integral_num);
    norm = dg::blas2::dot( integral_num, dg::create::weights( g1d), integral_num);
    std::cout << " Error norm of  1d integral function (backward) "<<norm<<"\n";
    // TEST if dot throws on NaN
    std::cout << "TEST if dot throws on Inf or Nan:\n";
    dg::blas1::transform( x,x, dg::LN<double>());
    bool hasnan = dg::blas1::reduce( x, false,
            thrust::logical_or<bool>(), dg::ISNFINITE<double>());
    std::cout << "x contains Inf or Nan numbers "<<std::boolalpha<<hasnan<<"\n";
    try{
        dg::blas1::dot( x,x);
    }catch ( std::exception& e)
    {
        std::cerr << "Error thrown as expected\n";
        //std::cerr << e.what() << std::endl;
    }
    std::cout << "Test MinMod function:\n";
    dg::MinMod minmod;
    std::cout << " 3 -5 ="<<minmod( 3,-5)<< " (0) "<<std::endl;
    std::cout << " 2 4 1 ="<<minmod( 2,4,1)<< " (1) "<<std::endl;
    std::cout << " 0 1 2 ="<<minmod( 0,1,2)<< " (0) "<<std::endl;
    std::cout << " -1 1 2 ="<<minmod( -1,1,2)<< " (0) "<<std::endl;
    std::cout << " -5 -3 -2 ="<<minmod( -5,-3,-2)<< " (-2) "<<std::endl;
    std::cout << "Test accuracy Dense Matrix\n";
    // massage a scalar product into dg::blas2::symv
    const dg::HVec func_h = dg::evaluate( function<double>, g2d);
    const dg::HVec w_h = dg::create::weights( g2d);
    std::vector<dg::DVec> matrix( func_h.size());
    for( unsigned i=0; i<func_h.size(); i++)
        matrix[i] = dg::DVec( 2, func_h[i]);
    dg::DVec integral_d( 2);
    dg::blas2::symv( 1., dg::asDenseMatrix( dg::asPointers( matrix)), w_h,
            0., integral_d);
    res.d = integral_d[0];
    std::cout << "2D integral               "<<std::setw(6)<<res.d <<"\t" << res.i + 4823491540355645440 << "\n";
    std::cout << "(We do not expect this to be correct because the Matrix-Vector product is not accurate nor binary reproducible)!\n";
    std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    std::cout << "2d error is               "<<(res.d-sol2d)<<"\n\n";

    std::cout << "COMPLEX SCALAR PRODUCTS\n";
    thrust::device_vector<thrust::complex<double>> cc3d( func3d.size());
    dg::blas1::transform( func3d, cc3d, []DG_DEVICE(double x){ return thrust::complex<double>{x,x};});
    thrust::complex<double> cintegral = dg::blas1::dot( w3d, cc3d);
    res.d =cintegral.real();
    std::cout << "3D integral (real)        "<<std::setw(6)<<cintegral.real() <<"\t" << res.i - 4675882723962622631<< "\n";
    res.d =cintegral.imag();
    std::cout << "3D integral (imag)        "<<std::setw(6)<<cintegral.imag() <<"\t" << res.i - 4675882723962622631<< "\n";
    sol2d = 0;
    std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    std::cout << "3d error is               "<<(cintegral.real()-sol2d)<<"\n\n";

    thrust::device_vector<thrust::complex<double>> cc1d( func1d.size());
    dg::blas1::transform( func1d, cc1d, []DG_DEVICE(double x){ return thrust::complex<double>{x,x};});
    cintegral = dg::blas1::dot( w1d, cc1d);
    res.d =cintegral.real();
    std::cout << "1D integral (real)        "<<std::setw(6)<<cintegral.real() <<"\t" << res.i - 4616944842743393935 << "\n";
    res.d =cintegral.imag();
    std::cout << "1D integral (imag)        "<<std::setw(6)<<cintegral.imag() <<"\t" << res.i - 4616944842743393935 << "\n";
    res.d = integral;
    sol = (exp(2.) -exp(1));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol<<std::endl;
    std::cout << "Relative 1d error is      "<<(cintegral.real()-sol)/sol<<"\n\n";
    std::cout << "Vector valued SCALAR PRODUCTS\n";
    std::vector<thrust::device_vector<thrust::complex<double>>> vx( 4, cc1d);
    std::vector<thrust::device_vector<double>> vw1d( 4, w1d);
    cintegral = dg::blas1::dot( vw1d, vx);
    sol = 4*(exp(2.) -exp(1));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol<<std::endl;
    std::cout << "Relative 1d error is      "<<(cintegral.real()-sol)/sol<<"\n\n";

    std::cout << "\nFINISHED! Continue with topology/derivatives_t.cu !\n\n";
    return 0;
}
