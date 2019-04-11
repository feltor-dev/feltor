#include <iostream>
#include <iomanip>
#include <cmath>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "dg/blas.h"

#include "evaluation.h"
#include "weights.h"

struct exp_function{
DG_DEVICE
double operator()( double x)
{
    return exp(x);
}
};
struct sin_function{
DG_DEVICE
double operator()( double x)
{
    return sin(x);
}
};

double function( double x, double y)
{
        return exp(x)*exp(y);
}
double function( double x, double y, double z)
{
        return exp(x)*exp(y)*exp(z);
}

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;

int main()
{
    std::cout << "This program tests the exblas::dot function. The tests succeed only if the evaluation and grid functions but also the weights and especially the exblas::dot function are correctly implemented and compiled. Furthermore, the compiler implementation of the exp function in the math library must be consistent across platforms to get reproducible results\n";
    std::cout << "A TEST is PASSED if the number in the second column shows EXACTLY 0!\n";
    unsigned n = 1, Nx = 12, Ny = 28, Nz = 100;
    std::cout << "On Grid "<<n<<" x "<<Nx<<" x "<<Ny<<" x "<<Nz<<"\n";

    dg::Grid1d g1d( 0, M_PI/2., n, Nx);
    dg::Grid2d g2d( 1, 2, 3, 4, n, Nx, Ny);
    dg::Grid3d g3d( 1, 2, 3, 4, 5, 6, n, Nx, Ny, Nz,dg::PER,dg::PER,dg::PER);

    //test evaluation functions
    const DVec func1d = dg::construct<DVec>( dg::evaluate( exp, g1d));
    const DVec func2d = dg::construct<DVec>( dg::evaluate( function, g2d));
    const DVec func3d = dg::construct<DVec>( dg::evaluate( function, g3d));
    const DVec w1d = dg::construct<DVec>( dg::create::weights( g1d));
    const DVec w2d = dg::construct<DVec>( dg::create::weights( g2d));
    const DVec w3d = dg::construct<DVec>( dg::create::weights( g3d));
    exblas::udouble res;

    double integral = dg::blas1::dot( w1d, func1d); res.d = integral;
    std::cout << "1D integral               "<<std::setw(6)<<integral <<"\t" << res.i - 4616944842743393935  << "\n";
    double sol = (exp(2.) -exp(1));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol<<std::endl;
    std::cout << "Absolute 1d error is      "<<(integral-sol)<<"\n\n";

    double integral2d = dg::blas1::dot( w2d, func2d); res.d = integral2d;
    std::cout << "2D integral               "<<std::setw(6)<<integral2d <<"\t" << res.i - 4639875759346476257<< "\n";
    double sol2d = (exp(2.)-exp(1))*(exp(4.)-exp(3));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol2d<<std::endl;
    std::cout << "Absolute 2d error is      "<<(integral2d-sol2d)<<"\n\n";

    double integral3d = dg::blas1::dot( w3d, func3d); res.d = integral3d;
    std::cout << "3D integral               "<<std::setw(6)<<integral3d <<"\t" << res.i - 4675882723962622631<< "\n";
    double sol3d = sol2d*(exp(6.)-exp(5.));
    std::cout << "Correct integral is       "<<std::setw(6)<<sol3d<<std::endl;
    std::cout << "Absolute 3d error is      "<<(integral3d-sol3d)<<"\n\n";

    double norm = dg::blas2::dot( func1d, w1d, func1d); res.d = norm;
    std::cout << "Square normalized 1D norm "<<std::setw(6)<<norm<<"\t" << res.i - 4627337306989890294 <<"\n";
    double solution = (exp(4.) -exp(2))/2.;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution<<std::endl;
    std::cout << "Relative 1d error is      "<<(norm-solution)/solution<<"\n\n";

    double norm2d = dg::blas2::dot( w2d, func2d); res.d = norm2d;
    std::cout << "Square normalized 2D norm "<<std::setw(6)<<norm2d<<"\t" << res.i - 4674091193523851724<<"\n";
    double solution2d = (exp(4.)-exp(2))/2.*(exp(8.) -exp(6.))/2.;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution2d<<std::endl;
    std::cout << "Relative 2d error is      "<<(norm2d-solution2d)/solution2d<<"\n\n";

    double norm3d = dg::blas2::dot( func3d, w3d, func3d); res.d = norm3d;
    std::cout << "Square normalized 3D norm "<<std::setw(6)<<norm3d<<"\t" << res.i - 4746764681002108278<<"\n";
    double solution3d = solution2d*(exp(12.) -exp(10.))/2.;
    std::cout << "Correct square norm is    "<<std::setw(6)<<solution3d<<std::endl;
    std::cout << "Relative 3d error is      "<<(norm3d-solution3d)/solution3d<<"\n\n";

    std::cout << "TEST result of a sin and exp function to compare compiler specific math libraries:\n";
    DVec x(1, 6.12610567450009658);
    dg::blas1::transform( x, x, sin_function() );
    res.d = x[0];
    std::cout << "Result of sin:    "<<res.i<<"\n"
              << "          GCC:    -4628567870976535683 (correct)"<<std::endl;
    DVec y(1, 5.9126151457310376);
    dg::blas1::transform( y, y, exp_function() );
    res.d = y[0];
    std::cout << "Result of exp:     "<<res.i<<"\n"
              << "          GCC:     4645210948416067678 (correct)"<<std::endl;

    //TEST OF INTEGRAL
    dg::HVec integral_num = dg::create::integral( dg::evaluate( cos, g1d), g1d);
    dg::HVec integral_ana = dg::evaluate( sin, g1d);
    dg::blas1::plus( integral_ana, -sin(g1d.x0()));
    dg::blas1::axpby( 1., integral_ana, -1., integral_num);
    norm = dg::blas2::dot( integral_num, dg::create::weights( g1d), integral_num);
    std::cout << " Error norm of  1d integral function "<<norm<<"\n";
    std::cout << "\nFINISHED! Continue with topology/derivatives_t.cu !\n\n";
    return 0;
}
