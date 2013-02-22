#include <iostream>
#include <cmath>


#include "integration.h"
#include "evaluation.h"
#include "rk.h"


#define P 4
#define K 3
const double Lx = 2.*M_PI;
const unsigned N = 40;

const double Time = 0.8; 
const unsigned NT = (unsigned)(Time*N*N/0.001/Lx/Lx); //courant condition  dt <= dx*dx/D


double sine( double x)
{
    return sin(x);
}
double solution( double x)
{
    return exp(-Time)*sin(x);
}
typedef std::vector<std::array<double, P>> Vector;
int main()
{
    double h = Lx/(double)N;
    double dt = Time/(double)NT;

    //generate y0 in l-space
    auto y0 = dg::evaluate< double(&)(double), P>( sine, 0., Lx, N);
    dg::Operator<double,P> forward( dg::DLT<P>::forward);
    for( unsigned i=0; i<N; i++)
        y0[i] = forward*y0[i];
    double norm_y0 = dg::BLAS2<dg::S, Vector>::ddot( y0, dg::S(h), y0);
    std::cout << "Normalized y0 (is Pi) " << norm_y0 << std::endl;

    //generate y1
    auto y1(y0);

    //generate runge right hand side and Runge-Kutta
    dg::RHS<P> rhs( h);
    dg::RK<K, dg::RHS<P>> rk( y0);

    for( unsigned i=0; i<NT; i++)
    {
        rk( rhs, y0, y1, dt);
        y0 = y1;
    }
    norm_y0 = dg::BLAS2<dg::S, Vector>::ddot( y0, dg::S(h), y0);
    std::cout << "Normalized y0 after "<<NT<<" steps is " << norm_y0 << std::endl;


    auto sol = dg::evaluate< double(&)(double), P>( solution,0.,Lx, N);
    for( unsigned i=0; i<N; i++)
        sol[i] = forward*sol[i];
    double norm_sol = dg::BLAS2<dg::S, Vector>::ddot( sol, dg::S(h), sol);
    std::cout << "Normalized solution " << norm_sol << std::endl;
    auto error(sol);

    dg::BLAS1<Vector>::daxpby( -1., y0, 1.,error);
    double norm_error = dg::BLAS2<dg::S, Vector>::ddot( error, dg::S(h),error);
    std::cout <<  "Relative error is " << sqrt( norm_error/norm_sol)<<std::endl;

    return 0;
}
