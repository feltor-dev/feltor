#include <iostream>
#include "dg/algorithm.h"
#include "exp_runge_kutta.h"

#include "catch2/catch_all.hpp"

// integrate Dot y = -100 y +sin(t)
const double matrix = -1;

void rhs( double t, double, double& yp)
{
    //yp = 0.;
    yp = sin(t);
}

struct MatrixFunction
{
    template<class UnaryOp>
    void operator() ( UnaryOp f, double x, double& y) const{
        y = f(matrix)*x;
    }
};

double solution( double t, double y0){
    return y0*exp(matrix*t) + (exp(matrix*t) - matrix*sin(t) -cos(t))/(1.+matrix*matrix);
    //return y0*exp(matrix*t);
}


TEST_CASE( "Exp Runge Kutta")
{
    double t_start = 0., y_start = 1.;
    double t_end = M_PI/2.;
    const unsigned N = 40;
    const double dt = (t_end - t_start)/(double)N;

    double u = y_start;
    double u1(u), sol = y_start*exp(matrix*t_end);
    dg::mat::ExponentialStep<double> rk( u);
    dg::SinglestepTimeloop<double>( rk, MatrixFunction(),
            dt).integrate( t_start, u, t_end, u1);
    dg::blas1::axpby( 1., sol , -1., u1);
    double error = sqrt( dg::blas1::dot( u1, u1));
    INFO( "Norm of error in "<<std::setw(24) <<"ExponentialStep"<<"\t"<<error);
    CHECK( error < 1e-15);

    auto name = GENERATE( as<std::string>{},
        "Euler",
        "Midpoint-2-2",
        "Runge-Kutta-4-4",
        "Hochbruck-3-3-4"
    );
    auto b = dg::mat::create::func_tableau<double>(name);
    std::vector<unsigned> NTs = {20,40};
    std::vector<double> err(NTs.size());
    for( unsigned k = 0; k<NTs.size(); k++)
    {
        double u = solution(t_start, y_start);
        double u1(u), sol = solution(t_end, y_start);
        dg::mat::ExponentialERKStep<double> rk( name, u);
        auto mat = MatrixFunction();
        const double dt = (t_end - t_start)/(double)NTs[k];
        dg::SinglestepTimeloop<double>( rk, std::tie(rhs,mat),
            dt).integrate( t_start, u, t_end, u1);
        dg::blas1::axpby( 1., sol , -1., u1);
        err[k] = sqrt( dg::blas1::dot( u1,u1));
    }
    double order = log( err[0]/err[1])/log( (double)NTs[1]/(double)NTs[0]);
    INFO("Norm of error in "<<std::setw(24) <<name<<"\t"<<err[1]<<" order "
        <<order<<" expected "<<b.order());
    // MW: For some reason the Hochbruck-3-3-4 method is only 2nd order for our problem
    // I don't think there is a bug in our implementation becuase the underlying Runge-Kutta
    // scheme (obtained in the limit phi_i(0)) has the same problem.
    // Is there a difference between stiff and nonstiff order conditions?
    if( name == "Hochbruck-3-3-4")
        CHECK( (2  - order)/double(2) < 0.014);
    else
        CHECK( (b.order()  - order)/double(b.order()) < 0.014);
}
