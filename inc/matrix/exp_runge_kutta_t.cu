#include <iostream>
#include "dg/algorithm.h"
#include "exp_runge_kutta.h"

// integrate Dot y = -100 y +sin(t)
const double matrix = -1;

void rhs( double t, double y, double& yp)
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


int main()
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
    std::cout << "Norm of error in "<<std::setw(24) <<"ExponentialStep"<<"\t"<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    std::vector<std::string> names{
        "Euler",
        "Midpoint-2-2",
        "Runge-Kutta-4-4",
        "Hochbruck-3-3-4"
    };
    for ( auto name : names)
    {
        double u = solution(t_start, y_start);
        double u1(u), sol = solution(t_end, y_start);
        dg::mat::ExponentialERKStep<double> rk( name, u);
        auto mat = MatrixFunction();
        dg::SinglestepTimeloop<double>( rk, std::tie(rhs,mat),
                dt).integrate( t_start, u, t_end, u1);
        dg::blas1::axpby( 1., sol , -1., u1);
        std::cout << "Norm of error in "<<std::setw(24) <<name<<"\t"<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    }
    return 0;
}
