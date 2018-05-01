#include <iostream>
#include <iomanip>
#include <functional>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "backend/typedefs.cuh"
#include "geometry/evaluation.cuh"
#include "arakawa.h"
#include "runge_kutta.h"


//![function]
//driven damped harmonic oscillator
void rhs(double t, const std::array<double,2>& y, std::array<double,2>& yp, double damping, double omega_0, double omega_drive){
    // y[0] = x, y[1] = v
    yp[0] = y[1];
    yp[1] = -2.*damping*omega_0*y[1] - omega_0*omega_0*y[0] + sin(omega_drive*t);
}
//![function]

std::array<double, 2> solution( double t, double damping, double omega_0, double omega_drive)
{
    double tmp1 = (2.*omega_0*damping);
    double tmp2 = (omega_0*omega_0 - omega_drive*omega_drive)/omega_drive;
    double amp = 1./sqrt( tmp1*tmp1 + tmp2*tmp2);
    double phi = atan( 2.*omega_drive*omega_0*damping/(omega_drive*omega_drive-omega_0*omega_0));

    double x = amp*sin(omega_drive*t+phi)/omega_drive;
    double v = amp*cos(omega_drive*t+phi);
    std::array<double,2> a = {x,v};
    return a;
}

int main()
{
    std::cout << "Program to test correct implementation of Runge Kutta methods in runge_kutta.h at the example of the driven harmonic oscillator\n";
    std::cout << std::scientific;
    //![doxygen]
    using namespace std::placeholders; //for _1, _2, _3
    const double damping = 0.2, omega_0 = 1.0, omega_drive = 0.9;
    const double t_start = 0, t_end = 1.;
    //initial condition
    std::array<double,2> u = solution(t_start, damping, omega_0, omega_drive);
    const unsigned N = 40;
    const double dt = t_end/(double)N;
    dg::RK<4, std::array<double,2>>  rk( u);
    auto functor = std::bind( rhs, _1, _2, _3, damping, omega_0, omega_drive);
    double t=0;
    for( unsigned i=0; i<N; i++)
        rk.step( functor, t, u, t, u, dt);
    dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u);
    std::cout << "Norm of error is "<<sqrt(dg::blas1::dot( u, u))<<"\n";
    //![doxygen]
    dg::RK<1, std::array<double,2>>  rk1( u);
    u = solution(t_start, damping, omega_0, omega_drive);
    t=0;
    for( unsigned i=0; i<N; i++)
        rk1.step( functor, t, u, t, u, dt);
    dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u);
    std::cout << "Norm of error is "<<sqrt(dg::blas1::dot( u, u))<<"\n";

    return 0;
}
