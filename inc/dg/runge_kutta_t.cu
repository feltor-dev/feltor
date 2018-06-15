#include <iostream>
#include <iomanip>
#include <functional>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "backend/typedefs.h"
#include "geometry/evaluation.h"
#include "arakawa.h"
#include "runge_kutta.h"


//![function]
void rhs(double t, const std::array<double,2>& y, std::array<double,2>& yp, double damping, double omega_0, double omega_drive){
    //damped driven harmonic oscillator
    // x -> y[0] , v -> y[1]
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
    return {x,v};
}

int main()
{
    std::cout << "Program to test correct implementation of Runge Kutta methods in runge_kutta.h at the example of the damped driven harmonic oscillator. Errors should be small! \n";
    std::cout << std::scientific;
    //![doxygen]
    //... in main
    //set start and end time, number of steps and timestep
    const double t_start = 0., t_end = 1.;
    const unsigned N = 40;
    const double dt = (t_end - t_start)/(double)N;
    //set physical parameters and initial condition
    const double damping = 0.2, omega_0 = 1.0, omega_drive = 0.9;
    std::array<double,2> u = solution(t_start, damping, omega_0, omega_drive);
    //construct Runge Kutta class
    dg::RK<2, std::array<double,2> >  rk( u);
    //construct a functor with the right interface
    using namespace std::placeholders; //for _1, _2, _3
    auto functor = std::bind( rhs, _1, _2, _3, damping, omega_0, omega_drive);
    //integration loop
    double t=t_start;
    for( unsigned i=0; i<N; i++)
        rk.step( functor, t, u, t, u, dt); //step inplace
    //now compute error
    dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u);
    std::cout << "Norm of error is "<<sqrt(dg::blas1::dot( u, u))<<"\n";
    //![doxygen]
    u = solution(t_start, damping, omega_0, omega_drive);
    std::array<double, 2> u1(u), sol = solution(t_end, damping, omega_0, omega_drive);
    dg::stepperRK<1>( functor, t_start, u, t_end, u1, N);
    dg::blas1::axpby( 1., sol , -1., u1);
    std::cout << "Norm of error in stepperRK<1> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::stepperRK<2>( functor, t_start, u, t_end, u1, N);
    dg::blas1::axpby( 1., sol , -1., u1);
    std::cout << "Norm of error in stepperRK<2> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::stepperRK<3>( functor, t_start, u, t_end, u1, N);
    dg::blas1::axpby( 1., sol , -1., u1);
    std::cout << "Norm of error in stepperRK<3> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::stepperRK<4>( functor, t_start, u, t_end, u1, N);
    dg::blas1::axpby( 1., sol , -1., u1);
    std::cout << "Norm of error in stepperRK<4> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::stepperRK<6>( functor, t_start, u, t_end, u1, N);
    dg::blas1::axpby( 1., sol , -1., u1);
    std::cout << "Norm of error in stepperRK<6> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::stepperRK<17>( functor, t_start, u, t_end, u1, N);
    dg::blas1::axpby( 1., sol , -1., u1);
    std::cout << "Norm of error in stepperRK<17> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::RK_opt<1, std::array<double,2> > rk_opt1(u);
    u1 = u;
    t=t_start;
    for( unsigned i=0; i<N; i++)
        rk_opt1.step( functor, t, u1, t, u1, dt); //step inplace
    dg::blas1::axpby( 1., sol, -1., u1);
    std::cout << "Norm of error in RK_opt<1> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::RK_opt<2, std::array<double,2> > rk_opt2(u);
    u1 = u;
    t=t_start;
    for( unsigned i=0; i<N; i++)
        rk_opt2.step( functor, t, u1, t, u1, dt); //step inplace
    dg::blas1::axpby( 1., sol, -1., u1);
    std::cout << "Norm of error in RK_opt<2> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::RK_opt<3, std::array<double,2> > rk_opt3(u);
    u1 = u;
    t=t_start;
    for( unsigned i=0; i<N; i++)
        rk_opt3.step( functor, t, u1, t, u1, dt); //step inplace
    dg::blas1::axpby( 1., sol, -1., u1);
    std::cout << "Norm of error in RK_opt<3> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";
    dg::RK_opt<4, std::array<double,2> > rk_opt4(u);
    u1 = u;
    t=t_start;
    for( unsigned i=0; i<N; i++)
        rk_opt4.step( functor, t, u1, t, u1, dt); //step inplace
    dg::blas1::axpby( 1., sol, -1., u1);
    std::cout << "Norm of error in RK_opt<4> is "<<sqrt(dg::blas1::dot( u1, u1))<<"\n";

    return 0;
}
