#include <iostream>
#include <iomanip>
#include <functional>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "backend/typedefs.h"
#include "geometry/evaluation.h"
#include "arakawa.h"
#include "runge_kutta.h"
#include "adaptive.h"


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
    std::cout << "Program to test correct implementation of adaptive methods in adaptive.h at the example of the damped driven harmonic oscillator. Errors should be small! \n";
    std::cout << std::scientific;
    //![doxygen]
    //... in main
    //set start and end time
    const double t_start = 0., t_end = 1.;
    //set physical parameters and initial condition
    const double damping = 0.2, omega_0 = 1.0, omega_drive = 0.9;
    std::array<double,2> u_start = solution(t_start, damping, omega_0, omega_drive), u_end(u_start);
    //construct a functor with the right interface
    using namespace std::placeholders; //for _1, _2, _3
    auto functor = std::bind( rhs, _1, _2, _3, damping, omega_0, omega_drive);
    double dt= 0;
    //integration
    int counter = dg::integrateRK45( functor, t_start, u_start, t_end, u_end, dt, 1e-6);
    //now compute error
    dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u_end);
    std::cout << "With "<<counter<<"\t Embedded RK 4-5 steps norm of error is\t "<<sqrt(dg::blas1::dot( u_end, u_end))<<"\n";
    //![doxygen]
    dt = 0;
    counter = dg::integrateHRK<4>( functor, t_start, u_start, t_end, u_end, dt, 1e-6);
    dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u_end);
    std::cout << "With "<<counter<<"\t Halfstep RK 4 steps norm of error is\t "<<sqrt(dg::blas1::dot( u_end, u_end))<<"\n";

    dt = 0;
    counter = dg::integrateHRK<6>( functor, t_start, u_start, t_end, u_end, dt, 1e-6);
    dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u_end);
    std::cout << "With "<<counter<<"\t Halfstep RK 6 steps norm of error is\t "<<sqrt(dg::blas1::dot( u_end, u_end))<<"\n";

    dt = 0.;
    counter = dg::integrateHRK<17>( functor, t_start, u_start, t_end, u_end, dt, 1e-6);
    dg::blas1::axpby( 1., solution(t_end, damping, omega_0, omega_drive), -1., u_end);
    std::cout << "With "<<counter<<"\t Halfstep RK 17 steps norm of error is\t "<<sqrt(dg::blas1::dot( u_end, u_end))<<"\n";

    return 0;
}
