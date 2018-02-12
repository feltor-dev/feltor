#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "backend/evaluation.cuh"
#include "arakawa.h"
#include "runge_kutta.h"
#include "backend/typedefs.cuh"

const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 20;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

//const unsigned NT = 2*T/0.01/lx*(double)Nx; //respect courant condition

//![function]
const unsigned NT = 20;
const unsigned s = 17;
const double T = 1.;
template<class Vector>
void function(const std::vector<Vector>& y, std::vector<Vector>& yp){ yp = y;}
double initial( double x, double y) { return sin(x)*sin(y); }
//![function]

double function( double x, double y){ return sin(y); }
//double result( double x, double y)  { return initial( x-cos(y)*T, y); }
double arak   ( double x, double y) { return -cos(y)*sin(y)*cos(x); }
double result( double x, double y)  { return initial(x,y)*exp(T);}


int main()
{
    //Also test std::vector<DVec> functionality
    std::cout << "# of 2d cells                     " << Nx*Ny <<std::endl;
    std::cout << "# of Legendre nodes per dimension "<< n <<std::endl;
    std::cout << "# of timesteps                    "<< NT <<std::endl;
    //![doxygen]
    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    dg::DVec init = dg::evaluate ( initial, grid );
    std::vector<dg::DVec> y0( 2, init), y1(y0);
    const double dt = T/(double)NT;

    dg::RK_classic<s, std::vector<dg::DVec> >  rk( y0);
    for( unsigned i=0; i<NT; i++)
    {
        rk( function<dg::DVec>, y0, y1, dt);
        y0.swap( y1);
    }
    //![doxygen]
    dg::DVec w2d = dg::create::weights( grid);
    const dg::DVec solution = dg::evaluate ( result, grid);

    dg::blas1::axpby( 1., solution, -1., y0[0]);
    std::cout << std::scientific;
    std::cout << "Norm of error is "<<sqrt(dg::blas2::dot( w2d, y0[0]))<<"\n"; //never forget the sqrt when computing errors
    //n = 1 -> p = 2 ( as it should be )
    //n = 2 -> p = 1 (is error dominated by error for dx(phi)?
    //n = 3 -> p = 3
    //n = 4 -> p = 3
    //n = 5 -> p = 5


    return 0;
}
