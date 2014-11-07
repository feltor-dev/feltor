#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "backend/evaluation.cuh"
#include "arakawa.h"
#include "runge_kutta.h"
#include "backend/typedefs.cuh"

//#include "cg.cuh"
//#include "laplace.cuh"
//#include "tensor.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 20;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 3;
const double T = 1.;
const unsigned NT = 2*T/0.01/lx*(double)Nx; //respect courant condition
const double dt = T/(double)NT;


double initial( double x, double y) { return sin(x)*sin(y); }
double function( double x, double y){ return sin(y); }
double result( double x, double y)  { return initial( x-cos(y)*T, y); }
double arak   ( double x, double y) { return -cos(y)*sin(y)*cos(x); }

template< class Vector_Type>
struct RHS
{
    typedef std::vector<Vector_Type> Vector;
    RHS(const Grid2d<double>& grid): arakawa( grid), phi( evaluate( function, grid)),
                                     temp(phi)
    { }
    void operator()(const Vector& y, Vector& yp)
    {
        for( unsigned i=0; i<y.size(); i++)
        {
            dg::blas1::axpby( 1., y[i], 0, temp);
            arakawa( phi, temp, yp[i]);
        }
    }
  private:
    ArakawaX<dg::DMatrix, Vector_Type> arakawa;
    Vector_Type phi, temp;
};

int main()
{
    dg::Grid2d<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    dg::DVec w2d = dg::create::weights( grid);
    //Also test std::vector<DVec> functionality
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout << "# of timesteps                    "<< NT <<endl;
    //cout <<fixed<< setprecision(2)<<endl;
    dg::DVec init = evaluate ( initial, grid );
    const dg::DVec solution = evaluate ( result, grid);
    std::vector<dg::DVec> y0( 2, init), y1(y0);

    RHS<dg::DVec> rhs( grid);

    //integrateRK4( rhs, y0, y1, T, 1e-10);
    //std::vector<DVec> error(2, solution);
    //double norm_sol = blas2::dot( w2d, solution);
    //blas1::axpby( -1., y1, 1., error);
    //double norm_error = blas2::dot( w2d, error);
    //cout << "Relative error is      "<< sqrt( norm_error/norm_sol)<<" \n";
    
    RK<k, std::vector<dg::DVec> >  rk( y0);
    for( unsigned i=0; i<NT; i++)
    {
        rk( rhs, y0, y1, dt);
        y0.swap( y1);
        //for( unsigned i=0; i<y0.size(); i++)
        //    thrust::swap( y0[i], y1[i]); 
            
    }

    blas1::axpby( 1., solution, -1., y0[0]);
    cout << scientific;
    cout << "Norm of error is "<<sqrt(blas2::dot( w2d, y0[0]))<<"\n"; //never forget the sqrt when computing errors
    //n = 1 -> p = 2 ( as it should be )
    //n = 2 -> p = 1 (is error dominated by error for dx(phi)?
    //n = 3 -> p = 3 
    //n = 4 -> p = 3
    //n = 5 -> p = 5 


    return 0;
}
