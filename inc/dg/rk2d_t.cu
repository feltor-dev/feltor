#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "preconditioner2d.cuh"
#include "evaluation.cuh"
#include "arakawa.cuh"
#include "rk.cuh"
#include "typedefs.cuh"

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
    RHS(const Grid<double>& grid): arakawa( grid), phi( expand( function, grid))
    { }
    void operator()( const Vector& y, Vector& yp)
    {
        for( unsigned i=0; i<y.size(); i++)
            arakawa( phi, y[i], yp[i]);
    }
  private:
    Arakawa<Vector_Type> arakawa;
    Vector_Type phi;
};

int main()
{
    dg::Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    dg::S2D<double> s2d( grid);
    //Also test std::vector<DVec> functionality
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout << "# of timesteps                    "<< NT <<endl;
    cout <<fixed<< setprecision(2)<<endl;
    DVec init = expand ( initial, grid );
    const DVec solution = expand ( result, grid);
    std::vector<DVec> y0( 2, init), y1(y0);
    
    RHS<DVec> rhs( grid);
    RK<k, std::vector<DVec> >  rk( y0);
    for( unsigned i=0; i<NT; i++)
    {
        rk( rhs, y0, y1, dt);
        y0.swap( y1);
        //for( unsigned i=0; i<y0.size(); i++)
        //    thrust::swap( y0[i], y1[i]); 
            
    }

    blas1::axpby( 1., solution, -1., y0[0]);
    cudaThreadSynchronize();
    cout << scientific;
    cout << "Norm of error is "<<sqrt(blas2::dot( s2d, y0[0]))<<"\n"; //never forget the sqrt when computing errors
    //n = 1 -> p = 2 ( as it should be )
    //n = 2 -> p = 1 (is error dominated by error for dx(phi)?
    //n = 3 -> p = 3 
    //n = 4 -> p = 3
    //n = 5 -> p = 5 


    return 0;
}
