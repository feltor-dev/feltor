#include <iostream>
#include <iomanip>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "preconditioner2d.cuh"
#include "evaluation.cuh"
#include "arakawa.cuh"
#include "rk.cuh"

//#include "cg.cuh"
//#include "laplace.cuh"
//#include "tensor.cuh"

using namespace std;
using namespace dg;

const unsigned n = 1;
const unsigned Nx = 40;
const unsigned Ny = 40;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
const double hx = lx/(double)Nx;
const double hy = ly/(double)Ny;

const unsigned k = 3;
const double T = 1.;
const unsigned NT = 2*T/0.01/hx; //respect courant condition
const double dt = T/(double)NT;

typedef thrust::device_vector<double> DVec;
//typedef thrust::host_vector<double> DVec;
typedef ArrVec2d<double, n, DVec > DArrVec;
typedef cusp::device_memory MemorySpace;

double initial( double x, double y) { return sin(x)*sin(y); }
double function( double x, double y){ return sin(y); }
double result( double x, double y)  { return initial( x-cos(y)*T, y); }
double arak   ( double x, double y) { return -cos(y)*sin(y)*cos(x); }



template< class Vector_Type, class MemorySpace>
struct RHS
{
    typedef Vector_Type Vector;
    RHS(): arakawa( Nx, Ny, hx, hy), phi( expand<double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny))
    {
        //typedef cusp::ell_matrix<int, double, MemorySpace> Matrix;
        //CG<Matrix, Vector_Type, T2D<double,n> > pcg( phi.data(), n*n*Nx*Ny);
        //Matrix A = dg::dgtensor<double, n>( 
        //                       dg::create::laplace1d_per<double, n>( Ny, hy), 
        //                       dg::S1D<double, n>( hx),
        //                       dg::S1D<double, n>( hy),
        //                       dg::create::laplace1d_per<double, n>( Nx, hx)); 
        //ArrVec2d<double, n, Vector> trick(phi);
        //blas2::symv( S2D<double, n>(hx,hy), phi.data(), trick.data());
        //cout << "Number of pcg iterations "<< pcg( A, phi.data(), trick.data(), T2D<double, n>(hx, hy), 1e-10)<<endl;

        //a = expand<double(&)(double, double), n>( arak, 0, lx, 0, ly, Nx, Ny);
        //cout << "phi \n" << phi<< endl;
    }
    void operator()( const Vector& y, Vector& yp)
    {
        ArrVec2d_View<double,n, const Vector> y_view( y, Nx), yp_view( yp, Nx);
        //cout << "Y \n"<<y_view;
        arakawa( phi.data(), y, yp);
        //cout << "YP \n"<<yp_view;
        //cout << "Norm "<< dg::blas2::dot( dg::S2D<double,n >( hx, hy), yp) << endl;
        //cout << "A \n" << a ;
        //cout << "Norm "<< dg::blas2::dot( dg::S2D<double,n >( hx, hy), a.data())<<endl;
        //double x;
        //cin >>  x;
    }
  private:
    Arakawa<double, n, Vector, MemorySpace> arakawa;
    ArrVec2d<double, n, Vector> phi;
};

int main()
{
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout << "# of timesteps                    "<< NT <<endl;
    cout <<fixed<< setprecision(2)<<endl;
    DArrVec init = expand< double(&)(double, double), n> ( initial, 0, lx, 0, ly, Nx, Ny), step(init);
    Arakawa<double, n, DVec, MemorySpace>( Nx, Ny, hx, hy, init.data());
    const DArrVec solution = expand< double(&)(double, double), n> ( result, 0, lx, 0, ly, Nx, Ny);
    
    RHS<DVec, MemorySpace> rhs;
    RK<3, RHS<DVec, MemorySpace> >  rk( init.data());
    for( unsigned i=0; i<NT; i++)
    {
        rk( rhs, init.data(), step.data(), dt);
        init = step;
    }

    blas1::axpby( 1., solution.data(), -1., init.data());
    cudaThreadSynchronize();
    cout << scientific;
    cout << "Norm of error is "<<blas2::dot( S2D<double, n>(hx, hy), init.data())<<"\n";
    //n = 1 -> p = 4 ?? weird (should be 2)
    //n = 2 -> p = 2 (is error dominated by error for dx(phi)?
    //n = 3 -> p = 6 
    //n = 4 -> p = 5.5
    //n = 5 -> p = 10 !!! ( is this because of "too good" functions??)


    return 0;
}
