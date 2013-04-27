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

const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 20;
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
    typedef std::vector<Vector_Type> Vector;
    RHS(): arakawa( Nx, Ny, hx, hy, -1, -1), phi( expand<double(&)(double, double), n>( function, 0, lx, 0, ly, Nx, Ny))
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
        //ArrVec2d_View<double,n, const Vector> y_view( y, Nx), yp_view( yp, Nx);
        //cout << "Y \n"<<y_view;
        for( unsigned i=0; i<y.size(); i++)
            arakawa( phi.data(), y[i], yp[i]);
        //cout << "YP \n"<<yp_view;
        //cout << "Norm "<< dg::blas2::dot( dg::S2D<double,n >( hx, hy), yp) << endl;
        //cout << "A \n" << a ;
        //cout << "Norm "<< dg::blas2::dot( dg::S2D<double,n >( hx, hy), a.data())<<endl;
        //double x;
        //cin >>  x;
    }
  private:
    Arakawa<double, n, Vector_Type, MemorySpace> arakawa;
    ArrVec2d<double, n, Vector_Type> phi;
};

int main()
{
    //Also test std::vector<DVec> functionality
    cout << "# of 2d cells                     " << Nx*Ny <<endl;
    cout << "# of Legendre nodes per dimension "<< n <<endl;
    cout << "# of timesteps                    "<< NT <<endl;
    cout <<fixed<< setprecision(2)<<endl;
    DArrVec init = expand< double(&)(double, double), n> ( initial, 0, lx, 0, ly, Nx, Ny);
    const DArrVec solution = expand< double(&)(double, double), n> ( result, 0, lx, 0, ly, Nx, Ny);
    std::vector<DVec> y0( 2, init.data()), y1(y0);
    
    RHS<DVec, MemorySpace> rhs;
    RK<3, RHS<DVec, MemorySpace> >  rk( y0);
    for( unsigned i=0; i<NT; i++)
    {
        rk( rhs, y0, y1, dt);
        for( unsigned i=0; i<y0.size(); i++)
            thrust::swap( y0[i], y1[i]); 
    }

    blas1::axpby( 1., solution.data(), -1., y0[0]);
    cudaThreadSynchronize();
    cout << scientific;
    cout << "Norm of error is "<<sqrt(blas2::dot( S2D<double, n>(hx, hy), y0[0]))<<"\n"; //never forget the sqrt when computing errors
    //n = 1 -> p = 2 ( as it should be )
    //n = 2 -> p = 1 (is error dominated by error for dx(phi)?
    //n = 3 -> p = 3 
    //n = 4 -> p = 3
    //n = 5 -> p = 5 


    return 0;
}
