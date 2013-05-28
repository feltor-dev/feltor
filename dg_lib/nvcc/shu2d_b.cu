#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "timer.cuh"

#include "functors.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "shu.cuh"
#include "rk.cuh"



using namespace std;
using namespace dg;

const unsigned n = 3;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 3;
const double U = 1.; //the dipole doesn't move with this velocity because box is not infinite
const double R = 0.2*lx;
const double T = 2.;
const double eps = 1e-6; //CG method

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;


double D = 0.0;
unsigned Nx = 16;
unsigned Ny = 16;

double initial( double x, double y){ return 2.*sin(x)*sin(y);}
double solution( double x, double y){ return 2.*sin(x)*sin(y)*exp(-2.*D*T);}
using namespace std;

int main()
{
    Timer t;
    ////////////////////////////////////////////////////////////
    cout << "Type # of grid cells in one dimension!\n";
    cin >> Nx;
    Ny = Nx; 
    cout << "Type diffusion constant!\n";
    cin >> D;
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Diffusion                   " << D <<endl;
    ////////////////////////////////////////////////////////////

    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;
    unsigned NT = (unsigned)(T*n*Nx/0.025/lx);
    cout << "Type # of timesteps\n";
    cin >> NT;
    const double dt = T/(double)NT;
    cout << "Runge Kutta stages          " << k <<endl;
    cout << "Timestep                    " << dt << endl;
    cout << "# of steps                  " << NT <<endl;
    ////////////////////////////////////////////////////////////

    DArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    //dg::Lamb lamb( 0.5*lx, 0.5*ly, R, U);
    //HArrVec omega = expand< dg::Lamb, n> ( lamb, 0, lx, 0, ly, Nx, Ny);
    HArrVec omega = expand< double(&)(double, double), n> ( initial, 0, lx, 0, ly, Nx, Ny);

    //dg::Lamb lamb2( 0.5*lx, 0.5*ly-0.9755*U*T, R, U);
    //HArrVec solh = expand< dg::Lamb, n> ( lamb2, 0, lx, 0, ly, Nx, Ny);
    HArrVec solh = expand< double(&)(double, double), n> ( solution, 0, lx, 0, ly, Nx, Ny);

    DVec sol = solh.data();
    DVec y0( omega.data()), y1( y0);
    //make solver and stepper
    Shu<double, n, DVec> test( Nx, Ny, hx, hy, D, eps);
    RK< k, Shu<double, n, DVec> > rk( y0);
    AB< k, Shu<double, n, DVec> > ab( y0);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0);
    double enstrophy = 0.5*blas2::dot( y0, S2D<double, n>(hx, hy), y0);
    double energy =    0.5*blas2::dot( y0, S2D<double, n>(hx, hy), test.potential()) ;

    double time = 0;
    ab.init( test, y0, dt);
    while( time < T)
    {
        //step 
        ab( test, y0, y1, dt);
        thrust::swap(y0, y1);
        time += dt;
    }
    ////////////////////////////////////////////////////////////////////
    //cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    //cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity           is: "<<blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
    cout << "Relative enstrophy error  is: "<<(0.5*blas2::dot( S2D<double, n>(hx, hy), y0) - enstrophy)/enstrophy<<"\n";
    test( y0, y1); //get the potential ready
    cout << "Relative energy error     is: "<<(0.5*blas2::dot( test.potential(), S2D<double, n>(hx, hy), y0) - energy)/energy<<"\n";

    blas1::axpby( 1., sol, -1., y0);
    cout << "Distance to solution "<<sqrt( blas2::dot( S2D<double,n >(hx,hy), y0)) << endl;

    //energy and enstrophy errrors are due to timestep only ( vorticity is exactly conserved)
    // k = 2 | p = 3
    // k = 3 | p = 4
    // k = 4 | p = 5

    //solution to sin(x)sin(y) 
    // n = 1 
    // n = 2 | p = 2
    // n = 3 | p = 2.6
    // n = 4 | p = 4

    return 0;

}
