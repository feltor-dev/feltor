#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "timer.cuh"

#include "functors.cuh"

#include "evaluation.cuh"
#include "shu.cuh"
#include "rk.cuh"

#include "xspacelib.cuh"
#include "typedefs.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 3;
const double U = 1.; //the dipole doesn't move with this velocity because box is not infinite
const double R = 0.2*lx;
const double T = 2.;
const double eps = 1e-7; //CG method


double D = 0.0;
unsigned Nx = 16;
unsigned Ny = 16;

double initial( double x, double y){ return 2.*sin(x)*sin(y);}
double solution( double x, double y){ return 2.*sin(x)*sin(y)*exp(-2.*D*T);}
using namespace std;

int main()
{
    Timer t;
    Grid<double, n> grid( 0, lx, 0, ly, Nx, Ny, dg::DIR, dg::DIR);
    S2D<double,n > s2d( grid);
    ////////////////////////////////////////////////////////////
    cout << "Solve 2D incompressible NavierStokes with sin(x)sin(y) or Lamb dipole initial condition\n";
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

    DVec stencil = expand( one, grid);

    //dg::Lamb lamb( 0.5*lx, 0.5*ly, R, U);
    //HVec omega = expand( lamb, grid);
    HVec omega = expand( initial, grid );

    //dg::Lamb lamb2( 0.5*lx, 0.5*ly-0.9755*U*T, R, U);
    //HVec solh = expand( lamb2, grid);
    HVec solh = expand( solution, grid );

    DVec sol = solh;
    DVec y0( omega), y1( y0);
    //make solver and stepper
    Shu<double, n, DVec> test( grid, D, eps);
    RK< k, DVec > rk( y0);
    AB< k, DVec > ab( y0);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for first rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil, s2d, y0);
    double enstrophy = 0.5*blas2::dot( y0, s2d, y0);
    double energy =    0.5*blas2::dot( y0, s2d, test.potential()) ;

    double time = 0;
    ab.init( test, y0, dt);
    while( time < T)
    {
        //step 
        ab( test, y0, y1, dt);
        y0.swap( y1);
        //thrust::swap( y0, y1);
        time += dt;
    }
    ////////////////////////////////////////////////////////////////////
    //cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    //cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity           is: "<<blas2::dot( stencil, s2d, y0) << "\n";
    cout << "Relative enstrophy error  is: "<<(0.5*blas2::dot( s2d, y0) - enstrophy)/enstrophy<<"\n";
    test( y0, y1); //get the potential ready
    cout << "Relative energy error     is: "<<(0.5*blas2::dot( test.potential(), s2d, y0) - energy)/energy<<"\n";

    blas1::axpby( 1., sol, -1., y0);
    cout << "Distance to solution "<<sqrt( blas2::dot( s2d, y0)) << endl;

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
