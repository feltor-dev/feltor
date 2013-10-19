#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/evaluation.cuh"
#include "dg/rk.cuh"
#include "dg/xspacelib.cuh"
#include "dg/typedefs.cuh"

#include "shu.cuh"

using namespace std;
using namespace dg;

const double lx = 200.; //2.*M_PI*50.;
const double ly = 200.; //2.*M_PI*50.;

const unsigned k = 3;
const double U = 1.; //the dipole doesn't move with this velocity because box is not infinite
const double R = 0.2*lx;
const double T = 2.;
//const double eps = 1e-7; //CG method


double D = 0.001;

const unsigned m = 50; //mode number
const double kx = 2.*M_PI* (double)m/lx; 
const double ky = 2.*M_PI* (double)m/ly; 
const double ksqr = (kx*kx+ky*ky) ;//4.*M_PI*M_PI*(1./lx/lx + 1./ly/ly);

double initial( double x, double y){ return sin(kx*x)*sin(ky*y);}
double solution( double x, double y){ return sin(kx*x)*sin(ky*y)*exp(-ksqr*D*T);}

//code for either lamb dipole or analytic sine function without graphics
int main()
{
    Timer t;
    unsigned n, Nx, Ny;
    double eps;
    ////////////////////////////////////////////////////////////
    cout << "Solve 2D incompressible NavierStokes with sin(x)sin(y) or Lamb dipole initial condition\n";
    cout << "Type n, Nx and Ny and eps\n";
    cin >> n >> Nx >>Ny>>eps;
    //cout << "Type diffusion constant!\n";
    //cin >> D;
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Diffusion                   " << D <<endl;
    ////////////////////////////////////////////////////////////
    Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::DIR, dg::DIR);
    DVec w2d( create::w2d(grid));

    unsigned NT = (unsigned)(T*n*n*Nx/0.1/lx);
    //cout << "Type # of timesteps\n";
    //cin >> NT;
    const double dt = T/(double)NT;
    cout << "Runge Kutta stages          " << k <<endl;
    cout << "Timestep                    " << dt << endl;
    cout << "# of steps                  " << NT <<endl;
    ////////////////////////////////////////////////////////////

    DVec stencil = evaluate( one, grid);

    //dg::Lamb lamb( 0.5*lx, 0.5*ly, R, U);
    //HVec omega = evaluate( lamb, grid);
    HVec omega = evaluate( initial, grid );

    //dg::Lamb lamb2( 0.5*lx, 0.5*ly-0.9755*U*T, R, U);
    //HVec solh = evaluate( lamb2, grid);
    HVec solh = evaluate( solution, grid );

    DVec sol = solh;
    DVec y0( omega), y1( y0);
    //make solver and stepper
    Shu<DVec> test( grid, D, eps);
    AB< k, DVec > ab( y0);

    test( y0, y1);
    double vorticity = blas2::dot( stencil, w2d, y0);
    double enstrophy = 0.5*blas2::dot( y0, w2d, y0);
    double energy =    0.5*blas2::dot( y0, w2d, test.potential()) ;

    double time = 0;
    ab.init( test, y0, dt);
    t.tic();
    while( time < T)
    {
        //step 
        ab( test, y0, y1, dt);
        y0.swap( y1);
        //thrust::swap( y0, y1);
        time += dt;
        if( fabs(blas2::dot( w2d, y0)) > 1e16) 
        {
            cerr << "Sim unstable at time "<<time<<"!\n\n\n";
            break;
        }
    }
    t.toc();
    cout << "Total simulation time:     "<<t.diff()<<"s\n";
    cout << "Average Time for one step: "<<t.diff()/(double)NT<<"s\n";
    ////////////////////////////////////////////////////////////////////
    //cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    //cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity           is: "<<blas2::dot( stencil, w2d, y0) << "\n";
    cout << "Relative enstrophy error  is: "<<(0.5*blas2::dot( w2d, y0) - enstrophy)/enstrophy<<"\n";
    test( y0, y1); //get the potential ready
    cout << "Relative energy error     is: "<<(0.5*blas2::dot( test.potential(), w2d, y0) - energy)/energy<<"\n";

    blas1::axpby( 1., sol, -1., y0);
    cout << "Absolute distance to solution "<<sqrt( blas2::dot( w2d, y0))<< endl;
    cout << "Relative distance to solution "<<sqrt( blas2::dot( w2d, y0))/sqrt( blas2::dot( w2d, sol)) << endl;

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
