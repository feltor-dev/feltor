#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "timer.cuh"

#include "draw/host_window.h"
#include "functors.cuh"

#include "evaluation.cuh"
#include "shu.cuh"
#include "rk.cuh"

#include "xspacelib.cuh"
#include "typedefs.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 32; 
const unsigned Ny = 32; 
const double lx = 1.;
const double ly = 1.;

const unsigned k = 3;
const double D = 0.0;
const double U = 1; //the dipole doesn't move with this velocity because box is not infinite
const double R = 0.2*lx;
const double T = 1.;//0.6;
const unsigned NT =  (unsigned)(T*n*Nx/0.05/lx);
const double eps = 1e-3; //CG method
const unsigned N = 3; //only output every Nth step 


using namespace std;

int main()
{
    Timer t;
    Grid<double, n> grid( 0, lx, 0, ly, Nx, Ny, dg::PER, dg::PER);
    S2D<double,n > s2d( grid.hx(), grid.hy());
    const double dt = T/(double)NT;
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    draw::HostWindow w( 600, 600);
    glfwSetWindowTitle( "Navier Stokes");
    ////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    //cout << "# of timesteps              " << NT << endl;
    cout << "Diffusion                   " << D <<endl;
    dg::Lamb lamb( 0.5*lx, 0.8*ly, R, U);
    HVec omega = expand ( lamb, grid);
    DVec stencil = expand( one, grid);
    dg::Lamb lamb2( 0.5*lx, 0.8*ly-0.9755*U*T, R, U);
    HVec solh = expand( lamb2, grid);
    DVec sol = solh ;
    DVec y0( omega ), y1( y0);
    //make solver and stepper
    Shu<double, n, DVec> test( grid, D, eps);
    RK< k, DVec > rk( y0);
    AB< k, DVec > ab( y0);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil , s2d, y0);
    double enstrophy = 0.5*blas2::dot( y0, s2d, y0);
    double energy =    0.5*blas2::dot( y0, s2d, test.potential()) ;

    double time = 0;
    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::DMatrix equidistant = dg::create::backscatter( grid, LSPACE );
    int running = GL_TRUE;
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( test, y0, dt);
    //cout << "Press any key to start!\n";
    //double x; 
    //cin >> x;
    while (running && time < T)
    {
        dg::blas2::symv( equidistant, y0, visual);
        cudaThreadSynchronize();
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        hvisual = visual;
        cudaThreadSynchronize();
        w.draw( hvisual, n*Nx, n*Ny, colors);
        //step 
        t.tic();
        for( unsigned i=0; i<N; i++)
        {
            ab( test, y0, y1, dt);
            y0.swap( y1);
            //thrust::swap(y0, y1);
        }
        t.toc();
        //cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        cout << "Simulation Time "<<time<< " \ttook "<<t.diff()/(double)N<<"\t per step"<<endl;
        time += N*dt;

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////
    cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity           is: "<<blas2::dot( stencil , s2d, y0) << "\n";
    cout << "Relative enstrophy error  is: "<<(0.5*blas2::dot( s2d, y0) - enstrophy)/enstrophy<<"\n";
    test( y0, y1); //get the potential ready
    cout << "Relative energy error     is: "<<(0.5*blas2::dot( test.potential(), s2d, y0) - energy)/energy<<"\n";

    blas1::axpby( 1., y0, -1, sol);
    cout << "Distance to solution: "<<sqrt(blas2::dot( s2d, sol ))<<endl;

    cout << "Press any key to quit!\n";
    cin >> x;
    return 0;

}
