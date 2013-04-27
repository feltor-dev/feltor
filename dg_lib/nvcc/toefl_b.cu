#include <iostream>
#include <iomanip>
#include <vector>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "cuda_texture.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "functions.h"
#include "functors.cuh"
#include "toefl.cuh"
#include "rk.cuh"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 10;
const double eps = 1e-6;
const double lx = 2.;
const double ly = 1.;

const double Pr = 10;
const double Ra = 5e6;

const unsigned k = 2;
const double dt = 1e-6;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;

using namespace std;

double groundState( double x, double y) { return ly/2. - y;}

int main()
{
    dg::Window w(800, 400);
    glfwSetWindowTitle( "Behold the convection\n");

    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;

    /////////////////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;

    //create initial vector
    dg::Gaussian g( lx/2., ly/2., .1, .1, 0.5);
    DArrVec theta = dg::expand<dg::Gaussian, n> ( g, 0.,lx, 0., ly, Nx, Ny);
    vector<DVec> y0(2, theta.data()), y1(y0);
    y0[1] = DVec( n*n*Nx*Ny, 0.); //omega is zero

    //create RHS and RK
    Toefl<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, Ra, Pr, eps); 
    RK< k, Toefl<double, n, DVec, cusp::device_memory> > rk( y0);

    //create visualisation vectors
    int running = GL_TRUE;
    DVec visual( Nx*Ny);
    HArrVec hstencil( Ny, Nx, 0);
    for( unsigned i=0; i<Ny; i++)
        for( unsigned j=0; j<Nx; j++)
            hstencil( i,j, 0,0) = 1.; //correct way to produce stencil exactly!!
    DVec stencil( hstencil.data()) ;
    DArrVec ground = expand< double(&)(double, double), n> ( groundState, 0, lx, 0, ly, Nx, Ny), temperature( ground);
    dg::ColorMapRedBlueExt colors( 1.);
    Timer t;
    while (running)
    {
        //compute the total temperature
        t.tic();
        blas1::axpby( 1., y0[0], 0., temperature.data());
        blas1::axpby( 1., ground.data(), 1., temperature.data());
        //reduce the field to the 00 values 
        thrust::remove_copy_if( (temperature.data()).begin(), (temperature.data()).end(), (stencil).begin(), visual.begin(), thrust::logical_not<double>() );
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., thrust::maximum<double>() );
        std::cout << "Color scale " << colors.scale() <<"\n";
        //draw and swap buffers
        w.draw( visual, Nx, Ny, colors);
        t.toc();
        std::cout << "Visualisation time " <<t.diff()<<"\n";
        //step 
        t.tic();
        rk( test, y0, y1, dt);
        for( unsigned i=0; i<2; i++)
            thrust::swap( y0[i], y1[i]);
        t.toc();
        std::cout << "Timer for one step "<<t.diff()<<"\n";
        //glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////

    return 0;

}
