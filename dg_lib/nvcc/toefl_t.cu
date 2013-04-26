#include <iostream>
#include <iomanip>
#include <vector>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "cuda_texture.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "functions.cuh"
#include "functors.cuh"
#include "toefl.cuh"
#include "rk.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 10;
const double lx = 2.;
const double ly = 1.;

const double Pr = 10;
const double Ra = 5e5;

const unsigned k = 2;
const double dt = 1e-6;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;

using namespace std;

int main()
{
    dg::Window w( 400, 300);
    glfwSetWindowTitle( "Behold the convection\n");

    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;

    /////////////////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;

    //create initial vector
    dg::Gaussian g( lx/2., ly/2., .01, .01, 1);
    DArrVec theta = dg::expand<dg::Gaussian, n> ( g, 0.,lx, 0., ly, Nx, Ny);
    vector<DVec> y0(2, theta.data()), y1(y0);
    y0[1] = DVec( n*n*Nx*Ny, 0.); //omega is zero

    Toefl<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, Ra, Pr, 1e-6); 

    //create visualisation vectors
    int running = GL_TRUE;
    DVec visual( Nx*Ny);
    DArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny), vorticity( stencil);
    dg::ColorMapRedBlueExt colors( 1.);
    while (running)
    {
        //reduce the field to the 00 values 
        thrust::remove_copy_if( y0[0].begin(), y0[0].end(), stencil.data().begin(), visual.begin(), thrust::logical_not<double>() );
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., thrust::maximum<double>() );
        //draw and swap buffers
        w.draw( visual, Nx, Ny, colors);
        //step 
        rk( test, y0, y1, dt);
        for( unsigned i=0; i<2; i++)
            thrust::swap( y0[i], y1[i];
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////

    return 0;

}
