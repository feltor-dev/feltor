#include <iostream>
#include <iomanip>
#include <vector>
#include <thrust/scatter.h>
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
const unsigned Nx = 100;
const unsigned Ny = 50;
const double eps = 1e-6;
const double lx = 2.;
const double ly = 1.;

const double Pr = 10;
const double Ra = 5e6;

const unsigned k = 2;
const double dt = 1e-7;

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;
typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

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

    //create equidistant backward transformation
    dg::Operator<double, n> backwardeq( dg::DLT<n>::backwardEQ);
    dg::Operator<double, n*n> backward2d = dg::tensor( backwardeq, backwardeq);
    HMatrix hbackward = dg::tensor( Nx*Ny, backward2d);
    DMatrix backward = hbackward;

    //create visualisation vectors
    int running = GL_TRUE;
    DVec visual( n*n*Nx*Ny);
    thrust::device_vector<int> map = dg::makePermutationMap<n>( Nx, Ny);
    DArrVec ground = expand< double(&)(double, double), n> ( groundState, 0, lx, 0, ly, Nx, Ny), temperature( ground);
    dg::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    while (running)
    {
        //compute the total temperature
        t.tic();
        blas1::axpby( 1., y0[0], 0., temperature.data());
        blas1::axpby( 1., ground.data(), 1., temperature.data());
        //transform field to an equidistant grid
        dg::blas2::symv( backward, temperature.data(), visual);
        thrust::scatter( visual.begin(), visual.end(), map.begin(), visual.begin());
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., thrust::maximum<double>() );
        t.toc();
        std::cout << "Preparation time "<<t.diff()<<"\n";
        //draw and swap buffers
        t.tic();
        w.draw( visual, n*Nx, n*Ny, colors);
        t.toc();
        std::cout << "Color scale " << colors.scale() <<"\n";
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
