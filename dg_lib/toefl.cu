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
const unsigned Nx = 33;
const unsigned Ny = 33;

const double lx = 1.;
const double ly = 1.;

const Parameter p = {0.005, 0.999, 0.001, 1, 48};

const unsigned k = 2;
const double dt = 1e-3;
const double eps = 1e-2; //The condition for conjugate gradient

const unsigned N = 10;// only every Nth computation is visualized

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
    dg::HostWindow w(400, 400);
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
    vector<DVec> y0(3, theta.data()), y1(y0);

    //create RHS and RK
    Toefl<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, p, eps); 
    RK< k, Toefl<double, n, DVec, cusp::device_memory> > rk( y0);
    test.update_exponent( y0, y0); //transform y0 to g0

    //create equidistant backward transformation
    dg::Operator<double, n> backwardeq( dg::DLT<n>::backwardEQ);
    dg::Operator<double, n*n> backward2d = dg::tensor( backwardeq, backwardeq);
    HMatrix hbackward = dg::tensor( Nx*Ny, backward2d);
    DMatrix backward = hbackward;

    //create visualisation vectors
    int running = GL_TRUE;
    DVec visual( n*n*Nx*Ny);
    HVec hvisual( n*n*Nx*Ny);
    thrust::device_vector<int> map = dg::makePermutationMap<n>( Nx, Ny);
    dg::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    while (running)
    {
        t.tic();
        //transform field to an equidistant grid
        test.update_log( y0, y0); //transform g0 to y0
        dg::blas2::symv( backward, y0[0], visual);
        thrust::scatter( visual.begin(), visual.end(), map.begin(), visual.begin());
        test.update_exponent( y0, y0); //transform y0 to g0
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        hvisual = visual;
        w.draw( hvisual, n*Nx, n*Ny, colors);
        t.toc();
        std::cout << "Color scale " << colors.scale() <<"\n";
        std::cout << "Visualisation time        " <<t.diff()<<"s\n";
        //step 
        t.tic();
        for( unsigned i=0; i<N; i++)
        {
            rk( test, y0, y1, dt);
            for( unsigned i=0; i<3; i++)
                thrust::swap( y0[i], y1[i]);
        }
        t.toc();
        std::cout << "Time for "<<N<<" step(s)      "<<t.diff()<<"s\n";
        //glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////

    return 0;

}
