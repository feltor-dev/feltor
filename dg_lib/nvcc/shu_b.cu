#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "timer.cuh"

#include "cuda_texture.cuh"
#include "functors.cuh"

#include "arrvec2d.cuh"
#include "evaluation.cuh"
#include "shu.cuh"
#include "rk.cuh"



using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 25;
const unsigned Ny = 25;
const double lx = 1.;
const double ly = 1.;

const unsigned k = 2;
const double D = 0.01;
const double T = 1.;
const unsigned NT = (unsigned)(D*T*n*n*Nx*Nx/0.01/lx/lx);

typedef thrust::device_vector< double>   DVec;
typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec2d< double, n, HVec>  HArrVec;
typedef dg::ArrVec2d< double, n, DVec>  DArrVec;
typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;


double initial( double x, double y){return 2.*sin(x)*sin(y);}
double solution( double x, double y) {return 2.*sin(x)*sin(y)*exp( -2.*T*D);}


using namespace std;

int main()
{
    Timer t;
    const double hx = lx/ (double)Nx;
    const double hy = ly/ (double)Ny;
    const double dt = T/(double)NT;
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    HostWindow w( 600, 600);
    glfwSetWindowTitle( "Navier Stokes");
    ////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    //cout << "# of timesteps              " << NT << endl;
    cout << "Diffusion                   " << D <<endl;
    dg::Lamb lamb( 0.5*lx, 0.5*ly, 0.2*lx, 1);
    HArrVec omega = expand< dg::Lamb, n> ( lamb, 0, lx, 0, ly, Nx, Ny);
    DArrVec stencil = expand< double(&)(double, double), n> ( one, 0, lx, 0, ly, Nx, Ny);
    //DArrVec sol = expand< double(&)(double, double), n> ( solution, 0, lx, 0, ly, Nx, Ny);
    DVec y0( omega.data()), y1( y0);
    Shu<double, n, DVec, cusp::device_memory> test( Nx, Ny, hx, hy, D);
    RK< 3, Shu<double, n, DVec, cusp::device_memory> > rk( y0);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0);
    double enstrophy = blas2::dot( y0, S2D<double, n>(hx, hy), y0);
    double energy =    blas2::dot( y0, S2D<double, n>(hx, hy), test.potential()) ;

    unsigned step = 0;
    ////////////////////////////////glfw//////////////////////////////
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
    while (running)
    {
        //transform field to an equidistant grid
        t.tic();
        cout << "Total vorticity is: "<<blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
        cout << "Total enstrophy is: "<<blas2::dot( S2D<double, n>(hx, hy), y0)<<"\n";
        test( y0, y1); //get the potential ready
        cout << "Total energy    is: "<<blas2::dot( test.potential(), S2D<double, n>(hx, hy), y0)<<"\n";
        t.toc();
        dg::blas2::symv( backward, y0, visual);
        thrust::scatter( visual.begin(), visual.end(), map.begin(), visual.begin());
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        std::cout << "Color scale " << colors.scale() <<"\n";
        //draw and swap buffers
        hvisual = visual;
        w.draw( hvisual, n*Nx, n*Ny, colors);
        //step 
        t.tic();
        rk( test, y0, y1, dt);
        thrust::swap(y0, y1);
        t.toc();
        cout << "Timer for one step: "<<t.diff()<<"s\n";
        cout << "STEP"<<++step<<"\t"<<dt<<endl;

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////
    /*
    cout << "Total vorticity is: "<< blas2::dot( stencil.data(), S2D<double, n>(hx, hy), y0) << "\n";
    cout << "Total enstrophy  is "<<blas2::dot( y0, S2D<double, n>(hx, hy), y0)<<"\n";
    blas1::axpby( 1., sol.data(), -1., y0);
    cudaThreadSynchronize();
    cout << "Distance to solution "<<sqrt( blas2::dot( S2D<double, n>(hx, hy), y0))<<endl; //don't forget sqrt when comuting errors
    */

    return 0;

}
