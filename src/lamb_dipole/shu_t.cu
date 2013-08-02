#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "draw/host_window.h"

#include "dg/functors.cuh"
#include "dg/arrvec2d.cuh"
#include "dg/evaluation.cuh"
#include "dg/xspacelib.cuh"
#include "dg/rk.cuh"
#include "dg/typedefs.cuh"

#include "shu.cuh"


using namespace std;
using namespace dg;

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

const unsigned k = 2;
const double D = 0.01;
const double T = 1.;


double initial( double x, double y){return 2.*sin(x)*sin(y);}
double solution( double x, double y) {return 2.*sin(x)*sin(y)*exp( -2.*T*D);}


int main()
{
    unsigned n, Nx, Ny;
    double eps;
    cout << "Type n, Nx, Ny and eps!\n";
    cin >> n >> Nx >> Ny>>eps;
    const unsigned NT = (unsigned)(D*T*n*n*Nx*Nx/0.01/lx/lx);
    
    Grid<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    DVec w2d( create::w2d( grid));
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
    dg::Lamb lamb( 0.5*lx, 0.5*ly, 0.2*lx, 1);
    HVec omega = evaluate ( lamb, grid);
    DVec stencil = evaluate( one, grid);
    //DArrVec sol = evaluate< double(&)(double, double), n> ( solution, 0, lx, 0, ly, Nx, Ny);
    DVec y0( omega), y1( y0);
    Shu<DVec> test( grid, D, eps);
    AB< k, DVec > ab( y0);

    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::DMatrix equidistant = dg::create::backscatter( grid, LSPACE );
    int running = GL_TRUE;
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( test, y0, dt);
    while (running)
    {
        //transform field to an equidistant grid
        cout << "Total vorticity is: "<<blas2::dot( stencil, w2d, y0) << "\n";
        cout << "Total enstrophy is: "<<blas2::dot( w2d, y0)<<"\n";
        //compute the color scale
        dg::blas2::mv( equidistant, y0, visual );
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<float>() );
        std::cout << "Color scale " << colors.scale() <<"\n";
        //draw and swap buffers
        hvisual = visual;
        w.draw( hvisual, n*Nx, n*Ny, colors);
        //step 
        ab( test, y0, y1, dt);
        //thrust::swap(y0, y1);
        y0.swap( y1);

        glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////
    /*
    cout << "Total vorticity is: "<< blas2::dot( stencil, w2d, y0) << "\n";
    cout << "Total enstrophy  is "<<blas2::dot( y0, w2d, y0)<<"\n";
    blas1::axpby( 1., sol.data(), -1., y0);
    cudaThreadSynchronize();
    cout << "Distance to solution "<<sqrt( blas2::dot( w2d, y0))<<endl; //don't forget sqrt when comuting errors
    */

    return 0;

}
