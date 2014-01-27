#include <iostream>
#include <iomanip>
#include <vector>
#include <thrust/host_vector.h>

#include "draw/host_window.h"

#include "evaluation.cuh"
#include "functions.h"
#include "functors.cuh"
#include "toefl.cuh"
#include "rk.cuh"
#include "xspacelib.cuh"
#include "typedefs.cuh"


using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned Nx = 100;
const unsigned Ny = 20;
const double lx = 5.;
const double ly = 1.;

const double Pr = 10;
const double Ra = 5e5;

const unsigned k = 3;
const double dt = 2e-7;
const unsigned N = 10; //steps between output

double eps = 1e-3;

using namespace std;

double groundState( double x, double y) { return ly/2. - y;}

int main()
{
    draw::HostWindow w(lx*200, 200);
    glfwSetWindowTitle( "Behold the convection\n");


    /////////////////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;

    //create initial vector
    const Grid<double> grid( 0, lx, 0, ly,n, Nx, Ny, dg::PER, dg::DIR);
    dg::Gaussian gaussian( 1., ly/2., .1, .1, 1);
    dg::DVec theta = dg::evaluate ( gaussian, grid);
    vector<dg::DVec> y0(2, theta), y1(y0);
    y0[1] = dg::DVec( grid.size(), 0.); //omega is zero

    //create RHS and AB
    Toefl< dg::DVec> test( grid, Ra, Pr, eps); 
    AB< k, vector<dg::DVec> > ab( y0);


    //create visualisation vectors
    int running = GL_TRUE;
    dg::DVec visual(  grid.size());
    dg::HVec hvisual( grid.size());
    dg::DVec ground = evaluate ( groundState, grid), temperature( ground);
    dg::DMatrix equidistant = dg::create::backscatter( grid, XSPACE );
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( test, y0, dt);
    double time = 0;
    while (running)
    {
        //compute the total temperature
        blas1::axpby( 1., y0[0],  0., temperature);
        blas1::axpby( 1., ground, 1., temperature);
        //transform field to an equidistant grid
        dg::blas2::symv( equidistant, temperature, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        w.title() << " temperature / "<<colors.scale() <<" time "<<time;
        //std::cout << "Color scale " << colors.scale() <<"\n";
        //draw and swap buffers
        hvisual = visual;
        w.draw( hvisual, n*grid.Nx(), n*grid.Ny(), colors);
        //step 
        for( unsigned i=0; i<N; i++)
        {
            ab( test, y0, y1, dt);
            y0.swap( y1);
            time += (double)N*dt;
        }
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////

    return 0;

}
