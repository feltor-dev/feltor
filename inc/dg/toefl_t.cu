#include <iostream>
#include <iomanip>
#include <vector>
#include <thrust/host_vector.h>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "backend/evaluation.cuh"
#include "backend/functions.h"
#include "functors.h"
#include "toefl.cuh"
#include "multistep.h"
#include "backend/xspacelib.cuh"
#include "backend/typedefs.cuh"


const unsigned n = 3;
const unsigned Nx = 20;
const unsigned Ny = 10;
const double lx = 2.;
const double ly = 1.;

const double Pr = 10;
const double Ra = 5e5;

const unsigned k = 2;
const double dt = 1e-6;

double groundState( double x, double y) { return ly/2. - y;}

int main()
{
    GLFWwindow* w = draw::glfwInitAndCreateWindow(800, 400, "Behold the convection");
    draw::RenderHostData render(1,1);
    /////////////////////////////////////////////////////////////////////////
    std::cout << "# of Legendre coefficients: " << n<<std::endl;
    std::cout << "# of grid cells:            " << Nx*Ny<<std::endl;
    std::cout << "Timestep                    " << dt << std::endl;

    //create initial vector
    const dg::Grid2d<double> grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::DIR);
    dg::Gaussian gaussian( lx/2., ly/2., .1, .1, 1);
    dg::DVec theta = dg::evaluate ( gaussian, grid);
    std::vector<dg::DVec> y0(2, theta);
    y0[1] = dg::DVec( grid.size(), 0.); //omega is zero

    //create RHS and RK
    dg::Toefl<dg::cartesian::Grid2d, dg::DMatrix, dg::DVec> test( grid, Ra, Pr, 1e-6); 
    dg::AB< k, std::vector<dg::DVec> > ab( y0);


    //create visualisation vectors
    dg::DVec dvisual(  grid.size());
    dg::HVec hvisual( grid.size());
    dg::DVec ground = dg::evaluate ( groundState, grid), temperature( ground);
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( test, y0, dt);
    while (!glfwWindowShouldClose(w))
    {
        //compute the total temperature
        dg::blas1::axpby( 1., y0[0],  0., temperature);
        dg::blas1::axpby( 1., ground, 1., temperature);
        //transform field to an equidistant grid
        dg::blas2::symv( equidistant, temperature, dvisual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( dvisual.begin(), dvisual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        dg::blas1::transfer( dvisual, hvisual);
        render.renderQuad( hvisual, n*grid.Nx(), n*grid.Ny(), colors);
        glfwSwapBuffers(w);
        glfwPollEvents();
        //step 
        ab( test, y0);
    }
    ////////////////////////////////////////////////////////////////////
    glfwTerminate();

    return 0;

}
