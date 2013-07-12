#include <iostream>
#include <iomanip>

#include "draw/host_window.h"
#include "xspacelib.cuh"


const unsigned n = 3;
const unsigned Nx = 40, Ny = 40;

using namespace std;

int main()
{
    //Create Window and set window title
    draw::HostWindow w( 400, 400);
    glfwSetWindowTitle( "Hello world\n");
    //generate a grid 
    dg::Grid<double> grid( 0, 1., 0, 1., n, Nx, Ny);
    // generate a vector on the grid to visualize 
    //dg::Gaussian g( 0.5, 0.5, .1, .1, 1);
    dg::Lamb g( 0.5*grid.lx(), 0.5*grid.ly(), .3, 1);
    dg::HVec vector = dg::evaluate(g, grid);

    //allocate storage for visual
    dg::HVec visual( grid.size());
    //transform vector to an equidistant grid
    dg::Matrix equidistant = dg::create::backscatter( grid);
    dg::blas2::mv( equidistant, vector, visual );

    //create a colormap
    draw::ColorMapRedBlueExt colors( 1.);
    //compute maximum value as scale
    colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
    std::cout << "Color scale: "<<colors.scale()<<std::endl;
    std::cout<< "Colors for x = 0: "<<colors( 0).r << " "<<colors(0).g <<" "<<colors(0).b<< std::endl;

    int running = GL_TRUE;
    while (running)
    {
        w.draw( visual, n*Nx, n*Ny, colors);
        glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }

    return 0;
}
