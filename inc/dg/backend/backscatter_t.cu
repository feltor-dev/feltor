#include <iostream>
#include <iomanip>

#include "draw/host_window.h"
#include "xspacelib.cuh"
#include "../functors.h"
#include "../blas2.h"

namespace dg{
typedef thrust::host_vector<double> HVec;
typedef cusp::coo_matrix<int, double, cusp::host_memory> Matrix;
}

const unsigned n = 3;
const unsigned Nx = 40, Ny = 40;

int main()
{
    //Create Window and set window title
    GLFWwindow* w = draw::glfwInitAndCreateWindow( 400, 400, "Hello world!");
    draw::RenderHostData render( 1, 1);
    //generate a grid 
    dg::Grid2d grid( 0, 1., 0, 1., n, Nx, Ny);
    // generate a vector on the grid to visualize 
    //dg::Gaussian g( 0.5, 0.5, .1, .1, 1);
    dg::Lamb g( 0.5*grid.lx(), 0.5*grid.ly(), .3, 1);
    dg::HVec vector = dg::evaluate(g, grid);

    //allocate storage for visual
    dg::HVec visual( grid.size());
    //transform vector to an equidistant grid
    dg::Matrix equidistant = dg::create::backscatter( grid);
    dg::blas2::gemv( equidistant, vector, visual );

    //create a colormap
    draw::ColorMapRedBlueExt colors( 1.);
    //compute maximum value as scale
    colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
    std::cout << "Color scale: "<<colors.scale()<<std::endl;
    std::cout<< "Colors for x = 0: "<<colors( 0).r << " "<<colors(0).g <<" "<<colors(0).b<< std::endl;

    while (!glfwWindowShouldClose( w ))
    {
        render.renderQuad( visual, n*Nx, n*Ny, colors);
        render.renderQuad( visual, n*Nx, n*Ny, colors);
        glfwSwapBuffers(w);
        glfwWaitEvents();
    }
    glfwTerminate();

    return 0;
}
