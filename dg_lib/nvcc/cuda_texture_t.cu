#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>

#include "arrvec2d.cuh"
#include "cuda_texture.cuh"
#include "functions.h"
#include "functors.cuh"
#include "evaluation.cuh"

const unsigned n = 3;
const unsigned Nx = 10, Ny = 10;
typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> HVec;
typedef dg::ArrVec2d< double, n, DVec> DArrVec;
typedef dg::ArrVec2d< double, n, HVec> HArrVec;

int main()
{
    //Create Window and set window title
    dg::Window w( 400, 300);
    glfwSetWindowTitle( "Hello world\n");
    // generate a gaussian and the stencil vector
    dg::Gaussian g( 0.5, 0.5, .1, .1, 1);
    DArrVec vector = dg::expand<dg::Gaussian, n> ( g, 0.,1., 0., 1., Nx, Ny);
    HArrVec stencil = dg::expand< double(&)(double, double), n> ( dg::one, 0., 1., 0., 1., Nx, Ny);
    // show the stencil on terminal
    std::cout << stencil<<std::endl;

    //allocate storage for stencil and visual
    DVec d_stencil(stencil.data());
    DVec visual( Nx*Ny);

    //reduce the gaussian to the 00 values and show them on terminal
    thrust::remove_copy_if( vector.data().begin(), vector.data().end(), d_stencil.begin(), visual.begin(), thrust::logical_not<double>() );
    dg::ArrVec2d<double, 1, HVec> h_visual( visual, Nx);
    std::cout << std::fixed << std::setprecision(2);
    std::cout << h_visual<<std::endl;

    //create a colormap
    dg::ColorMapRedBlueExt colors( 1.);
    //compute maximum value as scale
    colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., thrust::maximum<double>() );

    int running = GL_TRUE;
    while (running)
    {
        w.draw( visual, Nx, Ny, colors);
        glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    std::cout << "Don't manually close window -> segfault of unknown origin!\n"
              << "Use ESC instead!\n";

    return 0;
}
