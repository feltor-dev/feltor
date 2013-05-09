#include <iostream>
#include <iomanip>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/scatter.h>

#include "arrvec2d.cuh"
#include "operator_matrix.cuh"
#include "blas.h"
#include "dlt.h"
#include "cuda_texture.cuh"
#include "functions.h"
#include "functors.cuh"
#include "evaluation.cuh"

const unsigned n = 3;
const unsigned Nx = 10, Ny = 20;

typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> HVec;
typedef dg::ArrVec2d< double, n, DVec> DArrVec;
typedef dg::ArrVec2d< double, n, HVec> HArrVec;

typedef cusp::ell_matrix<int, double, cusp::host_memory> HMatrix;
typedef cusp::ell_matrix<int, double, cusp::device_memory> DMatrix;

using namespace std;
int main()
{
    //Create Window and set window title
    //dg::Window w( 400, 400);
    dg::HostWindow w( 400, 400);
    glfwSetWindowTitle( "Hello world\n");
    // generate a gaussian
    /*
    dg::Gaussian g( 0.5, 0.5, .1, .1, 1);
    DArrVec vector = dg::expand<dg::Gaussian, n> ( g, 0., 1., 0., 1., Nx, Ny);
    */
    dg::Lamb g( 0.5, 0.5, .1, 1);
    DArrVec vector = dg::expand<dg::Lamb, n> ( g, 0., 1., 0., 1., Nx, Ny);
    cout  << vector<<endl;

    //create equidistant backward transformation
    dg::Operator<double, n> backwardeq( dg::DLT<n>::backwardEQ);
    dg::Operator<double, n*n> backward2d = dg::tensor( backwardeq, backwardeq);
    HMatrix hbackward = dg::tensor( Nx*Ny, backward2d);
    DMatrix backward = hbackward;
    thrust::device_vector<int> map = dg::makePermutationMap<n>( Nx, Ny);

    //allocate storage for visual
    DVec visual( n*n*Nx*Ny);
    HVec hvisual( n*n*Nx*Ny);

    //transform vector to an equidistant grid
    dg::blas2::symv( backward, vector.data(), visual);
    thrust::scatter( visual.begin(), visual.end(), map.begin(), visual.begin());

    //create a colormap
    dg::ColorMapRedBlueExt colors( 1.);
    //compute maximum value as scale
    colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
    std::cout << "Maximum: "<<colors.scale()<<std::endl;
    
    std::cout<< "Colors for x = 0: "<<colors( 0).r << " "<<colors(0).g <<" "<<colors(0).b<< std::endl;

    int running = GL_TRUE;
    while (running)
    {
        hvisual = visual;
        w.draw( hvisual, n*Nx, n*Ny, colors);
        glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    std::cout << "Don't manually close window -> segfault of unknown origin!\n"
              << "Use ESC instead!\n";

    return 0;
}
