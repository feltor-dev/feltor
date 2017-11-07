#include <iostream>
#include <iomanip>
#include <sstream>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "draw/host_window.h"

#include "dg/algorithm.h"

#include "shu.cuh"


using namespace std;
using namespace dg;

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;

// const unsigned k = 2;
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
    const unsigned NT = (unsigned)(T*n*Nx/0.1/lx);
    
    Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    DVec w2d( create::weights( grid));
    const double dt = T/(double)NT;
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "Navier Stokes");
    draw::RenderHostData render( 1,1);
    ////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;
    //cout << "# of timesteps              " << NT << endl;
    cout << "Diffusion                   " << D <<endl;
    dg::Lamb lamb( 0.5*lx, 0.5*ly, 0.2*lx, 1);
    HVec omega = evaluate ( lamb, grid);
    DVec stencil = evaluate( one, grid);
    DVec y0( omega);
    Shu<dg::DMatrix, dg::DVec> test( grid, eps);
    Diffusion<DMatrix, DVec> diffusion( grid, D);
    Karniadakis< DVec > ab( y0, y0.size(), 1e-8);

    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( test, diffusion, y0, dt);
    while (!glfwWindowShouldClose(w))
    {
        //transform field to an equidistant grid
        cout << "Total vorticity is: "<<blas2::dot( stencil, w2d, y0) << "\n";
        cout << "Total enstrophy is: "<<blas2::dot( w2d, y0)<<"\n";
        //compute the color scale
        dg::blas2::symv( equidistant, y0, visual );
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<float>() );
        std::cout << "Color scale " << colors.scale() <<"\n";
        //draw and swap buffers
        dg::blas1::transfer(visual, hvisual);
        render.renderQuad( hvisual, n*Nx, n*Ny, colors);
        //step 
        ab( test,diffusion, y0 );

        glfwSwapBuffers(w);
        glfwWaitEvents();
    }
    glfwTerminate();
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
