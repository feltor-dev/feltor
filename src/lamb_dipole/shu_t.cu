#include <iostream>
#include <iomanip>
#include <sstream>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "draw/host_window.h"

#include "dg/algorithm.h"

#include "shu.cuh"


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
    std::cout << "Type n, Nx, Ny and eps!\n";
    std::cin >> n >> Nx >> Ny>>eps;
    const unsigned NT = (unsigned)(T*n*Nx/0.1/lx);

    dg::Grid2d grid( 0, lx, 0, ly, n, Nx, Ny, dg::PER, dg::PER);
    dg::DVec w2d( dg::create::weights( grid));
    const double dt = T/(double)NT;
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "Navier Stokes");
    draw::RenderHostData render( 1,1);
    ////////////////////////////////////////////////////////////
    std::cout << "# of Legendre coefficients: " << n<<std::endl;
    std::cout << "# of grid cells:            " << Nx*Ny<<std::endl;
    std::cout << "Timestep                    " << dt << std::endl;
    //std::cout << "# of timesteps              " << NT << std::endl;
    std::cout << "Diffusion                   " << D <<std::endl;
    dg::Lamb lamb( 0.5*lx, 0.5*ly, 0.2*lx, 1);
    dg::HVec omega = evaluate ( lamb, grid);
    dg::DVec stencil = evaluate( dg::one, grid);
    dg::DVec y0( omega);
    shu::Shu<dg::DMatrix, dg::DVec> test( grid, eps);
    //shu::Diffusion<DMatrix, DVec> diffusion( grid, D);
    dg::ModalFilter<dg::DMatrix, dg::DVec> filter( 36, 0.5, 8, grid);
    //dg::Karniadakis< dg::DVec > stepper( y0, y0.size(), 1e-8);
    dg::FilteredExplicitMultistep< dg::DVec > stepper( "TVB",3, y0);

    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    dg::DVec visual( grid.size());
    dg::HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
    double time = 0;
    //stepper.init( test, diffusion, time, y0, dt);
    stepper.init( test, filter, time, y0, dt);
    while (!glfwWindowShouldClose(w))
    {
        //transform field to an equidistant grid
        std::cout << "Total vorticity is: "<<dg::blas2::dot( stencil, w2d, y0) << "\n";
        std::cout << "Total enstrophy is: "<<dg::blas2::dot( w2d, y0)<<"\n";
        //compute the color scale
        dg::blas2::symv( equidistant, y0, visual );
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<float>() );
        std::cout << "Color scale " << colors.scale() <<"\n";
        //draw and swap buffers
        dg::blas1::transfer(visual, hvisual);
        render.renderQuad( hvisual, n*Nx, n*Ny, colors);
        //step
        //stepper.step( test,diffusion, time, y0 );
        stepper.step( test,filter, time, y0 );

        glfwSwapBuffers(w);
        glfwWaitEvents();
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////
    std::cout << "Total vorticity is: "<< dg::blas2::dot( stencil, w2d, y0) << "\n";
    std::cout << "Total enstrophy  is "<<dg::blas2::dot( y0, w2d, y0)<<"\n";
    //dg::blas1::axpby( 1., sol.data(), -1., y0);
    //std::cout << "Distance to solution "<<sqrt( dg::blas2::dot( w2d, y0))<<std::endl; //don't forget sqrt when comuting errors

    return 0;

}
