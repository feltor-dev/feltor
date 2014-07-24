#include <iostream>
#include <vector>
#include "xspacelib.cuh"
#include <fstream>
#include "draw/host_window.h"
// #include "draw/device_window.cuh"
int main()
{
    unsigned Nx=100, Ny=100,polcoeff=3;
    double gamma=30., eddysize=30.,Rmin,Zmin,Rmax,Zmax;
    double Nxh = Nx/2.,Nyh=Ny/2.,R0=1000.;
    double amplitude=0.5; 
    Rmin=R0-Nxh;
    Zmin=-Nyh;
    Rmax=R0+Nxh; 
    Zmax=Nyh;
    std::cout << "test the bath initialisation" <<"\n";

    //make dggrid
    dg::Grid2d<double> grid(Rmin,Rmax,Zmin,Zmax, polcoeff,Nx,Ny,dg::PER,dg::PER);
    //construct bathRZ
    dg::BathRZ bathRZ(32, 32, 1, Rmin,Zmin, gamma,eddysize,amplitude);
    //evaluate bathRZ on the dggrid on a hvector
    dg::HVec hvisual = dg::evaluate( bathRZ, grid);
    //allocate mem for visual
    dg::HVec visual( grid.size());
    //make equidistant grid from dggrid
    dg::HMatrix equigrid = dg::create::backscatter(grid);
    //evaluate on valzues from devicevector on equidistant visual hvisual vector
    dg::blas2::gemv( equigrid, hvisual, visual);

    //Create Window and set window title
    GLFWwindow* w = draw::glfwInitAndCreateWindow( 400, 400, "Random field");
    draw::RenderHostData render( 1, 1);
    // generate a vector on the grid to visualize 
  
    //create a colormap
    draw::ColorMapRedBlueExt colors( 1.);
    //set scale
    colors.scale() =  1.;
    while (!glfwWindowShouldClose( w ))
    {
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        glfwSwapBuffers(w);
        glfwWaitEvents();
    }
    glfwTerminate();
 return 0;
}