#include <iostream>
#include <vector>
#include <sstream>
#include "blas.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/projection.cuh"
#include "dg/backend/typedefs.cuh"
#include "functors.h"
#include "draw/host_window.h"

int main()
{
    unsigned n=3, Nx=100, Ny=100;
    unsigned n_out, Nx_out, Ny_out;
    std::cout << "Type n_out, Nx_out, Ny_out for interpolation(n, Nx, Ny = 3,100,100)\n";
    std::cin >> n_out>> Nx_out >> Ny_out;
    double gamma=30., eddysize=15.;
    double Rmin,Zmin,Rmax,Zmax;
    double Nxh = Nx/2.,Nyh=Ny/2.,R0=150.;
    double amplitude=0.1; 
    Rmin=R0-Nxh;
    Zmin=-Nyh;
    Rmax=R0+Nxh; 
    Zmax=Nyh;
    std::cout << "test the bath initialisation" <<"\n";
    std::stringstream title;

    //make dggrid and interpolation
    dg::Grid2d grid_old(Rmin,Rmax,Zmin,Zmax, n,Nx,Ny,dg::PER,dg::PER);
    dg::Grid2d grid_new(Rmin,Rmax,Zmin,Zmax, n_out,Nx_out,Ny_out,dg::PER,dg::PER);
    //dg::HMatrix interpolate = dg::create::interpolation( grid_new, grid_old);
    dg::IHMatrix interpolate = dg::create::projection( grid_old, grid_new);
    //construct bathRZ
    dg::BathRZ bathRZ(16, 16, 1, Rmin,Zmin, gamma,eddysize,amplitude);
    //evaluate bathRZ on the dggrid on a hvector
    dg::HVec hvisual_old = dg::evaluate( bathRZ, grid_old);
    dg::HVec hvisual_new( grid_new.size());
    dg::blas2::gemv( interpolate, hvisual_old, hvisual_new);
    dg::HVec w2d = dg::create::weights( grid_old);
    //allocate mem for visual
    dg::HVec visual_old( grid_old.size());
    dg::HVec visual_new( grid_new.size());
    //make equidistant trafo matrix from dggrid
    dg::IHMatrix equigrid_old = dg::create::backscatter(grid_old);
    dg::IHMatrix equigrid_new = dg::create::backscatter(grid_new);
    //evaluate on values from devicevector on equidistant visual hvisual vector
    dg::blas2::gemv( equigrid_old, hvisual_old, visual_old);
    dg::blas2::gemv( equigrid_new, hvisual_new, visual_new);

    //Create Window and set window title
    GLFWwindow* w = draw::glfwInitAndCreateWindow( 800, 400, "Random field");
    draw::RenderHostData render( 1, 2);
    // generate a vector on the grid to visualize 
      //create a colormap
    draw::ColorMapRedBlueExtMinMax colors(-1.0, 1.0);
    while (!glfwWindowShouldClose( w ))
    {

        colors.scalemax() = (float)thrust::reduce( visual_old.begin(), visual_old.end(),-100., thrust::maximum<double>()  );
        colors.scalemin() = (float)thrust::reduce( visual_old.begin(), visual_old.end(), colors.scalemax(), thrust::minimum<double>()  );
        title <<"bath / "<<colors.scalemin()<<"  " << colors.scalemax()<<"\t";

        render.renderQuad( visual_old, grid_old.n()*grid_old.Nx(), grid_old.n()*grid_old.Ny(), colors);
        render.renderQuad( visual_new, grid_new.n()*grid_new.Nx(), grid_new.n()*grid_new.Ny(), colors);
        title << std::fixed; 
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwSwapBuffers(w);
        glfwWaitEvents();
    }
    glfwTerminate();
    return 0;
}
