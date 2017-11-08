#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits.h>  // UINT_MAX is needed in cusp (v0.5.1) but limits.h is not included
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/backend/timer.cuh"
#include "dg/backend/evaluation.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/runge_kutta.h"
#include "dg/multistep.h"
#include "dg/helmholtz.h"
#include "dg/backend/typedefs.cuh"
#include "dg/functors.h"

#include "geometries/geometries.h"

#ifdef OPENGL_WINDOW
#include "draw/host_window.h"
#endif

#include "ns.h"
#include "parameters.h"

using namespace std;
using namespace dg;

#ifdef LOG_POLAR
    typedef dg::geo::LogPolarGenerator Generator;
#else
    typedef dg::geo::PolarGenerator Generator;
#endif

int main(int argc, char* argv[])
{
    Timer t;
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Reader reader;
    Json::Value js;
    if( argc == 1)
    {
        std::ifstream is("input.json");
        reader.parse(is,js,false);
    }
    else if( argc == 2)
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Parameters p( js);
    p.display( std::cout);

    //Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    Generator generator(p.r_min, p.r_max); // Generator is defined by the compiler
    dg::geo::CurvilinearGrid2d grid( generator, p.n, p.Nx, p.Ny, dg::DIR, dg::PER);

    DVec w2d( create::volume(grid));

#ifdef OPENGL_WINDOW
    //create CUDA context that uses OpenGL textures in Glfw window
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
    draw::RenderHostData render( 1,1);
#endif

    dg::Lamb lamb( p.posX, p.posY, p.R, p.U);
    HVec omega = evaluate ( lamb, grid);
#ifdef LOG_POLAR
    DVec stencil = evaluate( one, grid);
#else
    DVec stencil = evaluate( LinearX(1.0, p.r_min), grid);
#endif
    DVec y0( omega ), y1( y0);

    //make solver and stepper
    polar::Explicit<aGeometry2d, DMatrix, DVec> shu( grid, p.eps);
    polar::Diffusion<aGeometry2d, DMatrix, DVec> diffusion( grid, p.nu);
    Karniadakis< DVec > ab( y0, y0.size(), p.eps_time);

    t.tic();
    shu( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";

    double vorticity = blas2::dot( stencil , w2d, y0);
    DVec ry0(stencil);
    blas1::pointwiseDot( stencil, y0, ry0);
    double enstrophy = 0.5*blas2::dot( ry0, w2d, y0);
    double energy =    0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    cout << "Total vorticity:  "<<vorticity<<"\n";
    cout << "Total enstrophy:  "<<enstrophy<<"\n";
    cout << "Total energy:     "<<energy<<"\n";

    double time = 0;
#ifdef OPENGL_WINDOW
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
#endif

    ab.init( shu, diffusion, y0, p.dt);
    ab( shu, diffusion, y0); //make potential ready

    t.tic();
    while (time < p.maxout*p.itstp*p.dt)
    {
#ifdef OPENGL_WINDOW
        if(glfwWindowShouldClose(w))
            break;

        dg::blas2::symv( equidistant, ab.last(), visual);
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        dg::blas1::transfer( visual, hvisual);
        render.renderQuad( hvisual, p.n*p.Nx, p.n*p.Ny, colors);
        title << "Time "<<time<< " \ttook "<<t.diff()/(double)p.itstp<<"\t per step";
        glfwSetWindowTitle(w, title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers(w);
#endif

        //step 
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( shu, diffusion, y0 );
        }
        time += p.itstp*p.dt;

    }
    t.toc();

#ifdef OPENGL_WINDOW
    glfwTerminate();
#endif


    double vorticity_end = blas2::dot( stencil , w2d, ab.last());
    blas1::pointwiseDot( stencil, ab.last(), ry0);
    double enstrophy_end = 0.5*blas2::dot( ry0, w2d, ab.last());
    double energy_end    = 0.5*blas2::dot( ry0, w2d, shu.potential()) ;
    cout << "Vorticity error           :  "<<vorticity_end-vorticity<<"\n";
    cout << "Enstrophy error (relative):  "<<(enstrophy_end-enstrophy)/enstrophy<<"\n";
    cout << "Energy error    (relative):  "<<(energy_end-energy)/energy<<"\n";

    cout << "Runtime: " << t.diff() << endl;

    return 0;
}
