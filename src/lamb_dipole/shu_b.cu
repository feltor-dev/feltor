#include <iostream>
#include <iomanip>
#include <sstream>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/evaluation.cuh"
#include "dg/rk.cuh"
#include "dg/karniadakis.cuh"
#include "dg/xspacelib.cuh"
#include "dg/typedefs.cuh"

#include "draw/host_window.h"

#include "file/read_input.h"

#include "shu.cuh"
#include "parameters.h"



using namespace std;
using namespace dg;
const unsigned k = 3;

int main()
{
    Timer t;
    const Parameters p( file::read_input( "input.txt"));
    p.display();
    if( p.k != k)
    {
        std::cerr << "Time stepper needs recompilation!\n";
        return -1;
    }
    Grid2d<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    DVec w2d( create::w2d(grid));
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
    draw::RenderHostData render( 1,1);
    ////////////////////////////////////////////////////////////

    dg::Lamb lamb( p.posX*p.lx, p.posY*p.ly, p.R, p.U);
    HVec omega = evaluate ( lamb, grid);
    DVec stencil = evaluate( one, grid);
    DVec y0( omega ), y1( y0);
    //make solver and stepper
    Shu<DVec> test( grid, p.D, p.eps);
    Diffusion<DVec> diffusion( grid, p.D);
    Karniadakis< DVec > ab( y0, y0.size(), 1e-8);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil , w2d, y0);
    double enstrophy = 0.5*blas2::dot( y0, w2d, y0);
    double energy =    0.5*blas2::dot( y0, w2d, test.potential()) ;

    double time = 0;
    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::DMatrix equidistant = dg::create::backscatter( grid, XSPACE );
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( test, diffusion, y0, p.dt);
    //cout << "Press any key to start!\n";
    double x; 
    //cin >> x;
    while (!glfwWindowShouldClose(w) && time < p.maxout*p.itstp*p.dt)
    {
        dg::blas2::symv( equidistant, y0, visual);
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        hvisual = visual;
        render.renderQuad( hvisual, p.n*p.Nx, p.n*p.Ny, colors);
        title << "Time "<<time<< " \ttook "<<t.diff()/(double)p.itstp<<"\t per step"<<endl;
        glfwSetWindowTitle(w, title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers(w);
        //step 
        t.tic();
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( test,diffusion, y0 );
        }
        t.toc();
        //cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        time += p.itstp*p.dt;

    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////
    cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity          is: "<<blas2::dot( stencil , w2d, y0) << "\n";
    cout << "Relative enstrophy error is: "<<(0.5*blas2::dot( w2d, y0) - enstrophy)/enstrophy<<"\n";
    test( y0, y1); //get the potential ready
    cout << "Relative energy error    is: "<<(0.5*blas2::dot( test.potential(), w2d, y0) - energy)/energy<<"\n";

    //blas1::axpby( 1., y0, -1, sol);
    //cout << "Distance to solution: "<<sqrt(blas2::dot( w2d, sol ))<<endl;

    //cout << "Press any key to quit!\n";
    //cin >> x;
    return 0;

}
