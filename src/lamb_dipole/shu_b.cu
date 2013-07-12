#include <iostream>
#include <iomanip>
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/evaluation.cuh"
#include "dg/rk.cuh"
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
    Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    S2D<double> s2d( grid);
    /////////////////////////////////////////////////////////////////
    //create CUDA context that uses OpenGL textures in Glfw window
    draw::HostWindow w( 600, 600);
    ////////////////////////////////////////////////////////////

    dg::Lamb lamb( p.posX, p.posY, p.R, p.U);
    HVec omega = expand ( lamb, grid);
    DVec stencil = expand( one, grid);
    DVec y0( omega ), y1( y0);
    //make solver and stepper
    Shu<DVec> test( grid, p.D, p.eps);
    AB< k, DVec > ab( y0);

    t.tic();
    test( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil , s2d, y0);
    double enstrophy = 0.5*blas2::dot( y0, s2d, y0);
    double energy =    0.5*blas2::dot( y0, s2d, test.potential()) ;

    double time = 0;
    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::DMatrix equidistant = dg::create::backscatter( grid, LSPACE );
    int running = GL_TRUE;
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( test, y0, p.dt);
    //cout << "Press any key to start!\n";
    double x; 
    //cin >> x;
    while (running && time < p.maxout*p.itstp*p.dt)
    {
        dg::blas2::symv( equidistant, y0, visual);
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        hvisual = visual;
        w.draw( hvisual, p.n*p.Nx, p.n*p.Ny, colors);
        //step 
        t.tic();
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( test, y0, y1, p.dt);
            y0.swap( y1);
            //thrust::swap(y0, y1);
        }
        t.toc();
        //cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        w.title() << "Time "<<time<< " \ttook "<<t.diff()/(double)p.itstp<<"\t per step"<<endl;
        time += p.itstp*p.dt;

        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////
    cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity          is: "<<blas2::dot( stencil , s2d, y0) << "\n";
    cout << "Relative enstrophy error is: "<<(0.5*blas2::dot( s2d, y0) - enstrophy)/enstrophy<<"\n";
    test( y0, y1); //get the potential ready
    cout << "Relative energy error    is: "<<(0.5*blas2::dot( test.potential(), s2d, y0) - energy)/energy<<"\n";

    //blas1::axpby( 1., y0, -1, sol);
    //cout << "Distance to solution: "<<sqrt(blas2::dot( s2d, sol ))<<endl;

    cout << "Press any key to quit!\n";
    cin >> x;
    return 0;

}
