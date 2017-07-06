#include <iostream>
#include <iomanip>
#include <sstream>
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

#include "draw/host_window.h"

#include "shu.cuh"
#include "parameters.h"

double delta =0.05;
double rho =M_PI/15.;
double shearLayer(double x, double y){
    if( y<= M_PI)
        return delta*cos(x) - 1./rho/cosh( (y-M_PI/2.)/rho)/cosh( (y-M_PI/2.)/rho);
    return delta*cos(x) + 1./rho/cosh( (3.*M_PI/2.-y)/rho)/cosh( (3.*M_PI/2.-y)/rho);
}

using namespace std;
using namespace dg;

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Reader reader;
    Json::Value js;
    if( argc == 1)
    {
        std::ifstream is("input/default.json");
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
    /////////////////////////////////////////////////////////////////
    Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    DVec w2d( create::weights(grid));
    /////////////////////////////////////////////////////////////////
    std::stringstream title;
    GLFWwindow* w = draw::glfwInitAndCreateWindow(600, 600, "");
    draw::RenderHostData render( 1,1);
    ////////////////////////////////////////////////////////////

    dg::Lamb lamb( p.posX*p.lx, p.posY*p.ly, p.R, p.U);
    dg::HVec omega; 
    if( p.initial == "lamb")
        omega = dg::evaluate ( lamb, grid);
    else if ( p.initial == "shear")
        omega = dg::evaluate ( shearLayer, grid);

    DVec stencil = evaluate( one, grid);
    DVec y0( omega ), y1( y0);
    //subtract mean mass 
    if( p.bc_x == dg::PER && p.bc_y == dg::PER)
    {
        double meanMass = dg::blas2::dot( y0, w2d, stencil)/(double)(p.lx*p.ly);
        dg::blas1::axpby( -meanMass, stencil, 1., y0);
    }
    //make solver and stepper
    Shu<DMatrix, DVec> shu( grid, p.eps);
    Diffusion<DMatrix, DVec> diffusion( grid, p.D);
    Karniadakis< DVec > ab( y0, y0.size(), p.eps_time);

    Timer t;
    t.tic();
    shu( y0, y1);
    t.toc();
    cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = blas2::dot( stencil , w2d, y0);
    double enstrophy = 0.5*blas2::dot( y0, w2d, y0);
    double energy =    0.5*blas2::dot( y0, w2d, shu.potential()) ;
    
    std::cout << "Total energy:     "<<energy<<"\n";
    std::cout << "Total enstrophy:  "<<enstrophy<<"\n";
    std::cout << "Total vorticity:  "<<vorticity<<"\n";

    double time = 0;
    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    DVec visual( grid.size());
    HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
    ab.init( shu, diffusion, y0, p.dt);
    ab( shu, diffusion, y0); //make potential ready
    //cout << "Press any key to start!\n";
    //double x; 
    //cin >> x;
    while (!glfwWindowShouldClose(w) && time < p.maxout*p.itstp*p.dt)
    {
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
        //step 
        t.tic();
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( shu, diffusion, y0 );
        }
        t.toc();
        //cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        time += p.itstp*p.dt;

    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////
    cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<endl;
    cout << "Analytic formula energy    "<<lamb.energy()<<endl;
    cout << "Total vorticity          is: "<<blas2::dot( stencil , w2d, ab.last()) << "\n";
    cout << "Relative enstrophy error is: "<<(0.5*blas2::dot( w2d, ab.last()) - enstrophy)/enstrophy<<"\n";
    //shu( y0, y1); //get the potential ready
    cout << "Relative energy error    is: "<<(0.5*blas2::dot( shu.potential(), w2d, ab.last()) - energy)/energy<<"\n";

    //blas1::axpby( 1., y0, -1, sol);
    //cout << "Distance to solution: "<<sqrt(blas2::dot( w2d, sol ))<<endl;

    //cout << "Press any key to quit!\n";
    //cin >> x;
    return 0;

}
