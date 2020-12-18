#include <iostream>
#include <iomanip>
#include <sstream>
#include <limits.h>  // UINT_MAX is needed in cusp (v0.5.1) but limits.h is not included
#include <thrust/remove.h>
#include <thrust/host_vector.h>

#include "dg/algorithm.h"
#include "dg/file/json_utilities.h"

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

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    Json::Value js;
    if( argc == 1)
        file::file2Json( "input/default.json", js, file::comments::are_discarded);
    else if( argc == 2)
        file::file2Json( argv[1], js);
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }
    const Parameters p( js);
    p.display( std::cout);
    /////////////////////////////////////////////////////////////////
    dg::Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::DVec w2d( dg::create::weights(grid));
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

    dg::DVec stencil = evaluate( dg::one, grid);
    dg::DVec y0( omega ), y1( y0);
    //subtract mean mass 
    if( p.bc_x == dg::PER && p.bc_y == dg::PER)
    {
        double meanMass = dg::blas2::dot( y0, w2d, stencil)/(double)(p.lx*p.ly);
        dg::blas1::axpby( -meanMass, stencil, 1., y0);
    }
    //make solver and stepper
    shu::Shu<dg::DMatrix, dg::DVec> shu( grid, p.eps);
    //shu::Diffusion<dg::DMatrix, dg::DVec> diffusion( grid, p.D);
    dg::ModalFilter<dg::DMatrix, dg::DVec> filter( 36, 0.5, 8, grid);
    //dg::Karniadakis< dg::DVec > stepper( y0, y0.size(), p.eps_time);
    dg::FilteredExplicitMultistep< dg::DVec > stepper( "eBDF", 3, y0);
    //dg::ShuOsher<dg::DVec> stepper( "SSPRK-3-3", y0);

    dg::Timer t;
    t.tic();
    shu( 0., y0, y1);
    t.toc();
    std::cout << "Time for one rhs evaluation: "<<t.diff()<<"s\n";
    double vorticity = dg::blas2::dot( stencil , w2d, y0);
    double enstrophy = 0.5*dg::blas2::dot( y0, w2d, y0);
    double energy =    0.5*dg::blas2::dot( y0, w2d, shu.potential()) ;
    
    std::cout << "Total energy:     "<<energy<<"\n";
    std::cout << "Total enstrophy:  "<<enstrophy<<"\n";
    std::cout << "Total vorticity:  "<<vorticity<<"\n";

    double time = 0;
    ////////////////////////////////glfw//////////////////////////////
    //create visualisation vectors
    dg::DVec visual( grid.size());
    dg::HVec hvisual( grid.size());
    //transform vector to an equidistant grid
    dg::IDMatrix equidistant = dg::create::backscatter( grid );
    draw::ColorMapRedBlueExt colors( 1.);
    //stepper.init( shu, diffusion, time, y0, p.dt);
    stepper.init( shu, filter, time, y0, p.dt);
    //std::cout << "Press any key to start!\n";
    //double x;
    //cin >> x;
    while (!glfwWindowShouldClose(w) && time < p.maxout*p.itstp*p.dt)
    {
        dg::blas2::symv( equidistant, y0, visual);
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
            //stepper.step( shu, diffusion, time, y0 );
            stepper.step( shu, filter, time, y0 );
            //stepper.step( shu, filter, time, y0, time, y0, p.dt );
        }
        t.toc();
        //std::cout << "Timer for one step: "<<t.diff()/N<<"s\n";
        time += p.itstp*p.dt;

    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////
    std::cout << "Analytic formula enstrophy "<<lamb.enstrophy()<<std::endl;
    std::cout << "Analytic formula energy    "<<lamb.energy()<<std::endl;
    std::cout << "Total vorticity          is: "<<dg::blas2::dot( stencil , w2d, y0) << "\n";
    std::cout << "Relative enstrophy error is: "<<(0.5*dg::blas2::dot( w2d, y0) - enstrophy)/enstrophy<<"\n";
    std::cout << "Relative energy error    is: "<<(0.5*dg::blas2::dot( shu.potential(), w2d, y0) - energy)/energy<<"\n";

    //dg::blas1::axpby( 1., y0, -1, sol);
    //cout << "Distance to solution: "<<sqrt(dg::blas2::dot( w2d, sol ))<<std::endl;

    //cout << "Press any key to quit!\n";
    //cin >> x;
    return 0;

}
