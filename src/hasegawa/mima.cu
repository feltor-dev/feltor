#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "mima.cuh"
#include "dg/multistep.h"
#include "dg/backend/timer.cuh"
#include "../toefl/parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/

int main( int argc, char* argv[])
{
    ////Parameter initialisation ////////////////////////////////////////////
    std::stringstream title;
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
    /////////glfw initialisation ////////////////////////////////////////////
    std::ifstream is( "window_params.js");
    reader.parse( is, js, false);
    is.close();
    GLFWwindow* w = draw::glfwInitAndCreateWindow( js["width"].asDouble(), js["height"].asDouble(), "");
    draw::RenderHostData render(js["rows"].asDouble(), js["cols"].asDouble());
    /////////////////////////////////////////////////////////////////////////
    dg::Grid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    bool mhw = ( p.equations == "modified");
    dg::Mima< dg::DMatrix, dg::DVec > mima( grid, p.kappa, p.eps_pol, mhw); 
    dg::DVec one( grid.size(), 1.);
    //create initial vector
    dg::Gaussian gaussian( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.amp); //gaussian width is in absolute values
    dg::Vortex vortex( p.posX*grid.lx(), p.posY*grid.ly(), 0, p.sigma, p.amp);

    dg::DVec phi = dg::evaluate( vortex, grid), omega( phi), y0(phi), y1(phi);
    dg::Elliptic<dg::CartesianGrid2d, dg::DMatrix, dg::DVec> laplaceM( grid);
    dg::blas2::gemv( laplaceM, phi, omega);
    dg::blas1::axpby( 1., phi, 1., omega, y0);

    dg::DVec w2d( dg::create::weights( grid));
    if( p.bc_x == dg::PER && p.bc_y == dg::PER)
    {
        double meanMass = dg::blas2::dot( y0, w2d, one)/(double)(p.lx*p.ly);
        std::cout << "Mean Mass is "<<meanMass<<"\n";
        dg::blas1::axpby( -meanMass, one, 1., y0);
    }
    dg::Karniadakis<dg::DVec > ab( y0, y0.size(), p.eps_time);
    dg::Diffusion<dg::DMatrix,dg::DVec> diffusion( grid, p.nu);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::IHMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    ab.init( mima, diffusion, y0, p.dt);
    ab( mima, diffusion, y0);
    //y0.swap( y1); 
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while ( !glfwWindowShouldClose( w ))
    {
        if( p.bc_x == dg::PER && p.bc_y == dg::PER)
        {
            double meanMass = dg::blas2::dot( y0, w2d, one)/(double)(p.lx*p.ly);
            std::cout << "Mean Mass is "<<meanMass<<"\n";
        }
        //transform field to an equidistant grid
        dvisual = mima.potential();

        dg::blas1::transfer( dvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //transform phi
        dg::blas2::gemv( laplaceM, mima.potential(), y1);
        dg::blas1::transfer( y1, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        title <<"omega / "<<colors.scale()<<"\t";
        title << std::fixed;
        title << " &&   time = "<<time;
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers( w);

        //step 
#ifdef DG_BENCHMARK
        t.tic();
#endif//DG_BENCHMARK
        for( unsigned i=0; i<p.itstp; i++)
        {
            step++;
            if( p.bc_x == dg::PER && p.bc_y == dg::PER)
            {
                double meanMass = dg::blas2::dot( y0, w2d, one)/(double)(p.lx*p.ly);
                dg::blas1::axpby( -meanMass, one, 1., y0);
                meanMass = dg::blas2::dot( y0, w2d, one)/(double)(p.lx*p.ly);
                dg::blas1::axpby( -meanMass, one, 1., y0);
            }

            try{ ab( mima, diffusion, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose( w, GL_TRUE);
                break;
            }
        }
        time += (double)p.itstp*p.dt;
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "\n\t Step "<<step;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp<<"s\n\n";
#endif//DG_BENCHMARK
    }
    glfwTerminate();
    ////////////////////////////////////////////////////////////////////

    return 0;

}
