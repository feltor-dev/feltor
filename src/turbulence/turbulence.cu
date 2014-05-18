#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>

#include "draw/host_window.h"

#include "turbulence.cuh"
#include "dg/karniadakis.cuh"
#include "dg/timer.cuh"
#include "file/read_input.h"
#include "parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/

const unsigned k = 3; //!< a change of k needs a recompilation!

int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> v, v2;
    if( argc == 1)
    {

        try{v = file::read_input("input.txt");}catch( toefl::Message& m){m.display();}
    }
    else if( argc == 2)
    {
        v = file::read_input( argv[1]);
    }
    else
    {
        std::cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }

    v2 = file::read_input( "window_params.txt");
    GLFWwindow * w = draw::glfwInitAndCreateWindow( v2[2]*v2[3], v2[1]*v2[4], "");
    draw::RenderHostData render( v2[1], v2[2]);
    std::stringstream title;
    /////////////////////////////////////////////////////////////////////////
    const Parameters p( v);
    p.display( std::cout);
    if( p.k != k)
    {
        std::cerr << "ERROR: k doesn't match: "<<k<<" vs. "<<p.k<<"\n";
        return -1;
    }

    dg::Grid2d<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    grid.display();
    //create RHS 
    dg::Turbulence< dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, p.gradient, p.d); 
    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian

    dg::blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' 
    dg::blas2::symv( (dg::DVec)dg::create::v2d( grid), y0[1], y0[1]);

    dg::Karniadakis< std::vector<dg::DVec> > ab( y0, grid.size(), 1e-9);
    dg::Diffusion< dg::DVec> diff( grid, p.nu);

    dg::DVec dvisual( grid.size(), 0.);
    const dg::DVec gradient =  dg::evaluate( dg::LinearX( -p.gradient, 1+p.gradient*grid.lx()), grid);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    ab.init( test, diff, y0, p.dt);
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while (!glfwWindowShouldClose(w))
    {
        //transform field to an equidistant grid
        dvisual = y0[0];
        //dg::blas1::axpby( 1., dvisual, 1., gradient, dvisual);
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform field to an equidistant grid
        dg::blas1::axpby( 1., test.potential()[0], -0., test.potential()[1], dvisual);
        //dg::blas2::gemv( test.polarisationM(), test.potential()[0], dvisual);
        //dvisual = y0[1];
        //dg::blas1::axpby( 1., dvisual, 1., gradient, dvisual);
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"potential / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform field to an equidistant grid
        test.arakawa().bracket( test.potential()[0], test.potential()[0], dvisual);
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"v_E^2 / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform field to an equidistant grid
        //test.arakawa().bracket( y0[1], test.potential()[0], dvisual);
        dg::blas2::gemv( test.arakawa().dy(), test.potential()[0], dvisual);
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"dy phi / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform field to an equidistant grid
        dg::blas2::gemv( test.arakawa().dx(), test.potential()[0], dvisual);
        //dg::blas1::axpby( 1., gradient, 1., y0[1]);
        //dg::blas1::pointwiseDot( dvisual, y0[1], dvisual);
        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"dx phi / "<<colors.scale()<<"\t";
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);


        //transform phi
        dg::blas2::gemv( test.polarisationM(), test.potential()[0], dvisual);
        //dg::blas1::axpby(1, test.potential()[0], 0, dvisual);

        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        title <<"div( Ngrad phi) / "<<colors.scale()<<"\t";
        title << std::fixed; 
        title << " &&   time = "<<time;
        render.renderQuad( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        glfwSetWindowTitle( w, title.str().c_str());
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
            try{ ab( test, diff, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                glfwSetWindowShouldClose(w, GL_TRUE);
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
