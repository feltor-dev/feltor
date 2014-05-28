#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#include "draw/host_window.h"
//#include "draw/device_window.cuh"

#include "feltor.cuh"
#include "bessel.h"
#include "dg/rk.cuh"
#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/karniadakis.cuh"
#include "file/read_input.h"
#include "parameters.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - directly visualizes results on the screen using parameters in window_params.txt
*/


int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> v, v2;
    std::stringstream title;
    if( argc == 1)
    {
        try{
        v = file::read_input("input.txt");
        }catch( toefl::Message& m){m.display();}
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

    /////////////////////////////////////////////////////////////////////////
    const Parameters p( v);
    p.display( std::cout);
    v2 = file::read_input( "window_params.txt");
    GLFWwindow* w = draw::glfwInitAndCreateWindow( p.Nz/v2[2]*v2[3], v2[1]*v2[4], "");
    draw::RenderHostData render(v2[1], p.Nz/v2[2]);

    dg::Grid3d<double > grid( -p.a*(1+1e-1), p.a*(1+1e-1),  -p.a*(1+1e-1), p.a*(1+1e-1), 0, 1., p.n, p.Nx, p.Ny, p.Nz, dg::DIR, dg::DIR, dg::PER);
    //create RHS 
    eule::Feltor< dg::DVec > feltor( grid, p); 
    eule::Rolkar< dg::DVec > rolkar( grid, p.nu_perp, p.nu_parallel, p.a, p.thickness, p.mu_e*4.*M_PI*M_PI*p.R_0*p.R_0);
    //create initial vector
    dg::Vortex init0( 0., p.a - p.posX*p.thickness, 0, p.sigma, p.amp ,2.*M_PI*p.m_par); //gaussian width is in absolute values
    dg::Vortex init1( 0., -p.a + p.posX*p.thickness, 0, p.sigma, p.amp ,2.*M_PI*p.m_par); //gaussian width is in absolute values
    dg::Vortex init2( p.a - p.posX*p.thickness, 0., 0, p.sigma, p.amp ,2.*M_PI*p.m_par); //gaussian width is in absolute values
    dg::Vortex init3( -p.a + p.posX*p.thickness, 0., 0, p.sigma, p.amp ,2.*M_PI*p.m_par); //gaussian width is in absolute values
    eule::Gradient grad( p.a, p.thickness, p.lnn_inner);

    const dg::HVec gradient( dg::evaluate(grad, grid));
    std::vector<dg::DVec> y0(3, dg::evaluate( init0, grid)); // n_e' = gaussian
    std::vector<dg::DVec> y1(3, dg::evaluate( grad, grid)); 
    dg::blas1::axpby( 1., y1[0], 1., y0[0]);
    dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init1, grid), 1., y0[0]);
    dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init2, grid), 1., y0[0]);
    dg::blas1::axpby( 1., (dg::DVec)dg::evaluate(init3, grid), 1., y0[0]);
    dg::blas1::axpby( 1., y1[1], 0., y0[1]);
    dg::blas1::axpby( 0., y1[2], 0., y0[2]); //set U = 0

    //dg::blas2::symv( feltor.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    //dg::blas2::symv( (dg::DVec)dg::create::v2d( grid), y0[1], y0[1]);

    feltor.log( y0, y0, 2); //transform to logarithmic values

    dg::Karniadakis< std::vector<dg::DVec> > ab( y0, y0[0].size(), p.eps_time);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    double time = 0;
    
    ab.init( feltor, rolkar, y0, p.dt);
    const double mass0 = feltor.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = feltor.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while ( !glfwWindowShouldClose( w ))
    {
        //transform field to an equidistant grid
        feltor.exp( y0, y1, 2);
        //thrust::transform( y1[0].begin(), y1[0].end(), dvisual.begin(), dg::PLUS<double>(-1));

        hvisual = y1[0];
        dg::blas1::axpby( -1., gradient, 1., hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ne / "<<colors.scale()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }

        thrust::transform( y1[1].begin(), y1[1].end(), dvisual.begin(), dg::PLUS<double>(-1));
        hvisual = dvisual;
        //dg::HVec iris = dg::evaluate( eule::Iris( p.a, p.thickness), grid);
        //dg::blas1::pointwiseDot( iris, hvisual, hvisual);
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        title << std::setprecision(2) << std::scientific;
        title <<"ni / "<<colors.scale()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }

        //transform phi
        dg::blas2::gemv( rolkar.laplacianM(), feltor.potential()[0], y1[1]);
        hvisual = y1[1];
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        title <<"phi / "<<colors.scale()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }
        hvisual = y0[2];
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        title <<"Ue / "<<colors.scale()<<"\t";
        for( unsigned k=0; k<p.Nz/v2[2];k++)
        {
            unsigned size=grid.n()*grid.n()*grid.Nx()*grid.Ny();
            dg::HVec part( visual.begin() + k*v2[2]*size, visual.begin()+(k*v2[2]+1)*size);
            render.renderQuad( part, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);
        }


        title << std::fixed; 
        title << " &&   time = "<<time;
        glfwSetWindowTitle(w,title.str().c_str());
        title.str("");
        glfwPollEvents();
        glfwSwapBuffers( w);

        //step 
#ifdef DG_BENCHMARK
        t.tic();
#endif//DG_BENCHMARK
        //double x;
        //std::cin >> x;
        for( unsigned i=0; i<p.itstp; i++)
        {
            step++;
            std::cout << "(m_tot-m_0)/m_0: "<< (feltor.mass()-mass0)/mass_blob0<<"\t";
            E0 = E1;
            E1 = feltor.energy();
            diff = (E1 - E0)/p.dt;
            double diss = feltor.energy_diffusion( );
            std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
            std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";

            try{ ab( feltor, rolkar, y0);}
            catch( eule::Fail& fail) { 
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
