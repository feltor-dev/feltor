#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"

#include "esel.cuh"
#include "dg/rk.cuh"
#include "dg/timer.cuh"
#include "file/read_input.h"
#include "../galerkin/parameters.h"

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
        v = file::read_input("input.txt");
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
    draw::HostWindow w(v2[3], v2[4]);
    w.set_multiplot( v2[1], v2[2]);
    /////////////////////////////////////////////////////////////////////////
    const Parameters p( v);
    p.display( std::cout);
    if( p.k != k)
    {
        std::cerr << "ERROR: k doesn't match: "<<k<<" vs. "<<p.k<<"\n";
        return -1;
    }

    dg::Grid<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    dg::SOL sol( v[22], v[23], v[24], v[25]);
    //create RHS 
    dg::Esel< dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, sol); 
    //create initial vector
    dg::EXPX<double> exp( 1., -1./v[26]); 
    dg::Gaussian gaussian( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(2, dg::evaluate( exp, grid)), y1(y0); 
    std::vector<dg::DVec> y0p(2, dg::evaluate( gaussian, grid)); 
    dg::blas1::axpby( 1, y0p, 1, y0);

    //dg::blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    dg::blas2::symv( (dg::DVec)dg::create::v2d( grid), y0[1], y0[1]);
    assert( p.global);
    assert( p.bc_x == dg::DIR_NEU);

    test.log( y0, y0); //transform to logarithmic values

    dg::AB< k, std::vector<dg::DVec> > ab( y0);
    //dg::TVB< std::vector<dg::DVec> > ab( y0);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    dg::Timer t;
    bool running = true;
    double time = 0;
    ab.init( test, y0, p.dt);
    ab( test, y0, y1, p.dt);
    y0.swap( y1); 
    const double mass0 = test.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = test.energy(), energy0 = E0, E1 = 0, diff = 0;
    std::cout << "Begin computation \n";
    std::cout << std::scientific << std::setprecision( 2);
    unsigned step = 0;
    while (running)
    {
        //transform field to an equidistant grid
        {
            //test.exp( y1, y1); //plot logarithmic values
            thrust::transform( y1[0].begin(), y1[0].end(), dvisual.begin(), dg::PLUS<double>(1));
        }

        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        w.title() << std::setprecision(2) << std::scientific;
        w.title() <<"ne / "<<colors.scale()<<"\t";
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //transform phi
        //dg::blas2::gemv( test.laplacianM(), test.potential()[0], y1[1]);
        dg::blas1::axpby( 1., test.potential()[0], 0, y1[1]);
        hvisual = y1[1];
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        w.title() <<"omega / "<<colors.scale()<<"\t";
        w.title() << std::fixed; 
        w.title() << " &&   time = "<<time;
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //step 
#ifdef DG_BENCHMARK
        t.tic();
#endif//DG_BENCHMARK
        for( unsigned i=0; i<p.itstp; i++)
        {
            step++;
            if( p.global)
            {
                std::cout << "(m_tot-m_0)/m_0: "<< (test.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = test.energy();
                diff = (E1 - E0)/p.dt;
                double diss = test.energy_diffusion( );
                std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";

            }
            try{ ab( test, y0, y1, p.dt);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                running = false;
                break;
            }
            y0.swap( y1); //attention on -O3 ?
        }
        time += (double)p.itstp*p.dt;
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "\n\t Step "<<step;
        std::cout << "\n\t Average time for one step: "<<t.diff()/(double)p.itstp<<"s\n\n";
#endif//DG_BENCHMARK
        running = running && 
                  !glfwGetKey( GLFW_KEY_ESC) &&
                  glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////

    return 0;

}
