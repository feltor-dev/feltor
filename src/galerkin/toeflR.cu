#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"

#include "toeflR.cuh"
#include "dg/rk.cuh"
#include "dg/timer.cuh"
#include "file/read_input.h"
#include "parameters.h"


using namespace std;
using namespace dg;

const unsigned k = 3;

using namespace std;

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
        cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
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
        cerr << "ERROR: k doesn't match: "<<k<<" vs. "<<p.k<<"\n";
        return -1;
    }

    dg::Grid<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::ToeflR< dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, p.global); 
    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian

    blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    blas2::symv( (dg::DVec)create::v2d( grid), y0[1], y0[1]);

    if( p.global)
    {
        thrust::transform( y0[0].begin(), y0[0].end(), y0[0].begin(), dg::PLUS<double>(+1));
        thrust::transform( y0[1].begin(), y0[1].end(), y0[1].begin(), dg::PLUS<double>(+1));
        test.log( y0, y0); //transform to logarithmic values
    }

    dg::AB< k, std::vector<dg::DVec> > ab( y0);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec hvisual( grid.size(), 0.), visual(hvisual);
    dg::HMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    bool running = true;
    double time = 0;
    //compute mass 0
        test.exp( y0, y1);
        dg::DVec one( grid.size(), 1.);
        dg::DVec w2d = create::w2d( grid);
        const double mass0 = blas2::dot( one, w2d, y1[1]);
        thrust::transform( y1[1].begin(), y1[1].end(), y1[1].begin(), dg::PLUS<double>(-1));
        const double mass_blob0 = blas2::dot( one, w2d, y1[1]);
        double E0 = 0, E1 = 0, diff = 0;
    ab.init( test, y0, p.dt);
    std::cout << "Begin computation \n";
    while (running)
    {
        //transform field to an equidistant grid
        if( p.global)
        {
            test.exp( y0, y1);
            thrust::transform( y1[0].begin(), y1[0].end(), dvisual.begin(), dg::PLUS<double>(-1));
        }
        else
            hvisual = y0[0];

        hvisual = dvisual;
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        w.title() << setprecision(2) << scientific;
        w.title() <<"ne / "<<colors.scale()<<"\t";
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //transform phi
        dg::blas2::gemv( test.laplacianM(), test.potential()[0], y1[1]);
        hvisual = y1[1];
        dg::blas2::gemv( equi, hvisual, visual);
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        w.title() <<"phi / "<<colors.scale()<<"\t";
        w.title() << fixed; 
        w.title() << " &&   time = "<<time;
        w.draw( visual, grid.n()*grid.Nx(), grid.n()*grid.Ny(), colors);

        //step 
#ifdef DG_BENCHMARK
        t.tic();
#endif//DG_BENCHMARK
        for( unsigned i=0; i<p.itstp; i++)
        {
            try{ ab( test, y0, y1, p.dt);}
            catch( dg::Fail& fail) { 
                cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                cerr << "Does Simulation respect CFL condition?\n";
                running = false;
                break;
            }
            y0.swap( y1); //attention on -O3 ?
        }
        time += (double)p.itstp*p.dt;
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "\n        Average time for one step: "<<t.diff()/(double)p.itstp<<"s\n\n";
#else//DG_BENCHMARK
        test.exp( y0, y1);
        std::cout << scientific << setprecision( 3);
        std::cout << "m_tot/m_0: "<< (blas2::dot( one, w2d, y1[1])-mass0)/mass_blob0<<"\t";
        E0 = E1;
        E1 = test.energy( y0, test.potential()[0]);
        diff = (E1 - E0)/p.dt/(double)p.itstp;


        std::cout << "total energy: "<< diff<<"\t";//test.energy( y0, test.potential()[0])<<"\t";
        double diss = test.energy_dot( y0, test.potential());
        std::cout << "total energy dissipation: "<< diss<<"\n";
#endif//DG_BENCHMARK
        //std::cout << " Ratio "<< diff/diss <<"\n";
        //glfwWaitEvents();
        running = running && 
                  !glfwGetKey( GLFW_KEY_ESC) &&
                  glfwGetWindowParam( GLFW_OPENED);
    }
    ////////////////////////////////////////////////////////////////////

    return 0;

}
