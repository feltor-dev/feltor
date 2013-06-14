#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"

#include "toeflR.cuh"
#include "rk.cuh"
#include "../lib/read_input.h"
#include "parameters.h"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 4;
const unsigned k = 3;

using namespace std;

int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> v, v2;
    if( argc == 1)
    {
        v = toefl::read_input("input.txt");
    }
    else if( argc == 2)
    {
        v = toefl::read_input( argv[1]);
    }
    else
    {
        cerr << "ERROR: Too many arguments!\nUsage: "<< argv[0]<<" [filename]\n";
        return -1;
    }

    v2 = toefl::read_input( "window_params.txt");
    draw::HostWindow w(v2[3], v2[4]);
    w.set_multiplot( v2[1], v2[2]);
    /////////////////////////////////////////////////////////////////////////
    const Parameters p( v);
    p.display( std::cout);
    if( p.n != n || p.k != k)
    {
        cerr << "ERROR: n or k doesn't match: "<<k<<" vs. "<<p.k<<" and "<<n<<" vs. "<<p.n<<"\n";
        return -1;
    }

    dg::Grid<double, n > grid( 0, p.lx, 0, p.ly, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::ToeflR<double, n, dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, p.global); 
    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian

    blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    blas2::symv( V2D<double, n> ( grid), y0[1], y0[1]);

    if( p.global)
    {
        thrust::transform( y0[0].begin(), y0[0].end(), y0[0].begin(), dg::PLUS<double>(+1));
        thrust::transform( y0[1].begin(), y0[1].end(), y0[1].begin(), dg::PLUS<double>(+1));
        test.log( y0, y0); //transform to logarithmic values
    }

    dg::AB< k, std::vector<dg::DVec> > ab( y0);

    dg::DVec dvisual( grid.size(), 0.);
    dg::HVec visual( grid.size(), 0.);
    dg::DMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    bool running = true;
    double time = 0;
    ab.init( test, y0, p.dt);
    while (running)
    {
        //transform field to an equidistant grid
        if( p.global)
        {
            test.exp( y0, y1);
            thrust::transform( y1[0].begin(), y1[0].end(), y1[0].begin(), dg::PLUS<double>(-1));
            dg::blas2::gemv( equi, y1[0], dvisual);
        }
        else
            dg::blas2::gemv( equi, y0[0], dvisual);

        visual = dvisual; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        w.title() << setprecision(2) << scientific;
        w.title() <<"ne / "<<colors.scale()<<"\t";
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);

        //transform phi
        dg::blas2::gemv( test.laplacianM(), test.polarisation(), y1[1]);
        dg::blas2::gemv( equi, y1[1], dvisual);
        visual = dvisual; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        w.title() <<"phi / "<<colors.scale()<<"\t";
        w.title() << fixed; 
        w.title() << " &&   time = "<<time;
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);
#ifdef DG_DEBUG
        glfwWaitEvents();
#endif //DG_DEBUG

        //step 
        t.tic();
        for( unsigned i=0; i<p.itstp; i++)
        {
            ab( test, y0, y1, p.dt);
            y0.swap( y1); //attention on -O3 ?
            //for( unsigned i=0; i<y0.size(); i++)
            //    thrust::swap( y0[i], y1[i]);
        }
        time += (double)p.itstp*p.dt;
        t.toc();
        //glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    std::cout << "Average time for one step: "<<t.diff()/(double)p.itstp<<"s\n";
    ////////////////////////////////////////////////////////////////////

    return 0;

}
