#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"

#include "toefl.cuh"
#include "parameters.h"
#include "rk.cuh"
#include "../lib/read_input.h"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
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
    if( n != v[1] || k != v[4]) {
        cerr << "Order is wrong\n";
        return;
    }
    /////////////////////////////////////////////////////////////////////////
    const Parameters p( v);
    p.display( std::cout);


    dg::Grid<double, n > grid( 0, p.lx, 0, p.ly, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    dg::DVec ne = dg::evaluate ( g, grid);
    bool global = p.global;
    if( global)
        thrust::transform( ne.begin(), ne.end(), ne.begin(), dg::PLUS<double>(1));
    std::vector<dg::DVec> y0(2, ne), y1(y0); // n_e = n_i 

    //create RHS and RK
    dg::Toefl<double, n, dg::DVec > test( grid, global, p.eps_pol , p.kappa, p.nu, p.bc_x, p.bc_y); 
    if( global)
        test.log( y0,y0); //transform to logarithmic values
    dg::RK< k, std::vector<dg::DVec> > rk( y0);
    dg::AB< k, std::vector<dg::DVec> > ab( y0);

    dg::DVec dvisual( grid.size());
    dg::HVec visual( grid.size());
    dg::DMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    bool running = true;
    double time = 0, dt = p.dt;
    unsigned itstp = p.itstp;
    ab.init( test, y0, dt);
    while (running)
    {
        //transform field to an equidistant grid
        /*
        if( global)
        {
            test.exp( y0, y1);
            thrust::transform( y1[0].begin(), y1[0].end(), y1[0].begin(), dg::PLUS<double>(-1));
            dg::blas2::gemv( equi, y1[0], y1[1]);
        }
        else
            dg::blas2::gemv( equi, y0[0], y1[1]);
        visual = y1[1]; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw electrons
        w.title() <<"ne/ "<<colors.scale()<<"\t";
        w.draw( visual, n*v[1], n*v[2], colors, 0, 0);
        */
        //transform field to an equidistant grid
        if( global)
        {
            test.exp( y0, y1);
            thrust::transform( y1[1].begin(), y1[1].end(), y1[1].begin(), dg::PLUS<double>(-1));
            dg::blas2::gemv( equi, y1[1], y1[0]);
        }
        else
            dg::blas2::gemv( equi, y0[1], y1[0]);
        visual = y1[0]; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        w.title() <<"ni/ "<<colors.scale()<<"\t";
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);

        //transform phi
        dg::blas2::gemv( test.laplacianM(), test.polarisation(), y1[1]);
        dg::blas2::gemv( equi, y1[1], dvisual);
        visual = dvisual; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw phi and swap buffers
        w.title() <<"phi/ "<<colors.scale()<<"\t";
        w.title() << setprecision(2) << fixed;
        w.title() << " &&   time = "<<time;
        w.draw( visual, n*grid.Nx(), n*grid.Ny(), colors);

        //step 
        t.tic();
        for( unsigned i=0; i<itstp; i++)
        {
            ab( test, y0, y1, dt);
            y0.swap( y1); //attention on -O3 ?
            //for( unsigned i=0; i<y0.size(); i++)
            //    thrust::swap( y0[i], y1[i]);
        }
        time += (double)itstp*dt;
        t.toc();
        //glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    std::cout << "Average time for one step: "<<t.diff()/(double)itstp<<"s\n";
    ////////////////////////////////////////////////////////////////////

    return 0;

}
