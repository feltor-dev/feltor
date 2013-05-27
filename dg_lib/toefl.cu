#include <iostream>
#include <iomanip>
#include <vector>

#include "cuda_texture.cuh"

#include "toefl.cuh"
#include "rk.cuh"
#include "../lib/read_input.h"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned k = 3;

using namespace std;

int main()
{
    //do a cin for gridpoints
    std::vector<double> v = toefl::read_input( "input.txt");
    dg::HostWindow w(v[24], v[24]*v[5]/v[4]);
    /////////////////////////////////////////////////////////////////////////
    cout << "# of Legendre coefficients: " << n<<endl;
    cout << "# of grid cells:            " << Nx*Ny<<endl;
    cout << "Timestep                    " << dt << endl;

    dg::Grid<double,n > grid( 0, v[4], 0, v[5], (unsigned)v[1], (unsigned)v[2]);
    //create initial vector
    dg::Gaussian g( 0.4*lx, 0.5*ly, v[14]/2.355, v[14]/2.355, v[13]); //gaussian width is in absolute values
    dg::DVec ne = dg::evaluate ( g, grid);
    if( global)
        thrust::transform( ne.begin(), ne.end(), ne.begin(), dg::PLUS<double>(1));
    std::vector<dg::DVec> y0(2, ne), y1(y0); // n_e = n_i 

    //create RHS and RK
    dg::bc bc_x = dg::PER, bc_y = dg::PER;
    if( v[6]) bc_x = dg::DIR;
    if( v[7]) bc_y = dg::DIR;
    bool global = v[9];
    dg::Toefl<double, n, dg::DVec > test( grid, global, v[23], v[12], v[11]); 
    if( global)
        test.log( y0,y0); //transform to logarithmic values
    dg::RK< k, dg::Toefl<double, n, dg::DVec> > rk( y0);
    dg::AB< k, dg::Toefl<double, n, dg::DVec> > ab( y0);

    dg::HVec visual( n*n*v[1]*v[2]);
    dg::DMatrix equi = dg::create::backscatter( grid);
    dg::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    bool running = true;
    double time = 0, dt = v[3];
    ab.init( test, y0, dt);
    while (running)
    {
        t.tic();
        //transform field to an equidistant grid
        if( global)
        {
            test.exp( y0, y1);
            thrust::transform( y1[0].begin(), y1[0].end(), y1[0].begin(), dg::PLUS<double>(-1));
            dg::blas2::gemv( equi, y1[0], y1[1]);
        }
        else
            dg::blas2::gemv( equi, y0[0], y1[1]);
        t.toc();
        //std::cout << "Equilibration took        "<<t.diff()<<"s\n";
        t.tic();
        visual = y1[1]; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), -1., dg::AbsMax<double>() );
        //draw and swap buffers
        w.title() << scientific;
        w.title() <<"Scale "<<colors.scale()<<"\t";
        w.title() << setprecision(2) << fixed;
        w.title() << " &&   time = "<<time;
        w.draw( visual, n*Nx, n*Ny, colors);
        t.toc();
        //step 
        t.tic();
        for( unsigned i=0; i<v[21]; i++)
        {
            ab( test, y0, y1, dt);
            for( unsigned i=0; i<y0.size(); i++)
                thrust::swap( y0[i], y1[i]);
        }
        time += v[21]*dt;
        t.toc();
        //glfwWaitEvents();
        running = !glfwGetKey( GLFW_KEY_ESC) &&
                    glfwGetWindowParam( GLFW_OPENED);
    }
    std::cout << "Average time for one step: "<<t.diff()/v[21]<<"s\n";
    ////////////////////////////////////////////////////////////////////

    return 0;

}
