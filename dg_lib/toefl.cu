#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"

#include "toefl.cuh"
#include "rk.cuh"
#include "../lib/read_input.h"

#include "timer.cuh"

using namespace std;
using namespace dg;

const unsigned n = 3;
const unsigned k = 3;

using namespace std;
void display( const std::vector<double>& v, std::ostream& os = std::cout )
{
    os << "Physical parameters are: \n"
        <<"    Viscosity:       = "<<v[13]<<"\n"
        <<"    Curvature_y:     = "<<v[14]<<"\n"
        <<"    Ion-temperature: = "<<v[15]<<"\n";
    char local[] = "LOCAL" , global[] = "GLOBAL";
    os  <<"Mode is:   \n"
        <<"    "<<(v[12]?global:local)<<"\n";
    char per[] = "PERIODIC", dir[] = "DIRICHLET";
    os << "Boundary parameters are: \n"
        <<"    lx = "<<v[8]<<"\n"
        <<"    ly = "<<v[9]<<"\n"
        <<"Boundary conditions in x are: \n"
        <<"    "<<(v[10] ? dir:per)<<"\n"
        <<"Boundary conditions in y are: \n"
        <<"    "<<(v[11] ? dir:per)<<"\n";
    os << "Algorithmic parameters are: \n"
        <<"    n  = "<<v[1]<<"\n"
        <<"    Nx = "<<v[2]<<"\n"
        <<"    Ny = "<<v[3]<<"\n"
        <<"    k  = "<<v[4]<<"\n"
        <<"    dt = "<<v[5]<<"\n";
    os  <<"Blob parameters are: \n"
        << "    width is:     "<<v[17]<<"\n"
        << "    amplitude is: "<<v[16]<<"\n"
        << "    posX:         "<<v[18]<<"\n"
        << "    posY:         "<<v[19]<<"\n";
    os << "Stopping for CG:         "<<v[6]<<"\n"
        <<"Stopping for Gamma CG:   "<<v[7]<<"\n";


}
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
    display( v, std::cout);


    dg::bc bc_x = dg::PER, bc_y = dg::PER;
    if( v[10]) bc_x = dg::DIR;
    if( v[11]) bc_y = dg::DIR;
    unsigned Nx = v[2], Ny = v[3];
    double lx = v[8], ly = v[9];
    dg::Grid<double, n > grid( 0, lx, 0, ly, Nx, Ny, bc_x, bc_y);
    //create initial vector
    double n0 = v[16], sigma = v[17], posX = v[18], posY = v[19];
    dg::Gaussian g( posX*grid.lx(), posY*grid.ly(), sigma, sigma, n0); //gaussian width is in absolute values
    dg::DVec ne = dg::evaluate ( g, grid);
    bool global = v[12];
    if( global)
        thrust::transform( ne.begin(), ne.end(), ne.begin(), dg::PLUS<double>(1));
    std::vector<dg::DVec> y0(2, ne), y1(y0); // n_e = n_i 

    //create RHS and RK
    double eps_pol = v[6], nu = v[13], kappa = v[14];
    dg::Toefl<double, n, dg::DVec > test( grid, global, eps_pol , kappa, nu, bc_x, bc_y); 
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
    double time = 0, dt = v[5];
    unsigned itstp = v[20];
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
