#include <iostream>
#include <iomanip>
#include <vector>

#include "draw/host_window.h"

#include "toeflR.cuh"
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
        <<"    Coupling:        = "<<v[10]<<"\n"
        <<"    Viscosity:       = "<<v[11]<<"\n"
        <<"    Curvature_y:     = "<<v[12]<<"\n";
        //<<"   Species/Parameter   g\ta\tmu\ttau\n"
        //<<"    Electrons:         "<<g_e  <<"\t"<<"-1"<<"\t"<<"0"<<"\t"<<"1\n"
        //<<"    Ions:              "<<g[0] <<"\t"<<a[0]<<"\t"<<mu[0]<<"\t"<<tau[0]<<"\n"
        //<<"    Impurities:        "<<g[1] <<"\t"<<a[1]<<"\t"<<mu[1]<<"\t"<<tau[1]<<"\n";
    char per[] = "PERIODIC", dir[] = "DIRICHLET";
    os << "Boundary parameters are: \n"
        <<"    lx = "<<v[4]<<"\n"
        <<"    ly = "<<v[5]<<"\n"
        <<"Boundary conditions in x are: \n"
        <<"    "<<(v[6] ? dir:per)<<"\n"
        <<"Boundary conditions in y are: \n"
        <<"    "<<(v[7] ? dir:per)<<"\n";
    os << "Algorithmic parameters are: \n"
        <<"    Nx = "<<v[1]<<"\n"
        <<"    Ny = "<<v[2]<<"\n"
        <<"    dt = "<<v[3]<<"\n";
    char enabled[] = "ENABLED", disabled[] = "DISABLED";
    char local[] = "LOCAL" , global[] = "GLOBAL";

    os << "Impurities are: \n"
        <<"    "<<(v[16]?enabled:disabled)<<"\n"
        //<<"Global solvers are: \n"
        //<<"    "<<(global?enabled:disabled)<<"\n"
        <<"Modified Hasegawa Wakatani: \n"
        <<"    "<<(v[8]?enabled:disabled)<<"\n"
        <<"Mode is:   \n"
        <<"    "<<(v[9]?global:local)<<"\n";
    os  << "Blob width is:     "<<v[14]<<"\n"
        << "Blob amplitude is: "<<v[13]<<"\n"
        << "Stopping for CG:         "<<v[23]<<"\n"
        << "Stopping for Gamma CG:   "<<v[25]<<"\n";


}

int main( int argc, char* argv[])
{
    //Parameter initialisation
    std::vector<double> v;
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
    draw::HostWindow w(v[24], 2*v[24]*v[5]/v[4]);
    w.set_multiplot( 2, 1);
    /////////////////////////////////////////////////////////////////////////
    display( v, std::cout);

    dg::bc bc_x = dg::PER, bc_y = dg::PER;
    if( v[6]) bc_x = dg::DIR;
    if( v[7]) bc_y = dg::DIR;
    dg::Grid<double, n > grid( 0, v[4], 0, v[5], (unsigned)v[1], (unsigned)v[2], bc_x, bc_y);
    //create RHS 
    dg::ToeflR<double, n, dg::DVec > test( grid, v[12], v[11], v[15], v[23], v[25]); 
    //create initial vector
    dg::Gaussian g( 0.3*v[4], 0.6*v[5], v[14], v[14], v[13]); //gaussian width is in absolute values
    if( !v[9])
    {
        cerr << "ToeflR is always global\n";
        return;
    }
    std::vector<dg::DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian
    //test.init( y0);
    blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    blas2::symv( V2D<double, n> ( grid), y0[1], y0[1]);

    thrust::transform( y0[0].begin(), y0[0].end(), y0[0].begin(), dg::PLUS<double>(+1));
    thrust::transform( y0[1].begin(), y0[1].end(), y0[1].begin(), dg::PLUS<double>(+1));

    test.log( y0, y0); //transform to logarithmic values
    dg::AB< k, std::vector<dg::DVec> > ab( y0);

    dg::DVec dvisual( grid.size());
    dg::HVec visual( n*n*v[1]*v[2]);
    dg::DMatrix equi = dg::create::backscatter( grid);
    draw::ColorMapRedBlueExt colors( 1.);
    //create timer
    Timer t;
    bool running = true;
    double time = 0, dt = v[3];
    ab.init( test, y0, dt);
    while (running)
    {
        //transform field to an equidistant grid
        test.exp( y0, y1);
        thrust::transform( y1[0].begin(), y1[0].end(), y1[0].begin(), dg::PLUS<double>(-1));
        dg::blas2::gemv( equi, y1[0], dvisual);
        visual = dvisual; //transfer to host
        //compute the color scale
        colors.scale() =  (float)thrust::reduce( visual.begin(), visual.end(), 0., dg::AbsMax<double>() );
        //draw ions
        w.title() <<"ni/ "<<colors.scale()<<"\t";
        w.draw( visual, n*v[1], n*v[2], colors, 0, 0);

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
        w.draw( visual, n*v[1], n*v[2], colors, 1, 0);
#ifdef DG_BENCHMARK
        glfwWaitEvents();
#endif //DG_BENCHMARK

        //step 
        t.tic();
        for( unsigned i=0; i<v[21]; i++)
        {
            ab( test, y0, y1, dt);
            y0.swap( y1); //attention on -O3 ?
            //for( unsigned i=0; i<y0.size(); i++)
            //    thrust::swap( y0[i], y1[i]);
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
