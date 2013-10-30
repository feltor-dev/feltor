#include <iostream>
#include <iomanip>

#include "dg/timer.cuh"
#include "dg/functors.cuh"
#include "dg/evaluation.cuh"
#include "dg/rk.cuh"
#include "dg/xspacelib.cuh"
#include "dg/typedefs.cuh"

#include "file/read_input.h"
#include "file/file.h"

#include "shu.cuh"
#include "parameters.h"



using namespace std;
using namespace dg;
const unsigned k = 3;

int main( int argc, char * argv[])
{
    Timer t;
    //input files
    std::vector<double> v;
    std::string input;
    if( argc != 3)
    {
        std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else 
    {
        v = file::read_input( argv[1]);
        input = file::read_file( argv[1]);
    }
    const Parameters p( v);
    p.display();
    if( p.k != k)
    {
        std::cerr << "Time stepper needs recompilation!\n";
        return -1;
    }

    //initiate solver 
    Grid<double> grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    DVec w2d( create::w2d(grid));
    dg::Lamb lamb( p.posX*p.lx, p.posY*p.ly, p.R, p.U);
    HVec omega = evaluate ( lamb, grid);
    DVec y0( omega ), y1( y0);
    //make solver and stepper
    Shu<DVec> shu( grid, p.D, p.eps);
    AB< k, DVec > ab( y0);
    ab.init( shu, y0, p.dt);
    ab( shu, y0, y1, p.dt);
    y0.swap( y1); //y1 now contains value at zero time

    DVec stencil = evaluate( one, grid);
    double vorticity = blas2::dot( stencil , w2d, y1);
    double enstrophy = 0.5*blas2::dot( y1, w2d, y1);
    double energy =    0.5*blas2::dot( y1, w2d, shu.potential()) ;
    /////////////////////////////////////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    double time = 0;
    unsigned step = 0;
    dg::HVec output[3] = { y1, y1, shu.potential()}; //intermediate transport locations
    t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
    t5file.append( vorticity, enstrophy, energy, 0);
    try{
    for( unsigned i=0; i<p.maxout; i++)
    {

        dg::Timer ti;
        ti.tic();
        for( unsigned j=0; j<p.itstp; j++)
        {
            ab( shu, y0, y1, p.dt);
            y0.swap( y1); 
            vorticity = blas2::dot( stencil , w2d, y1);
            enstrophy = 0.5*blas2::dot( y1, w2d, y1);
            energy    = 0.5*blas2::dot( y1, w2d, shu.potential()) ;
            t5file.append( vorticity, enstrophy, energy, 0);
        }
        step+=p.itstp;
        time += p.itstp*p.dt;
        //output all fields
        output[0] = y1, output[1] = y1, output[2] = shu.potential(); 
        t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
        ti.toc();
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
    }
    }
    catch( dg::Fail& fail) { 
        std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        std::cerr << "Does Simulation respect CFL condition?\n";
    }

    return 0;

}
