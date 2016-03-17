#include <iostream>
#include <iomanip>
#include <vector>


#include "file/file.h"
#include "file/read_input.h"

#include "toeflR.cuh"
#include "dg/algorithm.h"
#include "dg/backend/xspacelib.cuh"
#include "parameters.h"

#include "dg/backend/timer.cuh"


/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - writes outputs to a given outputfile using hdf5. 
        density fields are the real densities in XSPACE ( not logarithmic values)
*/

const unsigned k = 3;//!< a change in k needs a recompilation

int main( int argc, char* argv[])
{
    //Parameter initialisation
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
    p.display( std::cout);
    if( p.k != k)
    {
        std::cerr << "ERROR: k doesn't match: "<<k<<" (code) vs. "<<p.k<<" (input)\n";
        return -1;
    }

    ////////////////////////////////set up computations///////////////////////////
    dg::Grid2d<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::ToeflR< dg::DMatrix, dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, p.global); 
    dg::Diffusion<dg::DMatrix, dg::DVec> diffusion( grid, p.nu, p.global);
    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); 
    std::vector<dg::DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian
    dg::blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    {
        dg::DVec v2d = dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[1], y0[1]);
    }

    if( p.global)
    {
        dg::blas1::transform( y0[0], y0[0], dg::PLUS<double>(+1));
        dg::blas1::transform( y0[1], y0[1], dg::PLUS<double>(+1));
        test.log( y0, y0); //transform to logarithmic values
    }
    //////////////////initialisation of timestepper and first step///////////////////
    double time = 0;
    //dg::AB< k, std::vector<dg::DVec> > ab( y0);
    dg::Karniadakis< std::vector<dg::DVec> > ab( y0, y0[0].size(), 1e-9);
    ab.init( test, diffusion, y0, p.dt);
    y0.swap( y1); //y1 now contains value at zero time
    if( p.global)
        test.exp( y1,y1); //transform to correct values
    /////////////////////////////set up hdf5/////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    dg::DVec transferD[3] = { y1[0], y1[1], test.potential()[0]}; //intermediate transport locations
    dg::HVec output[3];
    for( int i=0;i<3; i++)
        dg::blas1::transfer( transferD[i], output[i]);
    t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
    if( p.global) 
    {
        t5file.append( test.mass(), test.mass_diffusion(), test.energy(), test.energy_diffusion());
    }
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
    try
    {
#ifdef DG_BENCHMARK
    unsigned step = 0;
#endif //DG_BENCHMARK
    for( unsigned i=0; i<p.maxout; i++)
    {

#ifdef DG_BENCHMARK
        dg::Timer ti;
        ti.tic();
#endif//DG_BENCHMARK
        for( unsigned j=0; j<p.itstp; j++)
        {
            ab( test, diffusion, y0);
            y0.swap( y1); //attention on -O3 ?
            //store accuracy details
            if( p.global) 
            {
                t5file.append( test.mass(), test.mass_diffusion(), test.energy(), test.energy_diffusion());
            }
        }
        time += p.itstp*p.dt;
        //output all three fields
        if( p.global)
            test.exp( y1,y1); //transform to correct values
        transferD[0] = y1[0], transferD[1] = y1[1], transferD[2] = test.potential()[0]; //electrons
        for( int i=0;i<3; i++)
            dg::blas1::transfer( transferD[i], output[i]);
        t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
#ifdef DG_BENCHMARK
        ti.toc();
        step+=p.itstp;
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
    }
    }
    catch( dg::Fail& fail) { 
        std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        std::cerr << "Does Simulation respect CFL condition?\n";
    }
    t.toc(); 
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";

    return 0;

}

