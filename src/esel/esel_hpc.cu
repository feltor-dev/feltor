#include <iostream>
#include <iomanip>
#include <vector>


#include "esel.cuh"
#include "../toefl/parameters.h"
#include "dg/multistep.h"
#include "file/file.h"
#include "file/read_input.h"

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
    dg::SOL sol( v[22], v[23], v[24], v[25], v[27]);
    std::cout << "The SOL parameters are: \n"
              << "    x_l:     "<<v[22] <<"\n    x_w:     "<<v[23]<<"\n"
              << "    sigma_l: "<<v[24] <<"\n    sigma_w: "<<v[25]<<"\n";
    //create RHS 
    dg::Esel<dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, sol); 
    //create initial vector
    dg::EXPX<double> exp( 1., -1./v[26]); 
    std::vector<dg::DVec> y0(2, dg::evaluate( exp, grid)), y1(y0); 
    {
        dg::Gaussian gaussian( p.posX/2.*grid.lx(), p.posY/2.*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
        std::vector<dg::DVec> y0p(2, dg::evaluate( gaussian, grid)); 
        dg::blas1::axpby( 1, y0p, 1, y0);
    } 
    {
        dg::Gaussian gaussian( p.posX/2.*grid.lx(), p.posY/3.*grid.ly(), p.sigma, p.sigma, -p.n0); 
        std::vector<dg::DVec> y0p(2, dg::evaluate( gaussian, grid)); 
        dg::blas1::axpby( 1, y0p, 1, y0);
    }
    {
        dg::Gaussian gaussian( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); 
        std::vector<dg::DVec> y0p(2, dg::evaluate( gaussian, grid)); 
        dg::blas1::axpby( 1, y0p, 1, y0);
    }{
        dg::Gaussian gaussian( p.posX/2.*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, -p.n0); 
        std::vector<dg::DVec> y0p(2, dg::evaluate( gaussian, grid)); 
        dg::blas1::axpby( 1, y0p, 1, y0);
    }



    dg::blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    {
    dg::DVec v2d = dg::create::inv_weights(grid);
    dg::blas2::symv( v2d, y0[1], y0[1]);
    }
    assert( p.tau == 0);
    assert( p.global);
    assert( p.bc_x == dg::DIR_NEU);

    test.log( y0, y0); //transform to logarithmic values
    //////////////////initialisation of timestepper and first step///////////////////
    double time = 0;
    dg::AB< k, std::vector<dg::DVec> > ab( y0);
    ab.init( test, y0, p.dt);
    ab( test, y1);
    //y0.swap( y1); //y1 now contains value at zero time
    /////////////////////////////set up hdf5/////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    dg::HVec output[3] = { y1[0], y1[0], y1[0]}; //intermediate transport locations
    test.exp( y1,y1); //transform to correct values
    output[0] = y1[0], output[1] = y1[1], output[2] = test.potential()[0]; //electrons
    t5file.write( output[0], output[1], output[2], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
    t5file.append( test.mass(), test.mass_diffusion(), test.energy(), test.energy_diffusion());
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
            ab( test, y1);
            //store accuracy details
            t5file.append( test.mass(), test.mass_diffusion(), test.energy(), test.energy_diffusion());
        }
        time += p.itstp*p.dt;
        //output all three fields
        test.exp( y1,y1); //transform to correct values
        output[0] = y1[0], output[1] = y1[1], output[2] = test.potential()[0]; //electrons
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


