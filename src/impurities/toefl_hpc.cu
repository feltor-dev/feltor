#include <iostream>
#include <iomanip>
#include <vector>


#include "toeflI.cuh"
#include "../toefl/parameters.h"
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
    const Parameters p( v, 2);
    p.display( std::cout);
    if( p.k != k)
    {
        std::cerr << "ERROR: k doesn't match: "<<k<<" (code) vs. "<<p.k<<" (input)\n";
        return -1;
    }

    ////////////////////////////////set up computations///////////////////////////
    dg::Grid2d<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::ToeflI< dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.a_z, p.mu_z, p.tau_z, p.eps_pol, p.eps_gamma); 

    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); //gaussian width is in absolute values
    std::vector<dg::DVec> y0(3, dg::DVec( grid.size()) ), y1(y0);
    //dg::blas1::axpby( 1., y0[0], 1., (dg::DVec)dg::evaluate( g, grid), y0[0]);//n_e = 1+ gaussian
    dg::Helmholtz<dg::DMatrix, dg::DVec, dg::DVec> & gamma = test.gamma();
    if( v[25] == 1)
    {
        gamma.alpha() = -0.5*p.tau;
        y0[0] = dg::evaluate( g, grid);
        dg::blas2::symv( gamma, y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[1], y0[1]);
        dg::blas1::axpby( 1./(1.-p.a_z), y0[1], 0., y0[1]); //n_i ~1./a_i n_e
        y0[2] = dg::evaluate( dg::one, grid);
        dg::blas1::axpby( 1., y0[2], 1., y0[0]);
        dg::blas1::axpby( 1., y0[2], 1., y0[1]);
    }
    if( v[25] == 2) 
    {
        //init wall in y0[2]
        dg::GaussianX wall( v[26]*grid.lx(), v[28], v[27]); //position, sigma, amplitude
        dg::DVec wallv = dg::evaluate( wall, grid);
        gamma.alpha() = -0.5*p.tau_z*p.mu_z;
        dg::blas2::symv( gamma, wallv, y0[2]); 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a_z != 0.)
            dg::blas1::axpby( 1./p.a_z, y0[2], 0., y0[2]); //n_z ~1./a_z

        //init blob in y0[1]
        gamma.alpha() = -0.5*p.tau;
        y0[0] = dg::evaluate( g, grid);
        dg::blas2::symv( gamma, y0[0], y0[1]); 
        dg::blas2::symv( v2d, y0[1], y0[1]);
        if( p.a_z == 1)
        {
            std::cerr << "No blob with trace ions possible!\n";
            return -1;
        }
        dg::blas1::axpby( 1./(1-p.a_z), y0[1], 0., y0[1]); //n_i ~1./a_i n_e

        //sum up
        if( p.a_z != 0)
            dg::blas1::axpby( 1., wallv, 1., y0[0]); //add wall to blob in n_e
        dg::DVec one = dg::evaluate( dg::one, grid);
        for( unsigned i=0; i<3; i++)
            dg::blas1::axpby( 1., one, 1., y0[i]);
        
    }
    if( v[25] == 3) 
    {
        gamma.alpha() = -0.5*p.tau_z*p.mu_z;
        y0[0] = dg::evaluate( g, grid);
        dg::blas2::symv( gamma, y0[0], y0[2]); 
        dg::DVec v2d=dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[2], y0[2]);
        if( p.a_z == 0)
        {
            std::cerr << "No impurity blob with trace impurities possible!\n";
            return -1;
        }
        dg::blas1::axpby( 1./p.a_z, y0[2], 0., y0[2]); //n_z ~1./a_z n_e
        y0[1] = dg::evaluate( dg::one, grid);
        dg::blas1::axpby( 1., y0[1], 1., y0[0]);
        dg::blas1::axpby( 1., y0[1], 1., y0[2]);
    }

    test.log( y0, y0); //transform to logarithmic values
    //////////////////initialisation of timestepper and first step///////////////////
    double time = 0;
    dg::AB< k, std::vector<dg::DVec> > ab( y0);
    ab.init( test, y0, p.dt);
    ab( test, y1);
    /////////////////////////////set up hdf5/////////////////////////////////
    file::T5trunc t5file( argv[2], input);
    dg::HVec output[4] = { y1[0], y1[0], y1[0], y1[0]}; //intermediate transport locations
    test.exp( y1,y1); //transform to correct values
    output[0] = y1[0], output[1] = y1[1], output[2] = y1[2], output[3] = test.potential()[0]; //electrons
    t5file.write( output[0], output[1], output[2], output[3], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
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
        output[0] = y1[0], output[1] = y1[1], output[2] = y1[2], output[3] = test.potential()[0]; //electrons
        t5file.write( output[0], output[1], output[2], output[3], time, grid.n()*grid.Nx(), grid.n()*grid.Ny());
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

