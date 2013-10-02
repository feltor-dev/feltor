#include <iostream>
#include <iomanip>
#include <vector>


#include "toeflR.cuh"
#include "parameters.h"
#include "dg/rk.cuh"
#include "file/file.h"
#include "file/read_input.h"

#include "dg/timer.cuh"


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
    dg::Grid<double > grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y);
    //create RHS 
    dg::ToeflR<dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, p.global); 
    //create initial vector
    dg::Gaussian g( p.posX*grid.lx(), p.posY*grid.ly(), p.sigma, p.sigma, p.n0); 
    std::vector<dg::DVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian
    dg::blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    dg::blas2::symv( (dg::DVec)dg::create::v2d( grid), y0[1], y0[1]);
    if( p.global)
    {
        thrust::transform( y0[0].begin(), y0[0].end(), y0[0].begin(), dg::PLUS<double>(+1));
        thrust::transform( y0[1].begin(), y0[1].end(), y0[1].begin(), dg::PLUS<double>(+1));
        test.log( y0, y0); //transform to logarithmic values
    }
    //////////////////initialisation of timestepper and first step///////////////////
    double time = 0;
    dg::AB< k, std::vector<dg::DVec> > ab( y0);
    ab.init( test, y0, p.dt);
    ab( test, y0, y1, p.dt);
    y0.swap( y1); //y1 now contains value at zero time
    /////////////////////////////set up hdf5/////////////////////////////////
    dg::HVec output( y1[0]); //intermediate transport location
    hid_t   file, grp;
    herr_t  status;
    hsize_t dims[] = { grid.n()*grid.Ny(), grid.n()*grid.Nx() };
    file = H5Fcreate( argv[2], H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    hsize_t size = input.size();
    status = H5LTmake_dataset_char( file, "inputfile", 1, &size, input.data()); //name should precede t so that reading is easier
    std::vector<double> mass, diffusion, energy, dissipation;
    ///////////////////////////////////First Output (t = 0)/////////////////////////
    //output all three fields
    if( p.global)
        test.exp( y1,y1); //transform to correct values
    grp = H5Gcreate( file, file::setTime( time).data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT  );
    output = y1[0]; //electrons
    status = H5LTmake_dataset_double( grp, "electrons", 2,  dims, output.data());
    output = y1[1]; //ions
    status = H5LTmake_dataset_double( grp, "ions", 2,  dims, output.data());
    output = test.potential()[0];
    status = H5LTmake_dataset_double( grp, "potential", 2,  dims, output.data());
    H5Gclose( grp);
    if( p.global) 
    {
        mass.push_back( test.mass());
        diffusion.push_back( test.mass_diffusion());
        energy.push_back( test.energy()); 
        dissipation.push_back( test.energy_diffusion());
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
            ab( test, y0, y1, p.dt);
            y0.swap( y1); //attention on -O3 ?
            //store accuracy details
            if( p.global) 
            {
                mass.push_back( test.mass());
                diffusion.push_back( test.mass_diffusion());
                energy.push_back( test.energy()); 
                dissipation.push_back( test.energy_diffusion());
            }
        }
        time += p.itstp*p.dt;
        //output all three fields
        if( p.global)
            test.exp( y1,y1); //transform to correct values
        grp = H5Gcreate( file, file::setTime( time).data(), H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT  );
        output = y1[0]; //electrons
        status = H5LTmake_dataset_double( grp, "electrons", 2,  dims, output.data());
        output = y1[1]; //ions
        status = H5LTmake_dataset_double( grp, "ions", 2,  dims, output.data());
        output = test.potential()[0];
        status = H5LTmake_dataset_double( grp, "potential", 2,  dims, output.data());
        H5Gclose( grp);
#ifdef DG_BENCHMARK
        ti.toc();
        step+=p.itstp;
        std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n";
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
    std::cout <<"which is         \t"<<t.diff()/mass.size()<<"s/step\n";
    //std::cout << mass.size()<<"\n";

    if( p.global)
    {
        grp = H5Gcreate( file, "xfiles", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT  );
        dims[0] = mass.size();//(p.maxout+1)*p.itstp;
        status = H5LTmake_dataset_double( grp, "mass", 1,  dims, mass.data());
        status = H5LTmake_dataset_double( grp, "diffusion", 1,  dims, diffusion.data());
        status = H5LTmake_dataset_double( grp, "energy", 1,  dims, energy.data());
        status = H5LTmake_dataset_double( grp, "dissipation", 1,  dims, dissipation.data());
        H5Gclose( grp);
    }

    //writing takes the same time as device-host transfers
    ////////////////////////////////////////////////////////////////////
    H5Fclose( file);

    return 0;

}

