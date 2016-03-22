#include <iostream>
#include <iomanip>
#include <vector>


#include "file/file.h"
#include "file/read_input.h"
#include "file/nc_utilities.h"

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
    dg::ToeflR< dg::DMatrix, dg::DVec, dg::DVec > test( grid, p.kappa, p.nu, p.tau, p.eps_pol, p.eps_gamma, p.global); 
    dg::Diffusion<dg::DMatrix, dg::DVec, dg::DVec> diffusion( grid, p.nu, p.global);
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
        dg::blas1::transform( y0[0], y0[0], dg::PLUS<double>( +1));
        dg::blas1::transform( y0[1], y0[1], dg::PLUS<double>( +1));
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
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    err = nc_create( argv[2],NC_NETCDF4|NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid);
    //field IDs
    std::string names[3] = {"electrons", "ions", "potential"}; 
    int dataIDs[3]; 
    for( unsigned i=0; i<3; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    //energy IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, dissID, dEdtID;
    err = nc_def_var( ncid, "energy",      NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var( ncid, "mass",        NC_DOUBLE, 1, &EtimeID, &massID);
    err = nc_def_var( ncid, "dissipation", NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var( ncid, "dEdt",        NC_DOUBLE, 1, &EtimeID, &dEdtID);
    err = nc_enddef(ncid);
    size_t start[3] = {0, 0, 0};
    size_t count[3] = {1, grid.n()*grid.Ny(), grid.n()*grid.Nx()};
    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    //output all three fields
    std::vector<dg::DVec> transferD(3);
    std::vector<dg::HVec> output(3);
    transferD[0] = y1[0], transferD[1] = y1[1], transferD[2] = test.potential()[0]; //electrons
    for( int k=0;k<3; k++)
        output[k] = transferD[k]; 
    start[0] = 0;
    for( int k=0; k<3; k++)
        err = nc_put_vara_double( ncid, dataIDs[k], start, count, output[k].data() );
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    err = nc_close(ncid);
    ///////////////////////////////////////Timeloop/////////////////////////////////
    dg::Timer t;
    t.tic();
    try
    {
#ifdef DG_BENCHMARK
    unsigned step = 0;
#endif //DG_BENCHMARK
    for( unsigned i=1; i<=p.maxout; i++)
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
            time+=p.dt;
            if( p.global) 
            {
                err = nc_open(argv[2], NC_WRITE, &ncid);
                double ener=test.energy(), mass=test.mass(), diff=test.mass_diffusion(), dEdt=test.energy_diffusion();
                err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
                err = nc_put_vara_double( ncid, energyID,   Estart, Ecount, &ener);
                err = nc_put_vara_double( ncid, massID,     Estart, Ecount, &mass);
                err = nc_put_vara_double( ncid, dissID,     Estart, Ecount, &diff);
                err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount, &dEdt);
                err = nc_close(ncid);
            }
        }
        //output all three fields
        if( p.global)
            test.exp( y1,y1); //transform to correct values
        transferD[0] = y1[0], transferD[1] = y1[1], transferD[2] = test.potential()[0]; //electrons
        for( int k=0;k<3; k++)
            output[k] = transferD[k]; 
        err = nc_open(argv[2], NC_WRITE, &ncid);
        start[0] = i;
        for( int k=0; k<3; k++)
            err = nc_put_vara_double( ncid, dataIDs[k], start, count, output[k].data() );
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        err = nc_close(ncid);

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

