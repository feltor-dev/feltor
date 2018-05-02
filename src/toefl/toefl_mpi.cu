#include <iostream>
#include <iomanip>
#include <vector>

#include <mpi.h> //activate mpi

#include "netcdf_par.h"
#include "file/nc_utilities.h"

#include "toeflR.cuh"
#include "dg/algorithm.h"
#include "parameters.h"


int main( int argc, char* argv[])
{
    ////////////////////////////////setup MPI///////////////////////////////
    int provided;
    MPI_Init_thread( &argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if( provided != MPI_THREAD_FUNNELED)
    {
        std::cerr << "wrong mpi-thread environment provided!\n";
        return -1;
    }
    int periods[2] = {false, true}; //non-, periodic
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    int num_devices=0;
    cudaGetDeviceCount(&num_devices);
    if(num_devices==0){std::cerr << "No CUDA capable devices found"<<std::endl; return -1;}
    int device = rank % num_devices; //assume # of gpus/node is fixed
    cudaSetDevice( device);
#endif//cuda
    int np[2];
    if(rank==0)
    {
        std::cin>> np[0] >> np[1];
        std::cout << "Computing with "<<np[0]<<" x "<<np[1]<<" = "<<size<<std::endl;
        assert( size == np[0]*np[1]);
    }
    MPI_Bcast( np, 2, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 2, np, periods, true, &comm);
    ////////////////////////Parameter initialisation//////////////////////////
    Json::Value js;
    Json::CharReaderBuilder parser;
    parser["collectComments"] = false; //important since we want to write to netcdf
    std::string errs;
    if( argc != 3)
    {
        if(rank==0)std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else
    {
        std::ifstream is(argv[1]);
        parseFromStream( parser, is, &js, &errs);
    }
    const Parameters p( js);
    if(rank==0)p.display( std::cout);

    ////////////////////////////////set up computations///////////////////////////
    dg::MPIGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y, comm);
    dg::MPIGrid2d grid_out( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y, comm);
    //create RHS
    toefl::Explicit< dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec > test( grid, p);
    toefl::Implicit< dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec > diffusion( grid, p.nu);
    //////////////////create initial vector///////////////////////////////////////
    dg::Gaussian g( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp);
    std::vector<dg::MDVec> y0(2, dg::evaluate( g, grid)), y1(y0); // n_e' = gaussian
    dg::blas2::symv( test.gamma(), y0[0], y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    {
        dg::MDVec v2d = dg::create::inv_weights(grid);
        dg::blas2::symv( v2d, y0[1], y0[1]);
    }
    if( p.equations == "gravity_local" || p.equations == "gravity_global" || p.equations == "drift_global" ){
        y0[1] = dg::evaluate( dg::zero, grid);
    }
    //////////////////initialisation of timestepper and first step///////////////////
    double time = 0;
    dg::Karniadakis< std::vector<dg::MDVec> > stepper( y0, y0[0].size(), p.eps_time);
    stepper.init( test, diffusion, time, y0, p.dt);
    y1 = y0;
    /////////////////////////////set up netcdf/////////////////////////////////////
    file::NC_Error_Handle err;
    int ncid; MPI_Info info = MPI_INFO_NULL;
    err = nc_create_par( argv[2],NC_NETCDF4|NC_MPIIO|NC_CLOBBER,comm,info, &ncid);
    std::string input = js.toStyledString();
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids[3], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, grid_out.global());
    //field IDs
    std::string names[4] = {"electrons", "ions", "potential", "vorticity"};
    int dataIDs[4];
    for( unsigned i=0; i<4; i++){
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    //energy IDs
    int EtimeID, EtimevarID;
    err = file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, dissID, dEdtID;
    err = nc_def_var( ncid, "energy",      NC_DOUBLE, 1, &EtimeID, &energyID);
    err = nc_def_var( ncid, "mass",        NC_DOUBLE, 1, &EtimeID, &massID);
    err = nc_def_var( ncid, "dissipation", NC_DOUBLE, 1, &EtimeID, &dissID);
    err = nc_def_var( ncid, "dEdt",        NC_DOUBLE, 1, &EtimeID, &dEdtID);
      //mpi specific part
      for(unsigned i=0; i<4; i++)
          err = nc_var_par_access( ncid, dataIDs[i], NC_COLLECTIVE);
      err = nc_var_par_access( ncid, tvarID, NC_COLLECTIVE);
      err = nc_var_par_access( ncid, EtimevarID, NC_COLLECTIVE);
      err = nc_var_par_access( ncid, energyID, NC_COLLECTIVE);
      err = nc_var_par_access( ncid, massID, NC_COLLECTIVE);
      err = nc_var_par_access( ncid, dissID, NC_COLLECTIVE);
      err = nc_var_par_access( ncid, dEdtID, NC_COLLECTIVE);
      err = nc_enddef(ncid);
      int dims[2],  coords[2];
      MPI_Cart_get( comm, 2, dims, periods, coords);
      dg::MDVec transfer( dg::evaluate( dg::zero, grid));
    ///////////////////////////////////first output/////////////////////////
    size_t count[3] = {1, grid_out.n()*grid_out.local().Ny(), grid_out.n()*grid_out.local().Nx()};
    size_t start[3] = {0, coords[1]*count[1], coords[0]*count[2]};
    size_t Ecount[] = {1};
    size_t Estart[] = {0};
    std::vector<dg::DVec> transferD(4, dg::evaluate(dg::zero, grid_out.local()));
    dg::HVec transferH(dg::evaluate(dg::zero, grid_out.local()));
    dg::IDMatrix interpolate = dg::create::interpolation( grid_out.local(), grid.local());
    dg::blas2::symv( interpolate, y1[0].data(), transferD[0]);
    dg::blas2::symv( interpolate, y1[1].data(), transferD[1]);
    dg::blas2::symv( interpolate, test.potential()[0].data(), transferD[2]);
    dg::blas2::symv( diffusion.laplacianM(), test.potential()[0], transfer);
    dg::blas2::symv( interpolate, transfer.data(), transferD[3]);
    for( unsigned k=0; k<4; k++)
    {
        dg::blas1::transfer( transferD[k], transferH);
        err = nc_put_vara_double( ncid, dataIDs[k], start, count, transferH.data() );
    }
    err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    //err = nc_close(ncid);
    ///////////////////////////////////////Timeloop/////////////////////////////////
    const double mass0 = test.mass(), mass_blob0 = mass0 - grid.global().lx()*grid.global().ly();
    double E0 = test.energy(), E1 = 0, diff = 0;
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
            stepper.step( test, diffusion, time, y1);
            //store accuracy details
            {
                if(rank==0)std::cout << "(m_tot-m_0)/m_0: "<< (test.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = test.energy();
                diff = (E1 - E0)/p.dt;
                double diss = test.energy_diffusion( );
                if(rank==0)std::cout << "diff: "<< diff<<" diss: "<<diss<<"\t";
                if(rank==0)std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";
            }
            Estart[0] += 1;
            {
                //err = nc_open(argv[2], NC_WRITE, &ncid);
                double ener=test.energy(), mass=test.mass(), diff=test.mass_diffusion(), dEdt=test.energy_diffusion();
                err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
                err = nc_put_vara_double( ncid, energyID,   Estart, Ecount, &ener);
                err = nc_put_vara_double( ncid, massID,     Estart, Ecount, &mass);
                err = nc_put_vara_double( ncid, dissID,     Estart, Ecount, &diff);
                err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount, &dEdt);
                //err = nc_close(ncid);
            }
        }
        //////////////////////////write fields////////////////////////
        start[0] = i;
        dg::blas2::symv( interpolate, y1[0].data(), transferD[0]);
        dg::blas2::symv( interpolate, y1[1].data(), transferD[1]);
        dg::blas2::symv( interpolate, test.potential()[0].data(), transferD[2]);
        dg::blas2::symv( diffusion.laplacianM(), test.potential()[0], transfer);
        dg::blas2::symv( interpolate, transfer.data(), transferD[3]);
        //err = nc_open(argv[2], NC_WRITE, &ncid);
        for( int k=0;k<4; k++)
        {
            dg::blas1::transfer( transferD[k], transferH);
            err = nc_put_vara_double( ncid, dataIDs[k], start, count, transferH.data() );
        }
        err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        //err = nc_close(ncid);

#ifdef DG_BENCHMARK
        ti.toc();
        step+=p.itstp;
        if(rank==0)std::cout << "\n\t Step "<<step <<" of "<<p.itstp*p.maxout <<" at time "<<time;
        if(rank==0)std::cout << "\n\t Average time for one step: "<<ti.diff()/(double)p.itstp<<"s\n\n"<<std::flush;
#endif//DG_BENCHMARK
    }
    }
    catch( dg::Fail& fail) {
        if(rank==0)std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
        if(rank==0)std::cerr << "Does Simulation respect CFL condition?\n";
    }
    t.toc();
    unsigned hour = (unsigned)floor(t.diff()/3600);
    unsigned minute = (unsigned)floor( (t.diff() - hour*3600)/60);
    double second = t.diff() - hour*3600 - minute*60;
    if(rank==0)std::cout << std::fixed << std::setprecision(2) <<std::setfill('0');
    if(rank==0)std::cout <<"Computation Time \t"<<hour<<":"<<std::setw(2)<<minute<<":"<<second<<"\n";
    if(rank==0)std::cout <<"which is         \t"<<t.diff()/p.itstp/p.maxout<<"s/step\n";
    nc_close(ncid);
    MPI_Finalize();

    return 0;

}

