#include <iostream>
#include <iomanip>
#include <vector>

#include <mpi.h> //activate mpi

#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "toeflR.cuh"
#include "parameters.h"


/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - writes outputs to a given outputfile using hdf5. 
        density fields are the real densities in XSPACE ( not logarithmic values)
*/

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
    if( argc != 3)
    {
        if(rank==0)std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [outputfile]\n";
        return -1;
    }
    else
        dg::file::file2Json( argv[1], js, dg::file::comments::are_forbidden);
    std::string input = js.toStyledString(); //save input without comments, which is important if netcdf file is later read by another parser
    const Parameters p( js);
    if(rank==0)p.display( std::cout);

    ////////////////////////////////set up computations///////////////////////////
    dg::MPIGrid2d grid( 0, p.lx, 0, p.ly, p.n, p.Nx, p.Ny, p.bc_x, p.bc_y, comm);
    dg::MPIGrid2d grid_out( 0., p.lx, 0.,p.ly, p.n_out, p.Nx_out, p.Ny_out, p.bc_x, p.bc_y, comm);  
    //create RHS 
    ep::ToeflR< dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec > test( grid, p); 
    ep::Diffusion<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec> diffusion( grid, p.nu);
    //create initial vector
    dg::Gaussian g( p.posX*p.lx, p.posY*p.ly, p.sigma, p.sigma, p.amp); 
    dg::MDVec gauss = dg::evaluate( g, grid);
    std::vector<dg::MDVec> y0(2, gauss), y1(y0); // n_e' = gaussian
    dg::Helmholtz<dg::CartesianMPIGrid2d, dg::MDMatrix, dg::MDVec> gamma(  grid, -0.5*p.tau[0]*p.mu[0], dg::centered);
    dg::blas2::symv( gamma, gauss, y0[0]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    gamma.alpha() = -0.5*p.tau[1]*p.mu[1];
    dg::blas2::symv( gamma, gauss, y0[1]); // n_e = \Gamma_i n_i -> n_i = ( 1+alphaDelta) n_e' + 1
    //////////////////initialisation of timestepper and first step///////////////////
    double time = 0;
    dg::ImExMultistep_s< std::vector<dg::MDVec> > karniadakis( "ImEx-BDF-3-3", y0, y0[0].size(), 1e-9);
    karniadakis.init( test, diffusion, 0., y0, p.dt);
    y0.swap( y1); //y1 now contains value at zero time
    /////////////////////////////set up netcdf/////////////////////////////////////
    dg::file::NC_Error_Handle err;
    int ncid;
    if(rank==0)err = nc_create( argv[2],NC_NETCDF4|NC_CLOBBER, &ncid);
    if(rank==0)err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    int dim_ids[3], tvarID;
    if(rank==0)err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, grid_out.global());
    //field IDs
    std::string names[4] = {"electrons", "positrons", "potential", "vorticity"}; 
    int dataIDs[4]; 
    for( unsigned i=0; i<4; i++){
        if(rank==0)err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    //energy IDs
    int EtimeID, EtimevarID;
    if(rank==0)err = dg::file::define_time( ncid, "energy_time", &EtimeID, &EtimevarID);
    int energyID, massID, dissID, dEdtID;
    if(rank==0)err = nc_def_var( ncid, "energy",      NC_DOUBLE, 1, &EtimeID, &energyID);
    if(rank==0)err = nc_def_var( ncid, "mass",        NC_DOUBLE, 1, &EtimeID, &massID);
    if(rank==0)err = nc_def_var( ncid, "dissipation", NC_DOUBLE, 1, &EtimeID, &dissID);
    if(rank==0)err = nc_def_var( ncid, "dEdt",        NC_DOUBLE, 1, &EtimeID, &dEdtID);
    if(rank==0)err = nc_enddef(ncid);

    ///////////////////////////////////first output/////////////////////////
    int dims[2],  coords[2];
    MPI_Cart_get( comm, 2, dims, periods, coords);
    size_t count[3] = {1, grid_out.n()*grid_out.local().Ny(), grid_out.n()*grid_out.local().Nx()};
    size_t start[3] = {0, coords[1]*count[1], coords[0]*count[2]};
    size_t Estart[] = {0};
    size_t Ecount[] = {1};
    dg::MDVec transfer( dg::evaluate(dg::zero, grid));
    dg::MDVec transferD( dg::evaluate(dg::zero, grid_out));
    dg::MHVec transferH( dg::evaluate(dg::zero, grid_out));
    dg::MIDMatrix interpolate = dg::create::interpolation( grid_out, grid);
    for( unsigned i=0; i<2; i++)
    {
        dg::blas2::gemv( interpolate, y0[i], transferD);
        dg::assign( transferD, transferH);
        dg::file::put_vara_double( ncid, dataIDs[i], 0, grid_out, transferH);
    }
    //pot
    transfer = test.potential();
    dg::blas2::gemv( interpolate, transfer, transferD);
    dg::assign( transferD, transferH);
    dg::file::put_vara_double( ncid, dataIDs[2], 0, grid_out, transferH );
    //Vor
    transfer = test.potential();
    dg::blas2::gemv( diffusion.laplacianM(), transfer, y1[1]);
    dg::blas2::gemv( interpolate,y1[1], transferD);
    dg::assign( transferD, transferH);
    dg::file::put_vara_double( ncid, dataIDs[3], 0, grid_out, transferH );
    if(rank==0)err = nc_put_vara_double( ncid, tvarID, start, count, &time);
    if(rank==0)err = nc_close(ncid);
    ///////////////////////////////////////Timeloop/////////////////////////////////
    const double mass0 = test.mass(), mass_blob0 = mass0 - grid.lx()*grid.ly();
    double E0 = test.energy(), energy0 = E0, E1 = 0, diff = 0;
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
            karniadakis.step( test, diffusion, time, y0);
            y0.swap( y1); //attention on -O3 ?
            //store accuracy details
            {
                if(rank==0)std::cout << "(m_tot-m_0)/m_0: "<< (test.mass()-mass0)/mass_blob0<<"\t";
                E0 = E1;
                E1 = test.energy();
                diff = (E1 - E0)/p.dt;
                double diss = test.energy_diffusion( );
                if(rank==0)std::cout << "(E_tot-E_0)/E_0: "<< (E1-energy0)/energy0<<"\t";
                if(rank==0)std::cout << "Accuracy: "<< 2.*(diff-diss)/(diff+diss)<<"\n";
            }
            time+=p.dt;
            Estart[0] += 1;
            {
                double ener=test.energy(), mass=test.mass(), diff=test.mass_diffusion(), dEdt=test.energy_diffusion();
                if(rank==0)
                {
                    err = nc_open(argv[2], NC_WRITE, &ncid);
                    err = nc_put_vara_double( ncid, EtimevarID, Estart, Ecount, &time);
                    err = nc_put_vara_double( ncid, energyID,   Estart, Ecount, &ener);
                    err = nc_put_vara_double( ncid, massID,     Estart, Ecount, &mass);
                    err = nc_put_vara_double( ncid, dissID,     Estart, Ecount, &diff);
                    err = nc_put_vara_double( ncid, dEdtID,     Estart, Ecount, &dEdt);
                    err = nc_close(ncid);
                }
            }
        }
        //////////////////////////write fields////////////////////////
        if(rank==0)err = nc_open(argv[2], NC_WRITE, &ncid);
        start[0] = i;
        for( unsigned j=0; j<2; j++)
        {
            dg::blas2::gemv( interpolate, y0[j], transferD);
            dg::assign( transferD, transferH);
            dg::file::put_vara_double( ncid, dataIDs[j], i, grid, transferH);
        }
        transfer = test.potential();
        dg::blas2::gemv( interpolate, transfer, transferD);
        dg::assign( transferD, transferH);
        dg::file::put_vara_double( ncid, dataIDs[2], i, grid, transferH );
        transfer = test.potential();
        dg::blas2::gemv( diffusion.laplacianM(), transfer, y1[1]);        //correct?    
        dg::blas2::gemv( interpolate,y1[1], transferD);
        dg::assign( transferD, transferH);
        dg::file::put_vara_double( ncid, dataIDs[3], i, grid, transferH );
        if(rank==0)err = nc_put_vara_double( ncid, tvarID, start, count, &time);
        if(rank==0)err = nc_close(ncid);

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
    MPI_Finalize();

    return 0;

}

