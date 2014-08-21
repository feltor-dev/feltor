#include <iostream>
#include <iomanip>
#include <vector>
#include <sstream>
#include <cmath>

#include <mpi.h> //activate mpi
#include <netcdf_par.h>

#include "dg/algorithm.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/xspacelib.cuh"
#include "dg/backend/interpolation.cuh"
#include "file/read_input.h"
#include "file/nc_utilities.h"


#include "feltor.cuh"
#include "bessel.h"
#include "parameters.h"
#include "geometry.h"

/*
   - reads parameters from input.txt or any other given file, 
   - integrates the ToeflR - functor and 
   - writes outputs to a given outputfile using hdf5. 
        density fields are the real densities in XSPACE ( not logarithmic values)
*/

const unsigned k = 3;//!< a change in k needs a recompilation

int main( int argc, char* argv[])
{
    ////////////////////////////////setup MPI///////////////////////////////
    MPI_Init( &argc, &argv);
    int periods[3] = {false, false, true}; //non-, non-, periodic
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    int np[3];
    if(rank==0)
    {
        std::cin>> np[0] >> np[1] >>np[2];
        std::cout << "Computing with "<<np[0]<<" "<<np[1]<<" "<<np[2]<<std::endl;
        std::cout << "Size is "<<size<<std::endl;
        assert( size == np[0]*np[1]*np[2]);
    }
    MPI_Bcast( np, 3, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Comm comm;
    MPI_Cart_create( MPI_COMM_WORLD, 3, np, periods, true, &comm);
    ////////////////////////Parameter initialisation//////////////////////////
    std::vector<double> v,v3;
    std::string input, geom;
    if( argc != 4)
    {
        if(rank==0)std::cerr << "ERROR: Wrong number of arguments!\nUsage: "<< argv[0]<<" [inputfile] [geomfile] [outputfile]\n";
        return -1;
    }
    else 
    {
        v = file::read_input( argv[1]);
        input = file::read_file( argv[1]);
    }
    const eule::Parameters p( v);
    if(rank==0) p.display( std::cout);

    ////////////////////////////////set up computations///////////////////////////
    try{ v3 = file::read_input( argv[2]); }
    catch (toefl::Message& m) {  
        if(rank==0) m.display(); 
        geom = file::read_file( argv[2]);
        if(rank==0) std::cout << geom << std::endl;
        return -1;
    }
    geom = file::read_file( argv[2]);
    const solovev::GeomParameters gp(v3);
    if(rank==0) gp.display( std::cout);
    double Rmin=gp.R_0-(gp.boxscale)*gp.a;
    double Zmin=-(gp.boxscale)*gp.a*gp.elongation;
    double Rmax=gp.R_0+(gp.boxscale)*gp.a; 
    double Zmax=(gp.boxscale)*gp.a*gp.elongation;
   
    //Make grids: both the dimensions of grid and grid_out must be dividable by the mpi process numbers in that direction
     dg::MPI_Grid3d grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n, p.Nx, p.Ny, p.Nz, dg::DIR, dg::DIR, dg::PER, dg::cylindrical, comm);  
     dg::Grid3d<double> grid_out = dg::create::ghostless_grid( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out, p.Nz_out, comm);  
     dg::Grid3d<double> global_grid_out( Rmin,Rmax, Zmin,Zmax, 0, 2.*M_PI, p.n_out, p.Nx_out, p.Ny_out, p.Nz_out);  
     
    //create RHS 
    eule::Feltor< dg::MMatrix, dg::MVec, dg::MPrecon > feltor( grid, p,gp); 
    eule::Rolkar< dg::MMatrix, dg::MVec, dg::MPrecon > rolkar( grid, p,gp);

    //The initial field
    dg::Gaussian3d init0(gp.R_0+p.posX*gp.a, p.posY*gp.a, M_PI/p.Nz, p.sigma, p.sigma, p.sigma, p.amp);
//     dg::BathRZ init0(16,16,p.Nz,Rmin,Zmin, 30.,5.,p.amp);
//       solovev::ZonalFlow init0(gp,p.amp);
    solovev::Nprofile grad(gp); //initial profile
    
    std::vector<dg::MVec> y0(4, dg::evaluate( grad, grid)), y1(y0); 
    //damp the bath on psi boundaries 
    dg::blas1::pointwiseDot(rolkar.dampin(), dg::evaluate(init0, grid), y1[1]); //is damping on bath    
    dg::blas1::axpby( 1., y1[1], 1., y0[1]); //initialize ne
    //without FLR
    //dg::blas1::axpby( 1., y1[0], 1., y0[1]);
    //with FLR
    feltor.initializene(y0[1],y0[0]);    
    feltor.log( y0, y0, 2); 

    dg::blas1::axpby( 0., y0[2], 0., y0[2]); //set Ue = 0
    dg::blas1::axpby( 0., y0[3], 0., y0[3]); //set Ui = 0
    
    dg::Karniadakis< std::vector<dg::MVec> > karniadakis( y0, y0[0].size(), p.eps_time);
    karniadakis.init( feltor, rolkar, y0, p.dt);
    double time = 0;
    unsigned step = 0;

    /////////////////////////////set up netcdf/////////////////////////////////
    file::NC_Error_Handle err;
    int ncid;
    MPI_Info info = MPI_INFO_NULL;
    err = nc_create_par( argv[3], NC_NETCDF4|NC_MPIIO|NC_CLOBBER, comm, info, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "inputfile", input.size(), input.data());
    err = nc_put_att_text( ncid, NC_GLOBAL, "geomfile",  geom.size(), geom.data());
    int dimids[4];
    std::vector<std::string> names(5); 
    int dataIDs[names.size()], energyID, tvarID; //VARIABLE IDS
    names[0] = "electrons", names[1] = "ions", names[2] = "Ue", names[3] = "Ui";
    names[4] = "potential";
    //use global dimensionality
    err = file::define_dimensions( ncid, dimids, &tvarID, global_grid_out);
    for( unsigned i=0; i<names.size(); i++)
    {
        err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 4, dimids, &dataIDs[i]);
        err = nc_var_par_access( ncid, dataIDs[i], NC_COLLECTIVE);
    }
    nc_def_var( ncid, "energy", NC_DOUBLE, 1, dimids, &energyID);
    err = nc_var_par_access( ncid, tvarID, NC_COLLECTIVE);
    err = nc_var_par_access( ncid, energyID, NC_COLLECTIVE);
    err = nc_enddef(ncid);
    ///////////////////////////////////first output/////////////////////////
    int dims[3],  coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    size_t count[4] = {1., grid_out.Nz(), grid_out.n()*(grid_out.Ny()), grid_out.n()*(grid_out.Nx())};
    size_t start[4] = {0, coords[2]*count[1], coords[1]*count[2], coords[0]*count[3]};
    dg::MVec transferD( dg::evaluate(dg::zero, grid));
    dg::HVec transferH( dg::evaluate(dg::zero, grid_out));
    //create local interpolation matrix
    cusp::csr_matrix<int, double, cusp::host_memory> interpolate = dg::create::interpolation( grid_out, grid.local()); 
    feltor.exp( y0,y0,2); //transform to correct values
    if(rank==0)std::cout << "First write ...\n";
    for( unsigned i=0; i<4; i++)
    {
        dg::blas2::symv( interpolate, y0[i].data(), transferH);
        err = nc_put_vara_double( ncid, dataIDs[i], start, count, transferH.data() );
    }
    transferD = feltor.potential()[0];
    dg::blas2::symv( interpolate, transferD.data(), transferH);
    err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data());
    err = nc_put_vara_double( ncid, tvarID, &start[0], &count[0], &time);
    double E0 = feltor.energy(), energy0 = E0, E1 = 1, diff = 0;
    err = nc_put_vara_double( ncid, energyID, start, count, &E1);
    //err = nc_close(ncid);
    if(rank==0)std::cout << "First write successful!\n";
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
            try{ karniadakis( feltor, rolkar, y0);}
            catch( dg::Fail& fail) { 
                std::cerr << "CG failed to converge to "<<fail.epsilon()<<"\n";
                std::cerr << "Does Simulation respect CFL condition?\n";
                break;
            }
        }
        time += p.itstp*p.dt;
        start[0] = i;
        feltor.exp( y0,y0,2); //transform to correct values
        //err = nc_open_par( argv[3], NC_WRITE|NC_MPIIO, comm, info, &ncid);
        for( unsigned j=0; j<4; j++)
        {
            dg::blas2::symv( interpolate, y0[j].data(), transferH);
            err = nc_put_vara_double( ncid, dataIDs[j], start, count, transferH.data());
        }
        transferD = feltor.potential()[0];
        dg::blas2::symv( interpolate, transferD.data(), transferH);
        err = nc_put_vara_double( ncid, dataIDs[4], start, count, transferH.data() );
        err = nc_put_vara_double( ncid, tvarID, &start[0], &count[0], &time);
        E1 = feltor.energy()/energy0;
        err = nc_put_vara_double( ncid, energyID, start, count, &E1);

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
    err = nc_close(ncid);
    MPI_Finalize();

    return 0;

}

