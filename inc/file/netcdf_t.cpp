#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#include "writer.h"
#include "reader.h"

double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
double gradientX(double x, double y, double z){return cos(x)*sin(y)*cos(z);}
double gradientY(double x, double y, double z){return sin(x)*cos(y)*cos(z);}
double gradientZ(double x, double y, double z){return -sin(x)*sin(y)*sin(z);}


std::vector<dg::file::Record<void(dg::x::DVec&, const dg::x::Grid3d&, double)>> records = {
    {"vectorX", "X-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", "Y-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorZ", "Z-component of vector",
        [] ( dg::x::DVec& resultD, const dg::x::Grid3d& g, double time){
            resultD = dg::evaluate( gradientZ, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};

int main(int argc, char* argv[])
{
#ifdef WITH_MPI
    MPI_Init( &argc, &argv);
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm;
    //create a grid and some data
    if( size != 4){ std::cerr << "Please run with 4 threads!\n"; return -1;}
    std::stringstream ss;
    ss<< "2 1 2";
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);
#endif
#ifdef WITH_MPI
    std::string filename = "testmpi.nc";
#else
    std::string filename = "test.nc";
#endif
    DG_RANK0 std::cout << "WRITE A TIMEDEPENDENT SCALAR, SCALAR FIELD, AND VECTOR FIELD TO NETCDF4 FILE "
                       << filename<<"\n";
    double Tmax=2.*M_PI;
    double NT = 10;
    double h = Tmax/NT;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::Grid3d grid( x0,x1,x0,x1,x0,x1,3,20,20,20
#ifdef WITH_MPI
    , comm
#endif
    );
    //create NetCDF File
    int ncid;
    dg::file::NC_Error_Handle err;
    DG_RANK0 err = nc_create( filename.data(), NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4

    dg::file::Writer<dg::x::Grid0d> write0d( ncid, {}, {"time"});
    write0d.def( "time"); // just to check that it works
    write0d.def( "Energy");
    dg::file::WriteRecordsList<dg::x::Grid3d> write3d( ncid, grid, {"time", "z", "y", "x"}, records);
    auto grid_out = grid;
    grid_out.multiplyCellNumbers( 0.5, 0.5);
    int grpid =0;
    DG_RANK0 err = nc_def_grp( ncid, "projected", &grpid);
    dg::file::ProjectRecordsList<dg::x::Grid3d, dg::x::DMatrix, dg::x::DVec> project3d( grpid, grid, grid_out, {"ptime", "zr", "yr", "xr"}, records);
    dg::file::Writer<dg::x::Grid0d> project0d( grpid, {}, {"ptime"});

    for(unsigned i=0; i<=NT; i++)
    {
        DG_RANK0 std::cout<<"Write timestep "<<i<<"\n";
        double time = i*h;
        auto data = dg::evaluate( function, grid);
        dg::blas1::scal( data, cos( time));
        double energy = dg::blas1::dot( data, data);
        if( i%2)
            write0d.put( "Energy", energy, i);
        write0d.put( "time", time, i);
        write3d.write( records, grid, time);
        project0d.put( "ptime", time, i);
        project0d.put( "ptime", time, i); // test if values can be overwritten
        project3d.write( records, grid, time);
    }

    DG_RANK0 err = nc_close(ncid);
#ifdef WITH_MPI
    MPI_Barrier( MPI_COMM_WORLD);
#endif

    // open and read back in
    err = nc_open ( filename.data(), 0, &ncid); // all processes read
    err = nc_inq_grp_ncid( ncid, "projected", &grpid);
    dg::file::Reader<dg::x::Grid0d> read0d( ncid, {}, {"time"});
    dg::file::Reader<dg::x::Grid3d> read3d( ncid, grid, {"time", "z", "y", "x"});
    dg::file::Reader<dg::x::Grid3d> readP3d( grpid, grid_out, {"ptime", "zr", "yr", "xr"});
    for( auto name : read0d.names())
        DG_RANK0 std::cout << "Found 0d name "<<name<<"\n";
    for( auto name : read3d.names())
        DG_RANK0 std::cout << "Found 3d name "<<name<<"\n";
    for( auto name : readP3d.names())
        DG_RANK0 std::cout << "Found Projected 3d name "<<name<<"\n";
    unsigned num_slices = read0d.size();
    auto data = dg::evaluate( function, grid);
    auto dataP = dg::evaluate( function, grid_out);
    for(unsigned i=0; i<num_slices; i++)
    {
        DG_RANK0 std::cout<<"Read timestep "<<i<<"\n";
        double time, energy;
        read0d.get("time", time, i);
        read0d.get("Energy", energy, i);
        std::cout << "Enery "<<energy<<"\n";
        read3d.get( "vectorX", data, i);
        readP3d.get( "vectorX", dataP, i);
    }
    err = nc_close(ncid);
    assert(num_slices == NT+1);

#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
