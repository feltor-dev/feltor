#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "dg/algorithm.h"
#define _FILE_INCLUDED_BY_DG_
#include "easy_dims.h"
#include "easy_input.h"

double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
double function( double x, double y){return sin(x)*sin(y);}

int main(int argc, char* argv[])
{
#ifdef WITH_MPI
    dg::mpi_init( argc, argv);
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
    DG_RANK0 std::cout << "READ A TIME-DEPENDENT SCALAR, SCALAR FIELD AND SUB-FIELDS TO NETCDF4 FILE "
                       << filename<<"\n";
    DG_RANK0 std::cout << "THEN READ IT BACK IN AND TEST EQUALITY\n";
    double Tmax=2.*M_PI;
    double NT = 10;
    double h = Tmax/NT;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::CartesianGrid3d grid( x0,x1,x0,x1,x0,x1,3,10,10,20, dg::PER, dg::PER, dg::PER
#ifdef WITH_MPI
    , comm
#endif
    );
    dg::ClonePtr<dg::x::aGeometry2d> perp_grid_ptr = grid.perp_grid();
    dg::x::Grid2d grid2d = (dg::x::Grid2d)*perp_grid_ptr;
    dg::x::HVec data = dg::evaluate( function, grid);

    //create NetCDF File
    int ncid;
    dg::file::NC_Error_Handle err;
    err = nc_open( filename.data(), 0, &ncid);

    int dim_ids[4], tvarID;
    bool exists = dg::file::check_dimensions( ncid, dim_ids, &tvarID, grid);
    if( ! exists)
        std::cerr << "Dimensions do not exist!\n";

    int scalarID, arrayID, subArrayID;
    err = nc_inq_varid( ncid, "scalar", &scalarID);
    err = nc_inq_varid( ncid, "array", &arrayID);
    err = nc_inq_varid( ncid, "sub-array", &subArrayID);

    int staticArrayID;
    err = nc_inq_varid( ncid, "static-array", &staticArrayID);
    dg::file::get_var( ncid, staticArrayID, grid, data);

    for(unsigned i=0; i<=NT; i++)
    {
        DG_RANK0 std::cout<<"Read timestep "<<i<<"\n";
        double time = i*h;
        const size_t Tstart = i;
        data = dg::evaluate( function, grid);
        auto sliced_data = dg::split( data, grid); // a vector of views
        double energy;
        //read scalar data point
        dg::file::get_vara( ncid, scalarID, Tstart, dg::x::Grid0d(), energy);
        //read array
        dg::x::HVec temp(data);
        dg::file::get_vara( ncid, arrayID, Tstart, grid, temp);
        //read sub array
        dg::file::get_vara( ncid, subArrayID, Tstart, grid2d, sliced_data[0]);
#ifdef MPI_VERSION
        DG_RANK0 std::cout << "data "<<temp.data()[0]<<" dataP "<<sliced_data[0].data().data()[0]<<"\n";
#else
        std::cout << "data "<<temp[0]<<" dataP "<<sliced_data[0].data()[0]<<"\n";
#endif
        //read time
        dg::file::get_vara( ncid, tvarID, Tstart, dg::x::Grid0d(), time);
        DG_RANK0 std::cout << "At time "<<time<<"\n";
    }

    DG_RANK0 err = nc_close(ncid);
#ifdef WITH_MPI
    MPI_Finalize();
#endif
    return 0;
}
