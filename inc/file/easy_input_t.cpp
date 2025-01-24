#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#ifdef WITH_MPI
#include <mpi.h>
#endif

#include "catch2/catch.hpp"

#include "dg/algorithm.h"
#define _FILE_INCLUDED_BY_DG_
#include "easy_dims.h"
#include "easy_input.h"

static double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
static double function( double x, double y){return sin(x)*sin(y);}

TEST_CASE( "Easy input")
{
#ifdef WITH_MPI
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    MPI_Comm comm;
    //create a grid and some data
    int dims[3] = {0,0,0};
    MPI_Dims_create( size, 3, dims);
    std::stringstream ss;
    ss<< dims[0]<<" "<<dims[1]<<" "<<dims[2];
    dg::mpi_init3d( dg::PER, dg::PER, dg::PER, comm, ss);
#endif
#ifdef WITH_MPI
    std::string filename = "testmpi.nc";
#else
    std::string filename = "test.nc";
#endif
    INFO( "READ A TIME-DEPENDENT SCALAR, SCALAR FIELD AND SUB-FIELDS FROM NETCDF4 FILE "
                       << filename<<"\n");
    double Tmax=2.*M_PI;
    double NT = 10;
    double h = Tmax/NT;
    double x0 = 0., x1 = 2.*M_PI;
    dg::x::CartesianGrid3d grid( x0,x1,x0,x1,x0,x1,3,4,4,3, dg::PER, dg::PER, dg::PER
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
    INFO("Dimensions do not exist!\n");
    CHECK( exists);

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
}
