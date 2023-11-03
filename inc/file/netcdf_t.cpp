#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#include "dg/algorithm.h"
#define _FILE_INCLUDED_BY_DG_
#include "nc_utilities.h"

double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
double gradientX(double x, double y, double z){return cos(x)*sin(y)*cos(z);}
double gradientY(double x, double y, double z){return sin(x)*cos(y)*cos(z);}
double gradientZ(double x, double y, double z){return -sin(x)*sin(y)*sin(z);}

struct Record
{
    std::string name;
    std::string long_name;
    std::function<void (dg::x::DVec&, const dg::Grid3d&, double)> function;
};

std::vector<Record> records_list = {
    {"vectorX", "X-component of vector",
        [] ( dg::x::DVec& resultD, const dg::Grid3d& g, double time){
            resultD = dg::evaluate( gradientX, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorY", "Y-component of vector",
        [] ( dg::x::DVec& resultD, const dg::Grid3d& g, double time){
            resultD = dg::evaluate( gradientY, g);
            dg::blas1::scal( resultD, cos( time));
        }
    },
    {"vectorZ", "Z-component of vector",
        [] ( dg::x::DVec& resultD, const dg::Grid3d& g, double time){
            resultD = dg::evaluate( gradientZ, g);
            dg::blas1::scal( resultD, cos( time));
        }
    }
};

typedef thrust::host_vector<double> HVec;

int main()
{
    std::cout << "WRITE A TIMEDEPENDENT SCALAR, SCALAR FIELD, AND VECTOR FIELD TO NETCDF4 FILE test.nc\n";
    double Tmax=2.*M_PI;
    double NT = 10;
    double h = Tmax/NT;
    dg::Grid1d gx( 0, 2.*M_PI, 3, 10);
    dg::Grid1d gy( 0, 2.*M_PI, 3, 10);
    dg::Grid1d gz( 0, 2.*M_PI, 1, 20);
    dg::Grid3d g( gx, gy, gz);
    std::string hello = "Hello world\n";
    thrust::host_vector<double> data = dg::evaluate( function, g);
    int ncid;
    dg::file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4
    err = nc_put_att_text( ncid, NC_GLOBAL, "input", hello.size(), hello.data());

    int dim_ids[4], tvarID;
    err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, g);

    int dataID, scalarID;
    err = nc_def_var( ncid, "data", NC_DOUBLE, 1, dim_ids, &dataID);
    err = nc_def_var( ncid, "scalar", NC_DOUBLE, 4, dim_ids, &scalarID);

    dg::file::WriteRecordsList<4> records( ncid, dim_ids, records_list);
    size_t count[4] = {g.nz(), g.Nz(), g.ny()*g.Ny(), g.nx()*g.Nx()};
    size_t start[4] = {0, 0, 0, 0};
    for(unsigned i=0; i<=NT; i++)
    {
        double time = i*h;
        const size_t Tcount = 1;
        const size_t Tstart = i;
        start[0] = i;
        data = dg::evaluate( function, g);
        dg::blas1::scal( data, cos( time));
        double energy = dg::blas1::dot( data, data);
        //write scalar data point
        err = nc_put_vara_double( ncid, dataID, start, count, &energy);
        //write scalar field
        err = nc_put_vara_double( ncid, scalarID, start, count, data.data());
        //write vector field
        records.write( ncid, g, records_list, g, time);
        //write time
        err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
    }

    err = nc_close(ncid);
    return 0;
}
