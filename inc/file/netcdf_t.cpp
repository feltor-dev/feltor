#include <iostream>
#include <string>
#include <netcdf.h>
#include <cmath>

#include "dg/blas.h"
#include "dg/backend/grid.h"
#include "dg/backend/evaluation.cuh"
#include "dg/backend/weights.cuh"
#include "nc_utilities.h"

double function( double x, double y, double z){return sin(x)*sin(y)*cos(z);}
double gradientX(double x, double y, double z){return cos(x)*sin(y)*cos(z);}
double gradientY(double x, double y, double z){return sin(x)*cos(y)*cos(z);}
double gradientZ(double x, double y, double z){return -sin(x)*sin(y)*sin(z);}

typedef thrust::host_vector<double> HVec; 

int main()
{
    std::cout << "WRITE A TIMEDEPENDENT SCALAR, SCALAR FIELD, AND VECTOR FIELD TO A NETCDF4 FILE\n";
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
    file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4
    //err = nc_create( "test.nc", NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "input", hello.size(), hello.data());

    int dim_ids[4], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, g);

    int dataID, scalarID, vectorID[3];
    err = nc_def_var( ncid, "data", NC_DOUBLE, 1, dim_ids, &dataID);
    err = nc_def_var( ncid, "scalar", NC_DOUBLE, 4, dim_ids, &scalarID);
    err = nc_def_var( ncid, "vectorX", NC_DOUBLE, 4, dim_ids, &vectorID[0]);
    err = nc_def_var( ncid, "vectorY", NC_DOUBLE, 4, dim_ids, &vectorID[1]);
    err = nc_def_var( ncid, "vectorZ", NC_DOUBLE, 4, dim_ids, &vectorID[2]);
    err = nc_enddef( ncid);
    size_t count[4] = {1, g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
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
        HVec dataX = dg::evaluate( gradientX, g);
        HVec dataY = dg::evaluate( gradientY, g);
        HVec dataZ = dg::evaluate( gradientZ, g);
        dg::blas1::scal( dataX, cos( time));
        dg::blas1::scal( dataY, cos( time));
        dg::blas1::scal( dataZ, cos( time));
        err = nc_put_vara_double( ncid, vectorID[0], start, count, dataX.data());
        err = nc_put_vara_double( ncid, vectorID[1], start, count, dataY.data());
        err = nc_put_vara_double( ncid, vectorID[2], start, count, dataZ.data());
        //write time
        err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
    }

    err = nc_close(ncid);
    return 0;
}
