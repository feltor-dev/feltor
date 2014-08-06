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

int main()
{
    double Tmax=2.*M_PI;
    double NT = 100;
    double h = Tmax/NT;
    dg::Grid1d<double> gx( 0, 2.*M_PI, 3, 10);
    dg::Grid1d<double> gy( 0, 2.*M_PI, 3, 10);
    dg::Grid1d<double> gz( 0, 2.*M_PI, 1, 20);
    dg::Grid3d<double> g( gx, gy, gz);
    std::string hello = "Hello world\n";
    thrust::host_vector<double> data = dg::evaluate( function, g);
    int ncid, retval;
    //retval = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid); //for netcdf4
    file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_CLOBBER, &ncid);
    err = nc_put_att_text( ncid, NC_GLOBAL, "input", hello.size(), hello.data());

    int dim_ids[4], tvarID;
    err = file::define_dimensions( ncid, dim_ids, &tvarID, g);
    int dataID;
    err = nc_def_var( ncid, "data", NC_DOUBLE, 4, dim_ids, &dataID);
    err = nc_enddef( ncid);
    size_t count[4] = {1, g.Nz(), g.n()*g.Ny(), g.n()*g.Nx()};
    size_t start[4] = {0, 0, 0, 0};
    for(unsigned i=0; i<=NT; i++)
    {
        double time = i*h;
        const size_t Tcount = 1;
        const size_t Tstart = i;
        data = dg::evaluate( function, g);
        dg::blas1::scal( data, cos( time));
        start[0] = i;
        //write dataset (one timeslice)
        err = nc_put_vara_double( ncid, dataID, start, count, data.data());
        err = nc_put_vara_double( ncid, tvarID, &Tstart, &Tcount, &time);
    }
    err = nc_close(ncid);
    return 0;
}
