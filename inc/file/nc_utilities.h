#pragma once

#include <netcdf.h>
#include "thrust/host_vector.h"

#include "dg/backend/grid.h"
#include "dg/backend/weights.cuh"

namespace file
{

/**
 * @brief Define dimensions and associate values in NetCDF-file
 *
 * @param ncid file ID 
 * @param dimsIDs (write - only) 4D array of dimension IDs (time, z,y,x) 
 * @param tvarID (write - only) The ID of the time variable
 * @param g The grid from which to derive the dimensions
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::Grid3d<double>& g)
{
    dg::Grid1d<double> gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d<double> gy( g.y0(), g.y1(), g.n(), g.Ny());
    dg::Grid1d<double> gz( g.z0(), g.z1(), 1, g.Nz());
    thrust::host_vector<double> pointsX = dg::create::abscissas( gx);
    thrust::host_vector<double> pointsY = dg::create::abscissas( gy);
    thrust::host_vector<double> pointsZ = dg::create::abscissas( gz);
    int retval;
    int xid, yid, zid, tid;
    if( (retval = nc_def_dim( ncid, "x", gx.size(), &xid)) ) { return retval;}
    if( (retval = nc_def_dim( ncid, "y", gy.size(), &yid)) ){ return retval;}
    if( (retval = nc_def_dim( ncid, "z", gz.size(), &zid)) ){ return retval;}
    if( (retval = nc_def_dim( ncid, "time", NC_UNLIMITED, &tid)) ){ return retval;}
    int xvarID, yvarID, zvarID;
    if( (retval = nc_def_var( ncid, "x", NC_DOUBLE, 1, &xid, &xvarID))){return retval;}
    if( (retval = nc_def_var( ncid, "y", NC_DOUBLE, 1, &yid, &yvarID))){return retval;}
    if( (retval = nc_def_var( ncid, "z", NC_DOUBLE, 1, &zid, &zvarID))){return retval;}
    if( (retval = nc_def_var( ncid, "time", NC_DOUBLE, 1, &tid, tvarID)) ){ return retval;}
    std::string t = "time since start"; //needed for paraview to recognize timeaxis
    if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
    dimsIDs[0] = tid, dimsIDs[1] = zid, dimsIDs[2] = yid, dimsIDs[3] = xid;
    if( (retval = nc_enddef(ncid)) ) {return retval;} //not necessary for NetCDF4 files
    //write coordinate variables
    if( (retval = nc_put_var_double( ncid, xid, pointsX.data())) ){ return retval;}
    if( (retval = nc_put_var_double( ncid, yid, pointsY.data())) ){ return retval;}
    if( (retval = nc_put_var_double( ncid, zid, pointsZ.data())) ){ return retval;}
    if( (retval = nc_redef(ncid)) ) {return retval;} //not necessary for NetCDF4 files
    return retval;
}

/**
 * @brief Define dimensions and associate values in NetCDF-file
 *
 * @param ncid file ID 
 * @param dimsIDs (write - only) 3D array of dimension IDs (time, y,x) 
 * @param tvarID (write - only) The ID of the time variable
 * @param g The 2d grid from which to derive the dimensions
 *
 * @return if anything goes wrong it returns the netcdf code, else SUCCESS
 * @note File stays in define mode
 */
int define_dimensions( int ncid, int* dimsIDs, int* tvarID, const dg::Grid2d<double>& g)
{
    dg::Grid1d<double> gx( g.x0(), g.x1(), g.n(), g.Nx());
    dg::Grid1d<double> gy( g.y0(), g.y1(), g.n(), g.Ny());
    thrust::host_vector<double> pointsX = dg::create::abscissas( gx);
    thrust::host_vector<double> pointsY = dg::create::abscissas( gy);
    int retval;
    int xid, yid, tid;
    if( (retval = nc_def_dim( ncid, "x", gx.size(), &xid)) ) { return retval;}
    if( (retval = nc_def_dim( ncid, "y", gy.size(), &yid)) ){ return retval;}
    if( (retval = nc_def_dim( ncid, "time", NC_UNLIMITED, &tid)) ){ return retval;}
    int xvarID, yvarID, zvarID;
    if( (retval = nc_def_var( ncid, "x", NC_DOUBLE, 1, &xid, &xvarID))){return retval;}
    if( (retval = nc_def_var( ncid, "y", NC_DOUBLE, 1, &yid, &yvarID))){return retval;}
    if( (retval = nc_def_var( ncid, "time", NC_DOUBLE, 1, &tid, tvarID))){return retval;}
    std::string t = "time since start"; //needed for paraview to recognize timeaxis
    if( (retval = nc_put_att_text(ncid, *tvarID, "units", t.size(), t.data())) ){ return retval;}
    dimsIDs[0] = tid, dimsIDs[1] = yid, dimsIDs[2] = xid;
    if( (retval = nc_enddef(ncid)) ) {return retval;} //not necessary for NetCDF4 files
    //write coordinate variables
    if( (retval = nc_put_var_double( ncid, xid, pointsX.data())) ){ return retval;}
    if( (retval = nc_put_var_double( ncid, yid, pointsY.data())) ){ return retval;}
    if( (retval = nc_redef(ncid))) {return retval;} //not necessary for NetCDF4 files
    return retval;
}

} //namespace file
