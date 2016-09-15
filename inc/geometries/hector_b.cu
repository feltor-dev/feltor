#include <iostream>

#include "file/read_input.h"
#include "file/nc_utilities.h"

#include "dg/geometry/refined_grid.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
//#include "guenther.h"
#include "orthogonal.h"
#include "hector.h"



int main(int argc, char**argv)
{
    std::cout << "Type n, Nx, Ny\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    std::cout << "Type eps_uv \n";
    double eps_uv;
    std::cin >> eps_uv;
    std::vector<double> vin;
    try{ 
        if( argc==1)
        {
            vin = file::read_input( "geometry_params_Xpoint.txt"); 
        }
        else
        {
            vin = file::read_input( argv[1]); 
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<vin.size(); i++)
            std::cout << vin[i] << " ";
            std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    solovev::GeomParameters gp(vin);
    gp.display( std::cout);
    dg::Timer t;
    solovev::Psip psip( gp); 
    solovev::PsipR psipR( gp); 
    solovev::PsipZ psipZ( gp); 
    solovev::LaplacePsip lap( gp); 

    hector::Hector<dg::IDMatrix, dg::DMatrix, dg::DVec> hector( psip, psipR, psipZ, lap, psi_0, psi_1, gp.R_0, 0.);

    ///////////////////////////////FILE OUTPUT/////////////////////////////
    /*
    g2d_old.display();
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "hector.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    err = file::define_dimensions(  ncid, dim2d, g2d_old);
    int coordsID[2], psiID, volID,divBID, defID, errID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim2d, &psiID);
    err = nc_def_var( ncid, "volume", NC_DOUBLE, 2, dim2d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim2d, &divBID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim2d, &defID);
    err = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim2d, &errID);

    dg::HVec X( g2d_old.size()), Y(X);
    for( unsigned i=0; i<g2d_old.size(); i++)
    {
        X[i] = g2d_old.r()[i];
        Y[i] = g2d_old.z()[i];
    }
    err = nc_put_var_double( ncid, coordsID[0], X.data());
    err = nc_put_var_double( ncid, coordsID[1], Y.data());
    dg::blas1::transfer( u, X);
    err = nc_put_var_double( ncid, psiID, X.data());
    X = dg::evaluate( dg::coo1, g2d_old);
    dg::blas1::axpby( +1., u, 1.,  X);
    std::cout << "X[0] "<<X[0]<<"\n";
    err = nc_put_var_double( ncid, volID, X.data());
    //Y = dg::evaluate( dg::coo2, g2d_old);
    //dg::blas1::axpby( +1., v, 1., Y);
    //std::cout << "Y[0] "<<Y[0]<<"\n";
    //err = nc_put_var_double( ncid, divBID, Y.data());
    dg::blas1::transfer( u_zeta, X);
    err = nc_put_var_double( ncid, defID, X.data());
    dg::blas1::transfer( u_eta, X);
    err = nc_put_var_double( ncid, errID, X.data());
    err = nc_close(ncid);
    */
    return 0;
}
