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
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";

    double eps = 1e10, eps_old = 2e10;
    orthogonal::RingGrid2d<dg::DVec> g2d_old(gp, psi_0, psi_1, n, Nx, Ny, dg::DIR);
    dg::Elliptic<orthogonal::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> ellipticD_old( g2d_old, dg::DIR, dg::PER, dg::not_normed, dg::centered);

    dg::DVec u_old = dg::evaluate( dg::zero, g2d_old);
    dg::DVec lapu = g2d_old.lapx();
    dg::Invert<dg::DVec > invert_old( u_old, n*n*Nx*Ny, 1e-10);
    unsigned number = invert_old( ellipticD_old, u_old, lapu);
    while( (eps < eps_old||eps > 1e-7) && eps > eps_uv)
    {
        eps = eps_old;
        Nx*=2, Ny*=2;
        orthogonal::RingGrid2d<dg::DVec> g2d(gp, psi_0, psi_1, n, Nx, Ny,dg::DIR);
        dg::Elliptic<orthogonal::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> ellipticD( g2d, dg::DIR, dg::PER, dg::not_normed, dg::centered);
        lapu = g2d.lapx();
        const dg::DVec vol2d = dg::create::weights( g2d);
        const dg::IDMatrix Q = dg::create::interpolation( g2d, g2d_old);
        dg::DVec u = dg::evaluate( dg::zero, g2d), u_diff( u);
        dg::blas2::gemv( Q, u_old, u_diff);
        dg::Invert<dg::DVec > invertU( u_diff, n*n*Nx*Ny, 1e-10);
        number = invertU( ellipticD, u, lapu);
        dg::blas1::axpby( 1. ,u, -1., u_diff);
        eps = sqrt( dg::blas2::dot( u_diff, vol2d, u_diff) / dg::blas2::dot( u, vol2d, u) );
        std::cout << "Nx "<<Nx<<" Ny "<<Ny<<" error uv "<<eps<<"\n";
        g2d_old = g2d;
        u_old = u;
    }
    dg::HVec u,v;
    dg::blas1::transfer( u_old, u); 
    //dg::blas1::transfer( v_old, v); 
    //dg::HMatrix dxDIRNEU = dg::create::dx( g2d_old, dg::DIR_NEU);
    dg::HMatrix dxDIR = dg::create::dx( g2d_old, dg::DIR);
    dg::HMatrix dyPER = dg::create::dy( g2d_old, dg::PER);
    dg::HVec u_zeta(u), u_eta(u), v_zeta(v), v_eta(v);
    //dg::HVec u_x(u), u_y(u), v_x(v), v_y(v);
    dg::blas2::symv( dxDIR, u, u_zeta);
    dg::blas1::plus( u_zeta, +1.);
    dg::blas2::symv( dyPER, u, u_eta);
    //dg::blas2::symv( dxDIRNEU, v, v_zeta);
    //dg::blas2::symv( dyPER, v, v_eta);
    //dg::blas1::plus( v_eta, +1.);
    //dg::blas1::pointwiseDot( u_zeta, g2d_old.xr(), u_x); 
    //dg::blas1::pointwiseDot( 1., u_eta, g2d_old.yr(), 1., u_x); 
    //dg::blas1::pointwiseDot( u_zeta, g2d_old.xz(), u_y); 
    //dg::blas1::pointwiseDot( 1., u_eta, g2d_old.yz(), 1., u_y); 
    //dg::blas1::pointwiseDot( v_zeta, g2d_old.xr(), v_x); 
    //dg::blas1::pointwiseDot( 1., v_eta, g2d_old.yr(), 1., v_x); 
    //dg::blas1::pointwiseDot( v_zeta, g2d_old.xz(), v_y); 
    //dg::blas1::pointwiseDot( 1., v_eta, g2d_old.yz(), 1., v_y); 

    //dg::HVec xdiff(u), ydiff(u);
    //dg::blas1::axpby( 1., u_x, -1., v_y, xdiff);
    //dg::blas1::axpby( 1., u_y, +1., v_x, ydiff);
    //const dg::HVec vol2d = dg::create::weights( g2d_old);
    //std::cout << "rel conformity error X "<<sqrt(dg::blas2::dot( xdiff, vol2d, xdiff)/dg::blas2::dot( u_x, vol2d, u_x)) <<"\n";
    //std::cout << "rel conformity error Y "<<sqrt(dg::blas2::dot( ydiff, vol2d, ydiff)/dg::blas2::dot( u_y, vol2d, u_y)) <<"\n";

    ///////////////////////////////FILE OUTPUT/////////////////////////////
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
    return 0;
}
