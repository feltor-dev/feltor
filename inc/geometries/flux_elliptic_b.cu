#include <iostream>

#include "file/read_input.h"
#include "file/nc_utilities.h"

#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
#include "flux.h"

int main(int argc, char**argv)
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params_Xpoint.txt"); 
        }
        else
        {
            v = file::read_input( argv[1]); 
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    solovev::GeomParameters gp(v);
    gp.display( std::cout);
    dg::Timer t;
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Constructing grid ... \n";
    t.tic();
    flux::RingGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    flux::RingGrid2d<dg::DVec> g2d = g3d.perp_grid();
    dg::Elliptic<flux::RingGrid3d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g3d, dg::not_normed, dg::centered);

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testE.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    ncerr = file::define_dimensions(  ncid, dim2d, g2d);
    int coordsID[2], psiID, functionID, function2ID;
    ncerr = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    ncerr = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    ncerr = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim2d, &psiID);
    ncerr = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim2d, &functionID);
    ncerr = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim2d, &function2ID);

    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.r()[i];
        Y[i] = g2d.z()[i];
    }
    ncerr = nc_put_var_double( ncid, coordsID[0], X.data());
    ncerr = nc_put_var_double( ncid, coordsID[1], Y.data());
    ///////////////////////////////////////////////////////////////////////////
    dg::DVec x =    dg::pullback( dg::zero, g3d);
    const dg::DVec b =    dg::pullback( solovev::EllipticDirPerM(gp, psi_0, psi_1), g3d);
    const dg::DVec chi =  dg::pullback( solovev::Bmodule(gp), g3d);
    const dg::DVec solution = dg::pullback( solovev::FuncDirPer(gp, psi_0, psi_1 ), g3d);
    const dg::DVec vol3d = dg::create::volume( g3d);
    pol.set_chi( chi);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-10;
    dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny*Nz, eps);
    std::cout << "eps \t # iterations \t error \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    unsigned number = invert(pol, x,b);// vol3d, v3d );
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol3d, error);
    const double norm = dg::blas2::dot( vol3d, solution);
    std::cout << sqrt( err/norm) << "\t";
    dg::DVec gyy = g2d.g_xx(), gxx=g2d.g_yy(), vol = g2d.vol();
    dg::blas1::transform( gxx, gxx, dg::SQRT<double>());
    dg::blas1::transform( gyy, gyy, dg::SQRT<double>());
    dg::blas1::pointwiseDot( gxx, vol, gxx);
    dg::blas1::pointwiseDot( gyy, vol, gyy);
    dg::blas1::scal( gxx, g2d.hx());
    dg::blas1::scal( gyy, g2d.hy());
    std::cout << *thrust::max_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::max_element( gyy.begin(), gyy.end()) << "\t";
    std::cout<<t.diff()/(double)number<<"s"<<std::endl;

    dg::blas1::transfer( error, X );
    ncerr = nc_put_var_double( ncid, psiID, X.data());
    dg::blas1::transfer( x, X );
    ncerr = nc_put_var_double( ncid, functionID, X.data());
    dg::blas1::transfer( solution, X );
    ncerr = nc_put_var_double( ncid, function2ID, X.data());
    ncerr = nc_close( ncid);


    return 0;
}
