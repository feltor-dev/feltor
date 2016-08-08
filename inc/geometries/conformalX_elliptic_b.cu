#include <iostream>

#include "file/read_input.h"
#include "file/nc_utilities.h"

#include "dg/geometry/refined_grid.h"
#include "dg/geometry/refined_gridX.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/backend/gridX.h"
#include "dg/backend/derivativesX.h"
#include "dg/backend/evaluationX.cuh"
#include "dg/refined_elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
#include "conformal.h"
#include "conformalX.h"
#include "orthogonal.h"
#include "refined_orthogonalX.h"



int main(int argc, char**argv)
{
    std::cout << "Type n, Nx (fx = 1./4.), Ny (fy = 1./22.), Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::cout << "Type psi_0! \n";
    double psi_0, psi_1;
    std::cin >> psi_0;
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

    std::cout << "Type add_x and add_y  and howmany_x and howmany_y\n";
    double add_x, add_y;
    std::cin >> add_x >> add_y;
    double howmanyX, howmanyY;
    std::cin >> howmanyX >> howmanyY;
    orthogonal::refined::GridX3d<dg::DVec> g3d(add_x, add_y, howmanyX, howmanyY, gp, psi_0, 0.25, 1./22.,  n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    orthogonal::refined::GridX2d<dg::DVec> g2d = g3d.perp_grid();
    dg::Elliptic<orthogonal::refined::GridX3d<dg::DVec>, dg::Composite<dg::DMatrix>, dg::DVec> pol( g3d, dg::not_normed, dg::centered);
    //dg::RefinedElliptic<orthogonal::refined::GridX3d<dg::DVec>, dg::IDMatrix, dg::Composite<dg::DMatrix>, dg::DVec> pol( g3d, dg::not_normed, dg::centered);
    psi_1 = g3d.psi1();
    std::cout << "psi 1 is          "<<psi_1<<"\n";

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testE.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    ncerr = file::define_dimensions(  ncid, dim2d, g2d.associated().grid());
    int coordsID[2], psiID, functionID, function2ID;
    ncerr = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    ncerr = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    ncerr = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim2d, &psiID);
    ncerr = nc_def_var( ncid, "num_solution", NC_DOUBLE, 2, dim2d, &functionID);
    ncerr = nc_def_var( ncid, "ana_solution", NC_DOUBLE, 2, dim2d, &function2ID);

    dg::HVec X( g2d.associated().size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.associated().size(); i++)
    {
        X[i] = g2d.associated().r()[i];
        Y[i] = g2d.associated().z()[i];
    }
    ncerr = nc_put_var_double( ncid, coordsID[0], X.data());
    ncerr = nc_put_var_double( ncid, coordsID[1], Y.data());
    ///////////////////////////////////////////////////////////////////////////
    dg::DVec x =    dg::pullback( dg::zero, g3d.associated());
    dg::DVec x_fine =    dg::pullback( dg::zero, g3d);
    const dg::DVec b =    dg::pullback( solovev::EllipticDirNeuM(gp, psi_0, psi_1), g3d.associated());
    dg::DVec bmod(b);
    const dg::DVec chi =  dg::pullback( solovev::BmodTheta(gp), g3d.associated());
    const dg::DVec solution = dg::pullback( solovev::FuncDirNeu(gp, psi_0, psi_1 ), g3d.associated());
    const dg::DVec vol3d = dg::create::volume( g3d);
    dg::HVec inv_vol3d = dg::create::inv_weights( g3d);
    //dg::blas1::pointwiseDivide( inv_vol3d, g3d.weightsX(), inv_vol3d);
    //dg::blas1::pointwiseDivide( inv_vol3d, g3d.weightsY(), inv_vol3d);
    //dg::HVec inv_vol3d = dg::create::inv_volume( g3d);
    const dg::DVec v3d( inv_vol3d);
    const dg::IDMatrix Q = dg::create::interpolation( g3d);
    const dg::IDMatrix P = dg::create::projection( g3d);
    dg::DVec chi_fine = dg::pullback( dg::zero, g3d), b_fine(chi_fine);
    dg::blas2::gemv( Q, chi, chi_fine);
    dg::blas2::gemv( Q, b, b_fine);
    //pol.set_chi( chi);
    pol.set_chi( chi_fine);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-10;
    //dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny*Nz, eps);
    dg::Invert<dg::DVec > invert( x_fine, n*n*Nx*Ny*Nz, eps);
    std::cout << "eps \t # iterations \t error \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    //pol.compute_rhs( b, bmod);
    //unsigned number = invert(pol, x,bmod);// vol3d, v3d );
    unsigned number = invert(pol, x_fine ,b_fine, vol3d, v3d );
    dg::blas2::gemv( P, x_fine, x);
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol3d, error);
    const double norm = dg::blas2::dot( vol3d, solution);
    std::cout << sqrt( err/norm) << "\t";
    dg::HVec gyy, gxx, vol; 
    dg::blas1::transfer( g2d.g_xx(), gyy);
    dg::blas1::transfer( g2d.g_yy(), gxx); 
    dg::blas1::transfer( g2d.vol() , vol);
    dg::blas1::transform( gxx, gxx, dg::SQRT<double>());
    dg::blas1::transform( gyy, gyy, dg::SQRT<double>());
    dg::blas1::pointwiseDot( gxx, vol, gxx);
    dg::blas1::pointwiseDot( gyy, vol, gyy);
    dg::blas1::scal( gxx, g2d.hx());
    dg::blas1::scal( gyy, g2d.hy());
    double hxX = dg::interpolate( 0,  0, gxx, g2d);
    double hyX = dg::interpolate( 0,  0, gyy, g2d);
    std::cout << *thrust::max_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::max_element( gyy.begin(), gyy.end()) << "\t";
    std::cout << hxX << "\t";
    std::cout << hyX << "\t";
    std::cout<<t.diff()/(double)number<<"s"<<std::endl;

    dg::blas1::transfer( error, X);
    ncerr = nc_put_var_double( ncid, psiID, X.data());
    dg::blas1::transfer( x, X);
    ncerr = nc_put_var_double( ncid, functionID, X.data());
    dg::blas1::transfer( solution, X);
    ncerr = nc_put_var_double( ncid, function2ID, X.data());
    ncerr = nc_close( ncid);


    return 0;
}
