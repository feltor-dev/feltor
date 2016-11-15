#include <iostream>

#include "file/nc_utilities.h"

#include "dg/geometry/refined_gridX.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/backend/gridX.h"
#include "dg/backend/derivativesX.h"
#include "dg/backend/evaluationX.cuh"
#include "dg/refined_elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
//#include "ribeiroX.h"
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
    std::cout << "Type n_ref! \n";
    unsigned n_ref; 
    std::cin >> n_ref;
    Json::Reader reader;
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint_harmonic.js");
        reader.parse(is,js,false);
    }
    else
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    solovev::GeomParameters gp(js);
    gp.display( std::cout);
    dg::Timer t;
    std::cout << "Constructing grid ... \n";
    t.tic();

    std::cout << "Type add_x and add_y  and howmany_x and howmany_y\n";
    //std::cout << "Type muliple_x and multiple_y \n";
    double add_x, add_y;
    std::cin >> add_x >> add_y;
    double howmanyX, howmanyY;
    std::cin >> howmanyX >> howmanyY;

    ////////////////construct Generator////////////////////////////////////
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    solovev::PsipR psipR(gp); solovev::PsipZ psipZ(gp);
    solovev::LaplacePsip laplacePsip(gp); 
    double R0 = gp.R_0, Z0 = 0;
    double R_X = gp.R_0-1.4*gp.triangularity*gp.a;
    double Z_X = -1.0*gp.elongation*gp.a;
    //double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    //double Z_X = -1.1*gp.elongation*gp.a;
    dg::SeparatrixOrthogonal<solovev::Psip,solovev::PsipR,solovev::PsipZ,solovev::LaplacePsip> generator(psip, psipR, psipZ, laplacePsip, psi_0, R_X,Z_X, R0, Z0,1);
    dg::OrthogonalRefinedGridX3d<dg::DVec> g3d(add_x, add_y, howmanyX, howmanyY, generator, psi_0, 0.25, 1./22., n_ref, n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    //OrthogonalRefinedGridX3d<dg::DVec> g3d(add_x, add_y, gp, psi_0, 0.25, 1./22., n_ref, n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    dg::OrthogonalRefinedGridX2d<dg::DVec> g2d = g3d.perp_grid();
    dg::Elliptic<dg::OrthogonalRefinedGridX2d<dg::DVec>, dg::Composite<dg::DMatrix>, dg::DVec> pol( g2d, dg::not_normed, dg::centered);
    dg::RefinedElliptic<dg::OrthogonalRefinedGridX2d<dg::DVec>, dg::IDMatrix, dg::Composite<dg::DMatrix>, dg::DVec> pol_refined( g2d, dg::not_normed, dg::centered);
    double fx = 0.25;
    psi_1 = -fx/(1.-fx)*psi_0;
    std::cout << "psi 1 is          "<<psi_1<<"\n";

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testX.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
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
    dg::DVec x =         dg::evaluate( dg::zero, g2d.associated());
    dg::DVec x_fine =    dg::evaluate( dg::zero, g2d);
    const dg::DVec b =        dg::pullback( solovev::EllipticDirNeuM(gp, psi_0, psi_1,0,0,1,1), g2d.associated());
    const dg::DVec bFINE =    dg::pullback( solovev::EllipticDirNeuM(gp, psi_0, psi_1, 0,0,1,1), g2d);
    dg::DVec bmod(b);
    const dg::DVec chi =      dg::pullback( solovev::BmodTheta(gp), g2d.associated());
    const dg::DVec chiFINE =  dg::pullback( solovev::BmodTheta(gp), g2d);
    const dg::DVec solution =     dg::pullback( solovev::FuncDirNeu(gp, psi_0, psi_1, 0,0,1,1 ), g2d.associated());
    const dg::DVec solutionFINE = dg::pullback( solovev::FuncDirNeu(gp, psi_0, psi_1,0,0,1,1 ), g2d);
    //const dg::DVec b =        dg::pullback( solovev::LaplacePsi(gp), g2d.associated());
    //const dg::DVec bFINE =    dg::pullback( solovev::LaplacePsi(gp), g2d);
    //dg::DVec bmod(b);
    //const dg::DVec chi =      dg::evaluate( dg::one, g2d.associated());
    //const dg::DVec chiFINE =  dg::evaluate( dg::one, g2d);
    //const dg::DVec solution =     dg::pullback( psip, g2d.associated());
    //const dg::DVec solutionFINE = dg::pullback( psip, g2d);

    const dg::DVec vol3dFINE = dg::create::volume( g2d);
    dg::HVec inv_vol3dFINE = dg::create::inv_weights( g2d);
    const dg::DVec vol3d = dg::create::volume( g2d.associated());
    const dg::DVec v3dFINE( inv_vol3dFINE);
    const dg::IDMatrix Q = dg::create::interpolation( g2d);
    const dg::IDMatrix P = dg::create::projection( g2d);
    dg::DVec chi_fine = dg::evaluate( dg::zero, g2d), b_fine(chi_fine);
    dg::blas2::gemv( Q, chi, chi_fine);
    dg::blas2::gemv( Q, b, b_fine);
    //pol.set_chi( chi);
    //pol.set_chi( chi_fine);
    pol.set_chi( chiFINE);
    //pol_refined.set_chi( chi);
    pol_refined.set_chi( chiFINE);
    //compute error
    dg::DVec error( solution);
    dg::DVec errorFINE( solutionFINE);
    const double eps = 1e-11;
    std::cout << "eps \t # iterations \t errorCOARSE \t errorFINE \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny*Nz, eps);
    pol_refined.compute_rhs( bFINE, bmod);
    unsigned number = invert(pol_refined, x,bmod);// vol3d, v3d );
    dg::blas2::gemv( Q, x, x_fine);
    //dg::Invert<dg::DVec > invert( x_fine, x_fine.size(), eps);
    //unsigned number = invert(pol, x_fine ,b_fine, vol3dFINE, v3dFINE );
    //unsigned number = invert(pol, x_fine ,bFINE, vol3dFINE, v3dFINE );
    //dg::blas2::gemv( P, x_fine, x);
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    dg::blas1::axpby( 1.,x_fine,-1., solutionFINE, errorFINE);
    double errFINE = dg::blas2::dot( vol3dFINE, errorFINE);
    double err = dg::blas2::dot( vol3d, error);
    const double norm = dg::blas2::dot( vol3d, solution);
    const double normFINE = dg::blas2::dot( vol3dFINE, solutionFINE);
    std::cout << sqrt( err/norm) << "\t"<<sqrt( errFINE/normFINE)<<"\t";
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
    double hxX = dg::interpolate( 0, 0, gxx, g2d);
    double hyX = dg::interpolate( 0, 0, gyy, g2d);
    std::cout << *thrust::max_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::max_element( gyy.begin(), gyy.end()) << "\t";
    std::cout << hxX << "\t";
    std::cout << hyX << "\t";
    std::cout<<t.diff()/(double)number<<"s"<<std::endl;

    dg::blas1::transfer( error, X);
    ncerr = nc_put_var_double( ncid, psiID, X.data());
    dg::blas1::transfer( x, X);
    ncerr = nc_put_var_double( ncid, functionID, X.data());
    dg::blas1::transfer( solution, Y);
    //dg::blas1::axpby( 1., X., -1, Y);
    ncerr = nc_put_var_double( ncid, function2ID, Y.data());
    ncerr = nc_close( ncid);


    return 0;
}
