#include <iostream>

#include "file/nc_utilities.h"

#include "dg/geometry/refined_grid.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "dg/refined_elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
//#include "guenther.h"
#include "conformal.h"
#include "orthogonal.h"
#include "refined_curvilinear.h"
#include "refined_orthogonal.h"
#include "simple_orthogonal.h"
#include "testfunctors.h"


using namespace dg::geo::solovev;

int main(int argc, char**argv)
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    std::cout << "Type new_n, multiple_x and multiple_y \n";
    double n_ref, multiple_x, multiple_y;
    std::cin >> n_ref>>multiple_x >> multiple_y;
    Json::Reader reader;
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint.js");
        reader.parse(is,js,false);
    }
    else
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    //write parameters from file into variables
    GeomParameters gp(js);
    MagneticField c(gp);
    gp.display( std::cout);
    dg::Timer t;
    Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Constructing grid ... \n";
    t.tic();
//     ConformalGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
//     ConformalGrid2d<dg::DVec> g2d = g3d.perp_grid();
//     dg::Elliptic<ConformalGrid3d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g3d, dg::not_normed, dg::centered);
    dg::geo::SimpleOrthogonal<Psip, PsipR, PsipZ, LaplacePsip> 
        generator( c.psip, c.psipR, c.psipZ, c.laplacePsip, psi_0, psi_1, gp.R_0, 0., 1);
    dg::OrthogonalRefinedGrid3d<dg::DVec> g3d(multiple_x, multiple_y, generator, n_ref, n, Nx, Ny,Nz, dg::DIR);
    dg::OrthogonalRefinedGrid2d<dg::DVec> g2d = g3d.perp_grid();
    dg::Elliptic<dg::OrthogonalRefinedGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
    dg::RefinedElliptic<dg::OrthogonalRefinedGrid2d<dg::DVec>, dg::IDMatrix, dg::DMatrix, dg::DVec> pol_refined( g2d, dg::not_normed, dg::forward);
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testE.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    ncerr = file::define_dimensions(  ncid, dim2d, g2d.associated());
    int coordsID[2], psiID, functionID, function2ID;
    ncerr = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    ncerr = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    ncerr = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim2d, &psiID);
    ncerr = nc_def_var( ncid, "num_solution", NC_DOUBLE, 2, dim2d, &functionID);
    ncerr = nc_def_var( ncid, "ana_solution", NC_DOUBLE, 2, dim2d, &function2ID);

    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.associated().r()[i];
        Y[i] = g2d.associated().z()[i];
    }
    ncerr = nc_put_var_double( ncid, coordsID[0], X.data());
    ncerr = nc_put_var_double( ncid, coordsID[1], Y.data());
    ///////////////////////////////////////////////////////////////////////////
    dg::DVec x =    dg::evaluate( dg::zero, g2d.associated());
    dg::DVec x_fine =    dg::evaluate( dg::zero, g2d);
    //const dg::DVec b =    dg::pullback( dg::geo::EllipticDirNeuM<MagneticField>(c, gp.R_0, psi_0, psi_1, 440, -220, 40., 1), g2d.associated());
    //const dg::DVec chi =  dg::pullback( dg::geo::BmodTheta<MagneticField>(c, gp.R_0), g2d.associated());
    //const dg::DVec solution = dg::pullback( dg::geo::FuncDirNeu<MagneticField>(c,psi_0, psi_1, 440, -220, 40.,1 ), g2d.associated());
    const dg::DVec b =    dg::pullback( dg::geo::EllipticDirPerM<MagneticField>(c, gp.R_0, psi_0, psi_1, 4), g2d.associated());
    const dg::DVec chi =  dg::pullback( dg::geo::Bmodule<MagneticField>(c, gp.R_0), g2d.associated());
    const dg::DVec solution = dg::pullback( dg::geo::FuncDirPer<MagneticField>(c, gp.R_0, psi_0, psi_1, 4), g2d.associated());
    //const dg::DVec b =        dg::pullback( dg::geo::LaplacePsi(gp), g2d.associated());
    //const dg::DVec bFINE =    dg::pullback( dg::geo::LaplacePsi(gp), g2d);
    //const dg::DVec chi =      dg::pullback( dg::one, g2d.associated());
    //const dg::DVec chiFINE =  dg::pullback( dg::one, g2d);
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
    pol.set_chi( chi_fine);
    pol_refined.set_chi( chi_fine);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-10;
    std::cout << "eps \t # iterations \t error \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny*Nz, eps);
    dg::DVec bmod(b);
    pol_refined.compute_rhs( b_fine, bmod);
    unsigned number = invert(pol_refined, x,bmod);// vol3d, v3d );
    //dg::Invert<dg::DVec > invert( x_fine, x_fine.size(), eps);
    //unsigned number = invert(pol, x_fine ,b_fine, vol3dFINE, v3dFINE );
    //dg::blas2::gemv( P, x_fine, x);
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
    dg::blas1::transfer( solution, Y );
    //dg::blas1::axpby( 1., X., -1, Y);
    ncerr = nc_put_var_double( ncid, function2ID, Y.data());
    ncerr = nc_close( ncid);


    return 0;
}
