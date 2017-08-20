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
#include "taylor.h"
//#include "ribeiroX.h"
#include "refined_orthogonalX.h"
#include "separatrix_orthogonal.h"
#include "testfunctors.h"

using namespace dg::geo::taylor;
const char* parameters = "geometry_params_Xpoint_taylor.js";
//using namespace dg::geo::solovev;
//const char* parameters = "geometry_params_Xpoint.js";

int main(int argc, char**argv)
{
    std::cout << "Type n, Nx (fx = 1./4.), Ny (fy = 1./22.)\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
    std::cout << "Type psi_0 (-100)! \n";
    double psi_0, psi_1;
    std::cin >> psi_0;
    std::cout << "Type n_ref! \n";
    unsigned n_ref; 
    std::cin >> n_ref;
    Json::Reader reader;
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint_taylor.js");
        reader.parse(is,js,false);
    }
    else
    {
        std::ifstream is(argv[1]);
        reader.parse(is,js,false);
    }
    GeomParameters gp(js);
    gp.display( std::cout);
    dg::Timer t;
    std::cout << "Constructing grid ... \n";
    t.tic();

    //std::cout << "Type muliple_x and multiple_y \n";
    std::cout << "Type add_x and add_y  and howmany_x and howmany_y\n";
    double add_x, add_y;
    std::cin >> add_x >> add_y;
    double howmanyX, howmanyY;
    std::cin >> howmanyX >> howmanyY;

    ////////////////construct Generator////////////////////////////////////
    dg::geo::TokamakMagneticField c = dg::geo::taylor::createMagField(gp);
    std::cout << "Psi min "<<c.psip()(gp.R_0, 0)<<"\n";
    double R0 = gp.R_0, Z0 = 0;
    //double R_X = gp.R_0-1.4*gp.triangularity*gp.a;
    //double Z_X = -1.0*gp.elongation*gp.a;
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    dg::geo::SeparatrixOrthogonal generator(c.get_psip(), psi_0, R_X,Z_X, R0, Z0,0);
    //dg::geo::SimpleOrthogonalX generator(c.get_psip(), psi_0, R_X,Z_X, R0, Z0,0);
    //dg::CurvilinearGridX2d g2d( generator, 0.25, 1./22., n, Nx, Ny, dg::DIR, dg::NEU);
    dg::EquidistRefinementX equi(add_x, add_y, howmanyX, howmanyY)
    dg::OrthogonalRefinedGridX2d g2d( equi, generator, 0.25, 1./22., n_ref, n, Nx, Ny, dg::DIR, dg::NEU);
    dg::Elliptic<dg::OrthogonalRefinedGridX2d, dg::Composite<dg::DMatrix>, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
    dg::RefinedElliptic<dg::OrthogonalRefinedGridX2d, dg::IDMatrix, dg::Composite<dg::DMatrix>, dg::DVec> pol_refined( g2d, dg::not_normed, dg::forward);
    double fx = 0.25;
    psi_1 = -fx/(1.-fx)*psi_0;
    std::cout << "psi 1 is          "<<psi_1<<"\n";

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    std::cout << "Computing on "<<n<<" x "<<Nx<<" x "<<Ny<<" + "<<add_x<<" + "<<add_y<<" x "<<howmanyX<<" x "<<howmanyY<<"\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testX.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    ncerr = file::define_dimensions(  ncid, dim2d, g2d.grid());
    int coordsID[2], psiID, functionID, function2ID;
    ncerr = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    ncerr = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    ncerr = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim2d, &psiID);
    ncerr = nc_def_var( ncid, "num_solution", NC_DOUBLE, 2, dim2d, &functionID);
    ncerr = nc_def_var( ncid, "ana_solution", NC_DOUBLE, 2, dim2d, &function2ID);

    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.map()[0][i];
        Y[i] = g2d.map()[1][i];
    }
    ncerr = nc_put_var_double( ncid, coordsID[0], X.data());
    ncerr = nc_put_var_double( ncid, coordsID[1], Y.data());
    //////////////////blob solution////////////////////////////////////////////
    //const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM<MagneticField>(c,psi_0, psi_1, 450, -340, 40.,1.), g2d.associated());
    //const dg::DVec bFINE =        dg::pullback( dg::geo::EllipticBlobDirNeuM<MagneticField>(c,psi_0, psi_1, 450, -340, 40.,1.), g2d);
    //const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d.associated());
    //const dg::DVec chiFINE  =  dg::pullback( dg::ONE(), g2d);
    //const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu<MagneticField>(c, psi_0, psi_1, 450, -340, 40., 1. ), g2d.associated());
    //const dg::DVec solutionFINE =     dg::pullback( dg::geo::FuncDirNeu<MagneticField>(c, psi_0, psi_1, 450, -340, 40., 1. ), g2d);
    /////////////////blob on X-point///////////////////////////////////////////
    const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM<MagneticField>(c,psi_0, psi_1, 480, -420, 40.,1.), g2d.associated());
    const dg::DVec bFINE =        dg::pullback( dg::geo::EllipticBlobDirNeuM<MagneticField>(c,psi_0, psi_1, 480, -420, 40.,1.), g2d);
    const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d.associated());
    const dg::DVec chiFINE  =  dg::pullback( dg::ONE(), g2d);
    const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu<MagneticField>(c, psi_0, psi_1, 480, -420, 40., 1. ), g2d.associated());
    const dg::DVec solutionFINE =     dg::pullback( dg::geo::FuncDirNeu<MagneticField>(c, psi_0, psi_1, 480, -420, 40., 1. ), g2d);
    ///////////////////////////////////////////////////////////////////////////
    //const dg::DVec b =        dg::pullback( c.laplacePsip, g2d.associated());
    //const dg::DVec bFINE =    dg::pullback( c.laplacePsip, g2d);
    //const dg::DVec chi =      dg::evaluate( dg::one, g2d.associated());
    //const dg::DVec chiFINE =  dg::evaluate( dg::one, g2d);
    //const dg::DVec solution =     dg::pullback( c.psip, g2d.associated());
    //const dg::DVec solutionFINE = dg::pullback( c.psip, g2d);
    ///////////////////////////Dir/////FIELALIGNED SIN///////////////////
    //const dg::DVec b     =    dg::pullback( dg::geo::EllipticXDirNeuM<MagneticField>(c, gp.R_0, psi_0, psi_1), g2d.associated());
    //const dg::DVec bFINE =    dg::pullback( dg::geo::EllipticXDirNeuM<MagneticField>(c, gp.R_0, psi_0, psi_1), g2d);
    //dg::DVec chi  =  dg::pullback( dg::geo::Bmodule<MagneticField>(c, gp.R_0), g2d.associated());
    //dg::DVec chiFINE  =  dg::pullback( dg::geo::Bmodule<MagneticField>(c, gp.R_0), g2d);
    //dg::blas1::plus( chi, 1e5);
    //dg::blas1::plus( chiFINE, 1e5);
    ////const dg::DVec chi      =  dg::pullback( dg::ONE(), g2d.associated());
    ////const dg::DVec chiFINE  =  dg::pullback( dg::ONE(), g2d);
    //const dg::DVec solution     = dg::pullback( dg::geo::FuncXDirNeu<MagneticField>(c, psi_0, psi_1 ), g2d.associated());
    //const dg::DVec solutionFINE = dg::pullback( dg::geo::FuncXDirNeu<MagneticField>(c, psi_0, psi_1 ), g2d);
    //////////////////////////////////////////////////////////////////////////

    const dg::DVec vol3d     = dg::create::volume( g2d.associated());
    const dg::DVec vol3dFINE = dg::create::volume( g2d);
    const dg::DVec w3d       = dg::create::weights( g2d.associated());
    const dg::DVec v3d       = dg::create::inv_weights( g2d.associated());
    const dg::DVec v3dFINE   = dg::create::inv_weights( g2d);
    const dg::IDMatrix Q = dg::create::interpolation( g2d);
    const dg::IDMatrix P = dg::create::projection( g2d);
    dg::DVec chi_fine = dg::evaluate( dg::zero, g2d), b_fine(chi_fine);
    dg::blas2::gemv( Q, chi, chi_fine);
    dg::blas2::gemv( Q, b, b_fine);
    //pol.set_chi( chi);
    pol.set_chi( chiFINE);
    //pol_refined.set_chi( chi);
    pol_refined.set_chi( chiFINE);
    //compute error
    const double eps = 1e-11;
    std::cout << "eps \t # sandwich \t # direct \t error_sandwich \t error_direct \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    dg::DVec x_sandwich    =    dg::evaluate( dg::zero, g2d.associated());
    dg::DVec x_fine_sw     =    dg::evaluate( dg::zero, g2d);
    dg::DVec x_direct      =    dg::evaluate( dg::zero, g2d.associated());
    dg::DVec x_fine_di     =    dg::evaluate( dg::zero, g2d);
    dg::Invert<dg::DVec > invert1( x_sandwich, n*n*Nx*Ny, eps);
    dg::Invert<dg::DVec > invert2( x_fine_di,  n*n*Nx*Ny, eps);
    dg::DVec bmod(b);
    pol_refined.compute_rhs( bFINE, bmod);
    unsigned number_sw = invert1(pol_refined, x_sandwich, bmod, w3d, v3d );
    unsigned number_di = invert2(pol        , x_fine_di, bFINE, vol3dFINE,v3dFINE);
    //unsigned number_di = invert2(pol        , x_direct, bmod, w3d,v3d);
    dg::blas2::gemv( Q, x_sandwich, x_fine_sw);
    //dg::blas2::gemv( Q, x_direct,   x_fine_di);
    //dg::Invert<dg::DVec > invert( x_fine, x_fine.size(), eps);
    //unsigned number = invert(pol, x_fine ,b_fine, vol3dFINE, v3dFINE );
    //unsigned number = invert(pol, x_fine ,bFINE, vol3dFINE, v3dFINE );
    //dg::blas2::gemv( P, x_fine, x);
    std::cout <<number_sw<<"\t";
    std::cout <<number_di<<"\t";
    t.toc();
    dg::DVec error_sandwich( solutionFINE);
    dg::DVec error_direct(   solutionFINE);
    dg::blas1::axpby( 1.,x_fine_sw,  -1., solutionFINE, error_sandwich);
    dg::blas1::axpby( 1.,x_fine_di,  -1., solutionFINE, error_direct);
    //double errFINE = dg::blas2::dot( vol3dFINE, errorFINE);
    double err = dg::blas2::dot( vol3dFINE, error_sandwich);
    const double norm = dg::blas2::dot( vol3dFINE, solutionFINE);
    //const double normFINE = dg::blas2::dot( vol3dFINE, solutionFINE);
    std::cout << sqrt( err/norm) << "\t";//<<sqrt( errFINE/normFINE)<<"\t";
    err = dg::blas2::dot( vol3dFINE, error_direct);
    std::cout << sqrt( err/norm) << "\t";//<<sqrt( errFINE/normFINE)<<"\t";
    ///////////////////////////////////metric//////////////////////
    dg::SparseTensor<dg::DVec> metric = g2d.metric();
    dg::DVec gyy = metric.value(1,1), gxx = metric.value(0,0), vol = dg::tensor::volume(metric).value(); 
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

    dg::blas1::transfer( error_direct, X);
    ncerr = nc_put_var_double( ncid, psiID, X.data());
    dg::blas1::transfer( x_fine_di, X);
    ncerr = nc_put_var_double( ncid, functionID, X.data());
    dg::blas1::transfer( solutionFINE, Y);
    //dg::blas1::axpby( 1., X., -1, Y);
    ncerr = nc_put_var_double( ncid, function2ID, Y.data());
    ncerr = nc_close( ncid);


    return 0;
}
