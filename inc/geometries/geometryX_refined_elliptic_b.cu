#include <iostream>

#include "dg/algorithm.h"
#include "dg/file/file.h"
#include "dg/geometries/geometries.h"

//#include "taylor.h"
#include "testfunctors.h"

//const char* parameters = "geometry_params_Xpoint_taylor.json";
const char* parameters = "geometry_params_Xpoint.json";

int main(int argc, char**argv)
{
    std::cout << "Type n, Nx (fx = 1./4.), Ny (fy = 1./22.)\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;
    std::cout << "Type psi_0 (-15)! \n";
    double psi_0, psi_1;
    std::cin >> psi_0;
    auto js = dg::file::file2Json( argc == 1 ? parameters : argv[1]);
    //dg::geo::taylor::Parameters gp(js);
    //dg::geo::TokamakMagneticField c = dg::geo::createTaylorField(gp);
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
    //gp.display( std::cout);
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
    std::cout << "Psi min "<<c.psip()(gp.R_0, 0)<<"\n";
    double R0 = gp.R_0, Z0 = 0;
    //double R_X = gp.R_0-1.4*gp.triangularity*gp.a;
    //double Z_X = -1.0*gp.elongation*gp.a;
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    /////////////no monitor
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi;
    ////////////const monitor
    dg::geo::CylindricalSymmTensorLvl1 monitor_chi = make_Xconst_monitor( c.get_psip(), R_X, Z_X) ;
    /////////////monitor bumped around X-point
    //double radius;
    //std::cout << "Type radius\n";
    //std::cin >> radius;
    //dg::geo::CylindricalSymmTensorLvl1 monitor_chi = make_Xbump_monitor( c.get_psip(), R_X, Z_X, radius, radius) ;
    /////////////
    dg::geo::SeparatrixOrthogonal generator(c.get_psip(), monitor_chi, psi_0, R_X,Z_X, R0, Z0,0);
    //dg::geo::SimpleOrthogonalX generator(c.get_psip(), psi_0, R_X,Z_X, R0, Z0,0);
    //dg::CurvilinearGridX2d g2d( generator, 0.25, 1./22., n, Nx, Ny, dg::DIR, dg::NEU);
    dg::EquidistXRefinement equi(add_x, add_y, howmanyX, howmanyY);
    dg::geo::CurvilinearGridX2d g2d_coarse( generator, 0.25, 1./22., n, Nx, Ny, dg::DIR, dg::DIR);
    dg::geo::CurvilinearRefinedGridX2d g2d_fine( equi, generator, 0.25, 1./22., n, Nx, Ny, dg::DIR, dg::DIR);
    dg::Elliptic<dg::aGeometryX2d, dg::Composite<dg::DMatrix>, dg::DVec> pol( g2d_fine, dg::forward);
    //dg::RefinedElliptic<dg::aGeometryX2d, dg::IDMatrix, dg::Composite<dg::DMatrix>, dg::DVec> pol_refined( g2d_coarse, g2d_fine, dg::forward);
    double fx = 0.25;
    psi_1 = -fx/(1.-fx)*psi_0;
    std::cout << "psi 1 is          "<<psi_1<<"\n";

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    std::cout << "Computing on "<<n<<" x "<<Nx<<" x "<<Ny<<" + "<<add_x<<" x "<<add_y<<" x "<<howmanyX<<" x "<<howmanyY<<"\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    dg::file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testXrefined.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    ncerr = dg::file::define_dimensions(  ncid, dim2d, g2d_fine.grid());
    int coordsID[2], psiID, functionID, function2ID;
    ncerr = nc_def_var( ncid, "xc", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    ncerr = nc_def_var( ncid, "yc", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    ncerr = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim2d, &psiID);
    ncerr = nc_def_var( ncid, "num_solution", NC_DOUBLE, 2, dim2d, &functionID);
    ncerr = nc_def_var( ncid, "ana_solution", NC_DOUBLE, 2, dim2d, &function2ID);

    dg::HVec X( g2d_fine.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d_fine.size(); i++)
    {
        X[i] = g2d_fine.map()[0][i];
        Y[i] = g2d_fine.map()[1][i];
    }
    ncerr = nc_put_var_double( ncid, coordsID[0], X.data());
    ncerr = nc_put_var_double( ncid, coordsID[1], Y.data());
    //////////////////blob solution////////////////////////////////////////////
    //const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 480, -300, 70.,1.), g2d_coarse);
    //const dg::DVec bFINE =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 480, -300, 70.,1.), g2d_fine);
    //const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d_coarse);
    //const dg::DVec chiFINE  =  dg::pullback( dg::ONE(), g2d_fine);
    //const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 480, -300, 70., 1. ), g2d_coarse);
    //const dg::DVec solutionFINE =     dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 480, -300, 70., 1. ), g2d_fine);
    /////////////////blob on X-point///////////////////////////////////////////
    //const dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 420, -470, 50.,1.), g2d_coarse);
    //const dg::DVec bFINE =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 420, -470, 50.,1.), g2d_fine);
    //const dg::DVec chi  =  dg::pullback( dg::ONE(), g2d_coarse);
    //const dg::DVec chiFINE  =  dg::pullback( dg::ONE(), g2d_fine);
    //const dg::DVec solution =     dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 420, -470, 50., 1. ), g2d_coarse);
    //const dg::DVec solutionFINE =     dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 420, -470, 50., 1. ), g2d_fine);
    /////////////////Two blobs solution////////////////////////////////////////
    const dg::DVec b1 =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 420, -470, 50.,1.), g2d_coarse);
    dg::DVec b =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 480, -300, 70.,1.), g2d_coarse);
    dg::blas1::axpby( 1., b1, 1., b);
    //
    const dg::DVec b1fine =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 420, -470, 50.,1.), g2d_fine);
    dg::DVec bFINE =        dg::pullback( dg::geo::EllipticBlobDirNeuM(c,psi_0, psi_1, 480, -300, 70.,1.), g2d_fine);
    dg::blas1::axpby( 1., b1fine, 1., bFINE);
    //
    const dg::DVec chi =      dg::evaluate( dg::one, g2d_coarse);
    const dg::DVec chiFINE =  dg::evaluate( dg::one, g2d_fine);
    //
    const dg::DVec solution1 = dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 420, -470, 50., 1. ), g2d_coarse);
    dg::DVec solution = dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 480, -300, 70., 1. ), g2d_coarse);
    dg::blas1::axpby( 1., solution1, 1., solution);
    //
    const dg::DVec solution1fine = dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 420, -470, 50., 1. ), g2d_fine);
    dg::DVec solutionFINE = dg::pullback( dg::geo::FuncDirNeu(c, psi_0, psi_1, 480, -300, 70., 1. ), g2d_fine);
    dg::blas1::axpby( 1., solution1fine, 1., solutionFINE);
    ///////////////////////////Dir/////FIELALIGNED SIN///////////////////
    //const dg::DVec b     =    dg::pullback( dg::geo::EllipticXDirNeuM(c, psi_0, psi_1), g2d_coarse);
    //const dg::DVec bFINE =    dg::pullback( dg::geo::EllipticXDirNeuM(c, psi_0, psi_1), g2d_fine);
    //dg::DVec chi  =  dg::pullback( dg::geo::Bmodule(c), g2d_coarse);
    //dg::DVec chiFINE  =  dg::pullback( dg::geo::Bmodule(c), g2d_fine);
    //dg::blas1::plus( chi, 1e5);
    //dg::blas1::plus( chiFINE, 1e5);
    //const dg::DVec chi      =  dg::pullback( dg::ONE(), g2d_coarse);
    //const dg::DVec chiFINE  =  dg::pullback( dg::ONE(), g2d_fine);
    //const dg::DVec solution     = dg::pullback( dg::geo::FuncXDirNeu(c, psi_0, psi_1 ), g2d_coarse);
    //const dg::DVec solutionFINE = dg::pullback( dg::geo::FuncXDirNeu(c, psi_0, psi_1 ), g2d_fine);
    //////////////////////////////////////////////////////////////////////////

    const dg::DVec vol3d     = dg::create::volume( g2d_coarse);
    const dg::DVec vol3dFINE = dg::create::volume( g2d_fine);
    const dg::DVec w3d       = dg::create::weights( g2d_coarse);
    const dg::DVec v3d       = dg::create::inv_weights( g2d_coarse);
    const dg::DVec v3dFINE   = dg::create::inv_weights( g2d_fine);
    //const dg::IDMatrix Q     = dg::create::interpolation( g2d_fine, g2d_coarse);
    //const dg::IDMatrix P     = dg::create::projection( g2d_coarse, g2d_fine);
    //dg::DVec chi_fine = dg::evaluate( dg::zero, g2d_fine), b_fine(chi_fine);
    //dg::blas2::gemv( Q, chi, chi_fine);
    //dg::blas2::gemv( Q, b, b_fine);
    //pol.set_chi( chi);
    pol.set_chi( chiFINE);
    //pol_refined.set_chi( chi);
    //pol_refined.set_chi( chiFINE);
    //compute error
    const double eps = 1e-11;
    std::cout << "eps \t# direct error_direct \thx_max\thy_max\ttime/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    dg::DVec x_sandwich    =    dg::evaluate( dg::zero, g2d_coarse);
    dg::DVec x_fine_sw     =    dg::evaluate( dg::zero, g2d_fine);
    dg::DVec x_direct      =    dg::evaluate( dg::zero, g2d_coarse);
    dg::DVec x_fine_di     =    dg::evaluate( dg::zero, g2d_fine);
    //dg::PCG<dg::DVec > invert1( x_sandwich, n*n*Nx*Ny);
    dg::PCG<dg::DVec > invert2( x_fine_di,  n*n*Nx*Ny);
    unsigned number_di = invert2.solve(pol, x_fine_di, bFINE, pol.precond(), vol3dFINE, eps);
    //unsigned number_di = invert2(pol        , x_direct, bmod, w3d,v3d);
    //dg::blas2::gemv( Q, x_sandwich, x_fine_sw);
    //dg::blas2::gemv( Q, x_direct,   x_fine_di);
    //dg::PGC<dg::DVec > invert( x_fine, x_fine.size());
    //unsigned number = invert.solve(pol, x_fine ,b_fine, 1., vol3dFINE, eps );
    //unsigned number = invert.solve(pol, x_fine ,bFINE, 1., vol3dFINE, eps );
    //dg::blas2::gemv( P, x_fine, x);
    //std::cout <<0<<"\t";
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
    //std::cout << sqrt( err/norm) << "\t";//<<sqrt( errFINE/normFINE)<<"\t";
    err = dg::blas2::dot( vol3dFINE, error_direct);
    std::cout << sqrt( err/norm) << "\t";//<<sqrt( errFINE/normFINE)<<"\t";
    ///////////////////////////////////metric//////////////////////
    dg::SparseTensor<dg::DVec> metric = g2d_fine.metric();
    dg::DVec gyy = metric.value(1,1), gxx = metric.value(0,0), vol = dg::tensor::volume(metric);
    dg::blas1::transform( gxx, gxx, dg::SQRT<double>());
    dg::blas1::transform( gyy, gyy, dg::SQRT<double>());
    dg::blas1::pointwiseDot( gxx, vol, gxx);
    dg::blas1::pointwiseDot( gyy, vol, gyy);
    dg::blas1::scal( gxx, g2d_fine.hx());
    dg::blas1::scal( gyy, g2d_fine.hy());
    double hxX = dg::interpolate( dg::xspace, (dg::HVec)gxx, 0., 0., g2d_fine);
    double hyX = dg::interpolate( dg::xspace, (dg::HVec)gyy, 0., 0., g2d_fine);
    std::cout << *thrust::max_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::max_element( gyy.begin(), gyy.end()) << "\t";
    std::cout << hxX << "\t";
    std::cout << hyX << "\t";
    std::cout<<t.diff()/(double)number_di<<"s"<<std::endl;

    dg::assign( error_direct, X);
    ncerr = nc_put_var_double( ncid, psiID, X.data());
    dg::assign( x_fine_di, X);
    ncerr = nc_put_var_double( ncid, functionID, X.data());
    dg::assign( solutionFINE, Y);
    //dg::blas1::axpby( 1., X., -1, Y);
    ncerr = nc_put_var_double( ncid, function2ID, Y.data());
    ncerr = nc_close( ncid);


    return 0;
}
