#include <iostream>

#include "file/nc_utilities.h"

#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/geometry/geometry.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
#include "guenther.h"
#include "simple_orthogonal.h"
#include "curvilinear.h"
#include "testfunctors.h"


int main(int argc, char**argv)
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
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
    dg::geo::solovev::GeomParameters gp(js);
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField(gp);
    gp.display( std::cout);
    dg::Timer t;
    std::cout << "Psi min "<<c.psip()(gp.R_0, 0)<<"\n";
    std::cout << "Constructing grid ... \n";
    t.tic();
    dg::geo::SimpleOrthogonal generator( c.get_psip(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::CurvilinearProductGrid3d g3d( generator, n, Nx, Ny,Nz, dg::DIR);
    dg::CurvilinearGrid2d g2d = g3d.perp_grid();
    dg::Elliptic<dg::CurvilinearGrid2d, dg::DMatrix, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
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
    ///////////////////////////////////////////////////////////////////////////
    dg::DVec x = dg::evaluate( dg::zero, g2d);
    //const dg::DVec b =    dg::pullback( dg::geo::EllipticDirNeuM(c, psi_0, psi_1, 440, -220, 40., 1), g2d);
    //const dg::DVec chi =  dg::pullback( dg::geo::BmodTheta(c), g2d);
    //const dg::DVec solution = dg::pullback( dg::geo::FuncDirNeu(c,psi_0, psi_1, 440, -220, 40.,1 ), g2d);
    const dg::DVec b =    dg::pullback( dg::geo::EllipticDirPerM(c, psi_0, psi_1, 4), g2d);
    const dg::DVec chi =  dg::pullback( dg::geo::Bmodule(c), g2d);
    const dg::DVec solution = dg::pullback( dg::geo::FuncDirPer(c, psi_0, psi_1, 4), g2d);
    //const dg::DVec b =        dg::pullback( dg::geo::LaplacePsi(gp), g2d);
    //const dg::DVec chi =      dg::pullback( dg::one, g2d);
    //const dg::DVec solution =     dg::pullback( psip, g2d);

    const dg::DVec vol3d = dg::create::volume( g2d);
    pol.set_chi( chi);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-10;
    std::cout << "eps \t # iterations \t error \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny*Nz, eps);
    unsigned number = invert(pol, x,b);// vol3d, v3d );
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol3d, error);
    const double norm = dg::blas2::dot( vol3d, solution);
    std::cout << sqrt( err/norm) << "\t";

    dg::SparseTensor<dg::DVec> metric = g2d.metric();
    dg::DVec gyy = metric.value(1,1), gxx=metric.value(0,0), vol = dg::tensor::volume(metric).value();
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
