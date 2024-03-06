#include <iostream>
#include <memory>

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "flux.h"
#include "solovev.h"
#include "guenter.h"
#include "simple_orthogonal.h"
#include "curvilinear.h"
#include "testfunctors.h"

int main(int argc, char**argv)
{
    std::cout << "Type n (3), Nx (8), Ny (80), Nz (1)\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    auto js = dg::file::file2Json( argc == 1 ? "geometry_params_Xpoint.json" : argv[1]);
    //write parameters from file into variables
    dg::geo::solovev::Parameters gp(js);
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField(gp);
    gp.display( std::cout);
    dg::Timer t;
    std::cout << "Psi min "<<mag.psip()(gp.R_0, 0)<<"\n";
    std::cout << "Constructing grid ... \n";
    t.tic();
    //dg::geo::SimpleOrthogonal generator( mag.get_psip(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::FluxGenerator generator( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::CurvilinearProductGrid3d g3d( generator, n, Nx, Ny,Nz, dg::DIR);
    std::unique_ptr<dg::aGeometry2d> g2d( g3d.perp_grid() );

    dg::Elliptic<dg::aGeometry2d, dg::DMatrix, dg::DVec> pol( *g2d, dg::forward);
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    ///////////////////////////////////////////////////////////////////////////
    int ncid;
    dg::file::NC_Error_Handle ncerr;
    ncerr = nc_create( "testE.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    dg::file::Writer<dg::aGeometry2d> writer( ncid, *g2d, {"y", "x"});
    writer.def_and_put( "xc", {}, g2d->map()[0]);
    writer.def_and_put( "yc", {}, g2d->map()[1]);

    ///////////////////////////////////////////////////////////////////////////
    dg::DVec x = dg::evaluate( dg::zero, *g2d);
    const dg::DVec b =    dg::pullback( dg::geo::EllipticDirPerM(mag, psi_0, psi_1, 4), *g2d);
    const dg::DVec chi =  dg::pullback( dg::geo::Bmodule(mag), *g2d);
    const dg::DVec solution = dg::pullback( dg::geo::FuncDirPer(mag, psi_0, psi_1, 4), *g2d);
    const dg::DVec vol3d = dg::create::volume( *g2d);
    pol.set_chi( chi);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-10;
    dg::PCG<dg::DVec > pcg( x, n*n*Nx*Ny*Nz);
    std::cout << "eps \t # iterations \t error \t hx_max\t hy_max \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    unsigned number = pcg.solve(pol, x,b, pol.precond(), pol.weights(), eps);
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol3d, error);
    const double norm = dg::blas2::dot( vol3d, solution);
    std::cout << sqrt( err/norm) << "\t";

    dg::SparseTensor<dg::DVec> metric = g2d->metric();
    dg::DVec gyy = metric.value(1,1), gxx=metric.value(0,0), volume = dg::tensor::volume(metric);
    dg::blas1::transform( gxx, gxx, dg::SQRT<double>());
    dg::blas1::transform( gyy, gyy, dg::SQRT<double>());
    dg::blas1::pointwiseDot( gxx, volume, gxx);
    dg::blas1::pointwiseDot( gyy, volume, gyy);
    dg::blas1::scal( gxx, g2d->hx());
    dg::blas1::scal( gyy, g2d->hy());
    std::cout << *thrust::max_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::max_element( gyy.begin(), gyy.end()) << "\t";
    std::cout<<t.diff()/(double)number<<"s"<<std::endl;
    ///////////////////////////////////////////////////////////////////////
    std::cout << "TESTING VARIATION\n";
    pol.variation( x, x);
    const dg::DVec variation = dg::pullback( dg::geo::VariationDirPer( mag, psi_0, psi_1), *g2d);
    dg::blas1::axpby( 1., variation, -1., x);
    double result = dg::blas2::dot( x, vol3d, x);
    std::cout << "               distance to solution "<<sqrt( result)<<std::endl; //don't forget sqrt when comuting errors

    writer.def_and_put( "error", {}, (dg::HVec)error);
    writer.def_and_put( "num_solution", {}, (dg::HVec)x);
    writer.def_and_put( "ana_solution", {}, (dg::HVec)solution);
    ncerr = nc_close( ncid);


    return 0;
}
