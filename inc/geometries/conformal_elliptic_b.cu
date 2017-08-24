#include <iostream>

#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

//Grids
#include "curvilinear.h"
//generators
#include "simple_orthogonal.h"
#include "hector.h"
//and some functors
#include "solovev.h"
//#include "guenther.h"
#include "testfunctors.h"
#include "dg/backend/timer.cuh"

using namespace dg::geo::solovev;

const unsigned nIter=6;
template<class Geometry>
void compute_error_elliptic( const dg::geo::TokamakMagneticField& c, const Geometry& g2d, double psi_0, double psi_1, double eps)
{
    dg::DVec x =    dg::evaluate( dg::zero, g2d);
    /////////////////////////////DirNeu/////FLUXALIGNED//////////////////////
    //dg::Elliptic<Geometry, dg::DMatrix, dg::DVec> pol( g2d, dg::DIR_NEU, dg::PER, dg::not_normed, dg::forward);
    //const dg::DVec b =    dg::pullback( dg::geo::EllipticDirPerM(c, psi_0, 2.*psi_1-psi_0, 0), g2d);
    //const dg::DVec chi =  dg::pullback( dg::geo::Bmodule(c), g2d);
    //const dg::DVec solution = dg::pullback( dg::geo::FuncDirPer(c, psi_0, 2.*psi_1-psi_0, 0 ), g2d);
    /////////////////////////////Dir/////FIELALIGNED SIN///////////////////
    //dg::Elliptic<Geometry, dg::DMatrix, dg::DVec> pol( g2d, dg::DIR, dg::PER, dg::not_normed, dg::forward);
    //const dg::DVec b =    dg::pullback( dg::geo::EllipticDirPerM(c, psi_0, psi_1, 4), g2d);
    //const dg::DVec chi =  dg::pullback( dg::geo::Bmodule(c), g2d);
    //const dg::DVec solution = dg::pullback( dg::geo::FuncDirPer(c, psi_0, psi_1, 4 ), g2d);
    /////////////////////////////Dir//////BLOB/////////////////////////////
    dg::Elliptic<Geometry, dg::DMatrix, dg::DVec> pol( g2d, dg::DIR, dg::PER, dg::not_normed, dg::forward);
    const dg::DVec b =    dg::pullback( dg::geo::EllipticDirNeuM(c, psi_0, psi_1, 440, -220, 40.,1.), g2d);
    const dg::DVec chi =  dg::pullback( dg::geo::BmodTheta(c), g2d);
    const dg::DVec solution = dg::pullback(dg::geo::FuncDirNeu(c, psi_0, psi_1, 440, -220, 40.,1.), g2d);
    ///////////////////////////////////////////////////////////////////////
    const dg::DVec vol2d = dg::create::volume( g2d);
    pol.set_chi( chi);
    //compute error
    dg::DVec error( solution);
    std::cout << eps<<"\t"<<g2d.n()<<"\t"<<g2d.Nx()<<"\t"<<g2d.Ny()<<"\t";
    dg::Timer t;
    t.tic();
    dg::Invert<dg::DVec > invert( x, g2d.size(), eps);
    unsigned number = invert(pol, x,b);
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol2d, error);
    const double norm = dg::blas2::dot( vol2d, solution);
    std::cout << sqrt( err/norm) << "\t";
    std::cout<<t.diff()/(double)number<<"\t";
}

template<class Geometry>
void compute_cellsize( const Geometry& g2d)
{
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
    std::cout << *thrust::min_element( gxx.begin(), gxx.end()) << "\t";
    std::cout << *thrust::min_element( gyy.begin(), gyy.end()) << "\t";
}

int main(int argc, char**argv)
{
    std::cout << "Type nHector, NxHector, NyHector (13 2 10)\n";
    unsigned nGrid, NxGrid, NyGrid;
    std::cin >> nGrid>> NxGrid>>NyGrid;   
    std::cout << "Type nInit, NxInit, NyInit (for conformal grid, 3 2 20)\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
    const unsigned NxIni=Nx, NyIni=Ny;
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
    GeomParameters gp(js);
    gp.display( std::cout);
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField( gp); 
    const double eps = 1e-10;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "eps\tn\t Nx\t Ny \t # iterations \t error  \t time/iteration (s)\t hx_max\t hy_max\t hx_min\t hy_min \n";
    std::cout << "Orthogonal:\n";
    dg::geo::SimpleOrthogonal generator0(c.get_psip(), psi_0, psi_1, gp.R_0, 0., 0);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d g2d(generator0, n, Nx, Ny);
        compute_error_elliptic(c, g2d, psi_0, psi_1, eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "Orthogonal Adapted:\n";
    dg::geo::SimpleOrthogonal generator1(c.get_psip(), psi_0, psi_1, gp.R_0, 0., 1);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d g2d(generator1, n, Nx, Ny);
        compute_error_elliptic(c, g2d, psi_0, psi_1, eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "Conformal:\n";
    dg::geo::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hectorConf( c.get_psip(), psi_0, psi_1, gp.R_0, 0., nGrid,NxGrid,NyGrid, 1e-10, true);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d g2d(hectorConf, n, Nx, Ny);
        compute_error_elliptic(c, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "ConformalMonitor:\n";
    dg::geo::BinarySymmTensorLvl1 lc = dg::geo::make_LiseikinCollective( c.get_psip(), 0.1, 0.001);
    dg::geo::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hectorMonitor( c.get_psip(), lc, psi_0, psi_1, gp.R_0, 0., nGrid,NxGrid,NyGrid, 1e-10, true);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d g2d(hectorMonitor, n, Nx, Ny);
        compute_error_elliptic(c, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "ConformalAdaption:\n";
    dg::geo::BinaryFunctorsLvl1 nc = dg::geo::make_NablaPsiInvCollective( c.get_psip());
    dg::geo::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hectorAdapt( c.get_psip(), nc, psi_0, psi_1, gp.R_0, 0., nGrid,NxGrid,NyGrid, 1e-10, true);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d g2d(hectorAdapt, n, Nx, Ny);
        compute_error_elliptic(c, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "Ribeiro:\n";
    dg::geo::Ribeiro ribeiro( c.get_psip(), psi_0, psi_1, gp.R_0, 0.);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d g2d(ribeiro, n, Nx, Ny);
        compute_error_elliptic(c, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }

    return 0;
}
