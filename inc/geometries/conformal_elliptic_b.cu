#include <iostream>

#include "file/read_input.h"
#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
//#include "guenther.h"
#include "conformal.h"
#include "orthogonal.h"
#include "curvilinear.h"

const unsigned nIter=6;
template<class Geometry>
void compute_error_elliptic( const solovev::GeomParameters& gp, const Geometry& g2d, double psi_0, double psi_1, double eps)
{
    dg::DVec x =    dg::evaluate( dg::zero, g2d);
    /////////////////////////////DirNeu/////FLUXALIGNED//////////////////////
    //dg::Elliptic<Geometry, dg::DMatrix, dg::DVec> pol( g2d, dg::DIR_NEU, dg::PER, dg::not_normed, dg::forward);
    //const dg::DVec b =    dg::pullback( solovev::EllipticDirPerM(gp, psi_0, 2.*psi_1-psi_0, 0), g2d);
    //const dg::DVec chi =  dg::pullback( solovev::Bmodule(gp), g2d);
    //const dg::DVec solution = dg::pullback( solovev::FuncDirPer(gp, psi_0, 2.*psi_1-psi_0, 0 ), g2d);
    /////////////////////////////Dir/////FIELALIGNED SIN///////////////////
    //dg::Elliptic<Geometry, dg::DMatrix, dg::DVec> pol( g2d, dg::DIR, dg::PER, dg::not_normed, dg::forward);
    //const dg::DVec b =    dg::pullback( solovev::EllipticDirPerM(gp, psi_0, psi_1, 4), g2d);
    //const dg::DVec chi =  dg::pullback( solovev::Bmodule(gp), g2d);
    //const dg::DVec solution = dg::pullback( solovev::FuncDirPer(gp, psi_0, psi_1, 4 ), g2d);
    /////////////////////////////Dir//////BLOB/////////////////////////////
    dg::Elliptic<Geometry, dg::DMatrix, dg::DVec> pol( g2d, dg::DIR, dg::PER, dg::not_normed, dg::forward);
    const dg::DVec b =    dg::pullback( solovev::EllipticDirNeuM(gp, psi_0, psi_1, 440, -220, 40.,1.), g2d);
    const dg::DVec chi =  dg::pullback( solovev::BmodTheta(gp), g2d);
    const dg::DVec solution = dg::pullback( solovev::FuncDirNeu(gp, psi_0, psi_1, 440, -220, 40.,1.), g2d);
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
    dg::DVec gyy = g2d.g_xx(), gxx=g2d.g_yy(), vol = g2d.vol();
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
    solovev::GeomParameters gp(js);
    gp.display( std::cout);
    solovev::CollectivePsip c( gp); 
    const double eps = 1e-10;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "eps\tn\t Nx\t Ny \t # iterations \t error  \t time/iteration (s)\t hx_max\t hy_max\t hx_min\t hy_min \n";
    std::cout << "Orthogonal:\n";
    dg::SimpleOrthogonal<solovev::Psip, solovev::PsipR, solovev::PsipZ, solovev::LaplacePsip> generator0(c.psip, c.psipR, c.psipZ, c.laplacePsip, psi_0, psi_1, gp.R_0, 0., 0);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::OrthogonalGrid2d<dg::DVec> g2d(generator0, n, Nx, Ny);
        compute_error_elliptic(gp, g2d, psi_0, psi_1, eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "Orthogonal Adapted:\n";
    dg::SimpleOrthogonal<solovev::Psip, solovev::PsipR, solovev::PsipZ, solovev::LaplacePsip> generator1(c.psip, c.psipR, c.psipZ, c.laplacePsip, psi_0, psi_1, gp.R_0, 0., 1);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::OrthogonalGrid2d<dg::DVec> g2d(generator1, n, Nx, Ny);
        compute_error_elliptic(gp, g2d, psi_0, psi_1, eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "Conformal:\n";
    dg::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hectorConf( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, psi_0, psi_1, gp.R_0, 0., nGrid,NxGrid,NyGrid, 1e-10, true);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::ConformalGrid2d<dg::DVec> g2d(hectorConf, n, Nx, Ny);
        compute_error_elliptic(gp, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "ConformalMonitor:\n";
    dg::LiseikinCollective<solovev::PsipR, solovev::PsipZ, solovev::PsipRR, solovev::PsipRZ, solovev::PsipZZ> lc( c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, 0.1, 0.001);
    dg::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hectorMonitor( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, lc.chi_XX, lc.chi_XY, lc.chi_YY, lc.divChiX, lc.divChiY, psi_0, psi_1, gp.R_0, 0., nGrid,NxGrid,NyGrid, 1e-10, true);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d<dg::DVec> g2d(hectorMonitor, n, Nx, Ny);
        compute_error_elliptic(gp, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "ConformalAdaption:\n";
    dg::NablaPsiInvCollective<solovev::PsipR, solovev::PsipZ, solovev::PsipRR, solovev::PsipRZ, solovev::PsipZZ> nc( c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ);
    dg::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hectorAdapt( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, nc.nablaPsiInv, nc.nablaPsiInvX, nc.nablaPsiInvY, psi_0, psi_1, gp.R_0, 0., nGrid,NxGrid,NyGrid, 1e-10, true);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::OrthogonalGrid2d<dg::DVec> g2d(hectorAdapt, n, Nx, Ny);
        compute_error_elliptic(gp, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }
    Nx=NxIni, Ny=NyIni;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    std::cout << "Ribeiro:\n";
    dg::Ribeiro<solovev::Psip, solovev::PsipR, solovev::PsipZ, solovev::PsipRR, solovev::PsipRZ, solovev::PsipZZ>
      ribeiro( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, psi_0, psi_1, gp.R_0, 0.);
    for( unsigned i=0; i<nIter; i++)
    {
        dg::CurvilinearGrid2d<dg::DVec> g2d(ribeiro, n, Nx, Ny);
        compute_error_elliptic(gp, g2d, psi_0, psi_1,eps);
        compute_cellsize(g2d);
        std::cout <<std::endl;
        Nx*=2; Ny*=2;
    }

    return 0;
}
