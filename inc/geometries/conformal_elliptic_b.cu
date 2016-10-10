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



int main(int argc, char**argv)
{
    //std::cout << "Type nGrid, NxGrid, NyGrid (for hector)\n";
    //unsigned nGrid, NxGrid, NyGrid;
    //std::cin >> nGrid>> NxGrid>>NyGrid;   
    std::cout << "Type n, Nx, Ny (for conformal grid)\n";
    unsigned n, Nx, Ny;
    std::cin >> n>> Nx>>Ny;   
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
    dg::Timer t;
    std::cout << "Constructing grid ... \n";
    t.tic();
    solovev::CollectivePsip c( gp); 
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //dg::Hector<dg::IHMatrix, dg::HMatrix, dg::HVec> hector( c.psip, c.psipR, c.psipZ, c.laplacePsip, psi_0, psi_1, gp.R_0, 0.);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //dg::NablaPsiInvCollective<solovev::PsipR, solovev::PsipZ, solovev::PsipRR, solovev::PsipRZ, solovev::PsipZZ> nc( c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ);
    //dg::Hector<dg::IDMatrix, dg::DMatrix, dg::DVec> hector( c.psip, c.psipR, c.psipZ, c.laplacePsip, nc.nablaPsiInv, nc.nablaPsiInvX, nc.nablaPsiInvY, psi_0, psi_1, gp.R_0, 0.);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //dg::LiseikinCollective<solovev::PsipR, solovev::PsipZ, solovev::PsipRR, solovev::PsipRZ, solovev::PsipZZ> lc( c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, 0.1, 0.001);
    //dg::Hector<dg::IDMatrix, dg::DMatrix, dg::DVec> hector( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, lc.chi_XX, lc.chi_XY, lc.chi_YY, lc.divChiX, lc.divChiY, psi_0, psi_1, gp.R_0, 0.);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    //dg::SimpleOrthogonal<solovev::Psip, solovev::PsipR, solovev::PsipZ, solovev::LaplacePsip> generator(c.psip, c.psipR, c.psipZ, c.laplacePsip, psi_0, psi_1, gp.R_0, 0., 1);
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dg::Ribeiro<solovev::Psip, solovev::PsipR, solovev::PsipZ, solovev::PsipRR, solovev::PsipRZ, solovev::PsipZZ>
      ribeiro( c.psip, c.psipR, c.psipZ, c.psipRR, c.psipRZ, c.psipZZ, psi_0, psi_1, gp.R_0, 0.);

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    std::cout << "eps\t Nx\t Ny \t # iterations \t error \t hx_max\t hy_max \t time/iteration (s) \n";
    for( unsigned i=0; i<6; i++)
    {
        //dg::conformal::RingGrid2d<dg::DVec> g2d(hector, n, Nx, Ny, dg::DIR_NEU);
        //dg::Elliptic<dg::conformal::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
        //dg::orthogonal::RingGrid2d<dg::DVec> g2d(hector, n, Nx, Ny, dg::DIR_NEU);
        //dg::Elliptic<dg::orthogonal::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
        //dg::curvilinear::RingGrid2d<dg::DVec> g2d(hector, n, Nx, Ny, dg::DIR_NEU);
        //dg::Elliptic<dg::curvilinear::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
        //dg::orthogonal::RingGrid2d<dg::DVec> g2d(generator, n, Nx, Ny, dg::DIR_NEU);
        //dg::Elliptic<dg::orthogonal::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
        dg::curvilinear::RingGrid2d<dg::DVec> g2d(ribeiro, n, Nx, Ny, dg::DIR_NEU);
        dg::Elliptic<dg::curvilinear::RingGrid2d<dg::DVec>, dg::DMatrix, dg::DVec> pol( g2d, dg::not_normed, dg::forward);
        dg::DVec x =    dg::evaluate( dg::zero, g2d);
        /////////////////////////////DirNeu/////FLUXALIGNED//////////////////////
        const dg::DVec b =    dg::pullback( solovev::EllipticDirPerM(gp, psi_0, 2.*psi_1-psi_0, 0), g2d);
        const dg::DVec chi =  dg::pullback( solovev::Bmodule(gp), g2d);
        const dg::DVec solution = dg::pullback( solovev::FuncDirPer(gp, psi_0, 2.*psi_1-psi_0, 0 ), g2d);
        /////////////////////////////Dir/////FIELALIGNED SIN///////////////////
        //const dg::DVec b =    dg::pullback( solovev::EllipticDirPerM(gp, psi_0, psi_1, 4), g2d);
        //const dg::DVec chi =  dg::pullback( solovev::Bmodule(gp), g2d);
        //const dg::DVec solution = dg::pullback( solovev::FuncDirPer(gp, psi_0, psi_1, 4 ), g2d);
        /////////////////////////////Dir//////BLOB/////////////////////////////
        //const dg::DVec b =    dg::pullback( solovev::EllipticDirNeuM(gp, psi_0, psi_1, 510, -140, 40.,1.), g2d);
        //const dg::DVec chi =  dg::pullback( solovev::BmodTheta(gp), g2d);
        //const dg::DVec solution = dg::pullback( solovev::FuncDirNeu(gp, psi_0, psi_1, 510, -140, 40.,1.), g2d);
        const dg::DVec vol2d = dg::create::volume( g2d);
        pol.set_chi( chi);
        //compute error
        dg::DVec error( solution);
        const double eps = 1e-10;
        std::cout << eps<<"\t"<<Nx<<"\t"<<Ny<<"\t";
        t.tic();
        dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny, eps);
        unsigned number = invert(pol, x,b);
        std::cout <<number<<"\t";
        t.toc();
        dg::blas1::axpby( 1.,x,-1., solution, error);
        double err = dg::blas2::dot( vol2d, error);
        const double norm = dg::blas2::dot( vol2d, solution);
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
        std::cout << *thrust::min_element( gxx.begin(), gxx.end()) << "\t";
        std::cout << *thrust::min_element( gyy.begin(), gyy.end()) << "\t";
        std::cout<<t.diff()/(double)number<<std::endl;
        Nx*=2; Ny*=2;
    }

    return 0;
}
