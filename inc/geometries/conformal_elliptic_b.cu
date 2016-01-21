#include <iostream>

#include "file/read_input.h"

#include "dg/backend/timer.cuh"
#include "dg/backend/grid.h"
#include "dg/backend/gridX.h"
#include "dg/elliptic.h"
#include "dg/cg.h"

#include "solovev.h"
#include "conformal.h"
#include "conformalX.h"



int main(int argc, char**argv)
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params.txt"); 
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
    std::cout << "Constructing conformal grid ... \n";
    t.tic();

    solovev::ConformalRingGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    solovev::ConformalRingGrid2d<dg::DVec> g2d = g3d.perp_grid();
    dg::Elliptic<solovev::ConformalRingGrid3d<dg::DVec>, dg::DMatrix, dg::DVec, dg::DVec> pol( g3d, dg::not_normed, dg::centered);
    
    //solovev::ConformalXGrid3d<dg::DVec> g3d(gp, psi_0, 0.25, 0.,  n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    //solovev::ConformalXGrid2d<dg::DVec> g2d = g3d.perp_grid();
    //dg::Elliptic<solovev::ConformalXGrid3d<dg::DVec>, dg::Composite<dg::DMatrix>, dg::DVec, dg::DVec> pol( g3d, dg::not_normed, dg::centered);
    //psi_1 = g3d.psi1();

    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s\n";
    std::cout << "psi 1 is          "<<psi_1<<"\n";

    dg::DVec x =    dg::pullback( dg::zero, g3d);
    const dg::DVec b =    dg::pullback( solovev::EllipticDirNeuM(gp, psi_0, psi_1), g3d);
    const dg::DVec chi =  dg::pullback( solovev::Bmodule(gp), g3d);
    const dg::DVec solution = dg::pullback( solovev::FuncDirNeu(gp, psi_0, psi_1 ), g3d);
    const dg::DVec vol3d = dg::create::volume( g3d);
    pol.set_chi( chi);
    //compute error
    dg::DVec error( solution);
    const double eps = 1e-6;
    dg::Invert<dg::DVec > invert( x, n*n*Nx*Ny*Nz, eps);
    std::cout << "eps \t # iterations \t error \t time/iteration \n";
    std::cout << eps<<"\t";
    t.tic();
    unsigned number = invert(pol, x,b);
    std::cout <<number<<"\t";
    t.toc();
    dg::blas1::axpby( 1.,x,-1., solution, error);
    double err = dg::blas2::dot( vol3d, error);
    const double norm = dg::blas2::dot( vol3d, solution);
    std::cout << sqrt( err/norm) << "\t" <<t.diff()/(double)number<<"s"<<std::endl;


    return 0;
}
