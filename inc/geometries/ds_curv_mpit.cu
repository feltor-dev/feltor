#include <iostream>

#include "mpi.h"

#include "dg/backend/timer.cuh"
#include "dg/backend/mpi_init.h"
#include "dg/backend/functions.h"
#include "dg/blas.h"
#include "dg/functors.h"
#include "dg/geometry/geometry.h"
#include "testfunctors.h"
#include "ds.h"
#include "solovev.h"
#include "flux.h"
#include "toroidal.h"
#include "mpi_curvilinear.h"


int main(int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
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
    dg::geo::solovev::Parameters gp(js);
    if(rank==0)std::cout << "Start DS test on flux grid!"<<std::endl;
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    dg::Timer t;
    t.tic();
    unsigned mx=1, my=10;
    double psi_0 = -20, psi_1 = -4;
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    if(rank==0)std::cout << "Constructing Grid...\n";
    dg::geo::CurvilinearProductMPIGrid3d g3d(flux, n, Nx, Ny,Nz, dg::DIR, dg::PER, dg::PER, comm);
    if(rank==0)std::cout << "Constructing Fieldlines...\n";
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIHMatrix, dg::MHMatrix, dg::MHVec> ds( mag, g3d, dg::NEU, dg::NEU, dg::geo::FullLimiter(), dg::normed, dg::centered, 1e-8, mx, my, false, true);
    
    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s\n";
    dg::MHVec B = dg::pullback( dg::geo::InvB(mag), g3d), divB(B);
    dg::MHVec lnB = dg::pullback( dg::geo::LnB(mag), g3d), gradB(B);
    dg::MHVec gradLnB = dg::pullback( dg::geo::GradLnB(mag), g3d);
    dg::MHVec ones3d = dg::evaluate( dg::one, g3d);
    dg::MHVec vol3d = dg::create::volume( g3d);
    dg::blas1::pointwiseDivide( ones3d, B, B);
    dg::MHVec function = dg::pullback( dg::geo::FuncNeu(mag), g3d), derivative(function);
    ds( function, derivative);

    ds.centeredAdj( 1., B, 0., divB);
    double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    if(rank==0)std::cout << "Divergence of B is "<<norm<<"\n";

    ds.centered( 1., lnB, 0., gradB);
    norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    double norm2 = sqrt(dg::blas2::dot(gradLnB, vol3d, gradLnB));
    if(rank==0)std::cout << "rel. error of lnB is    "<<norm2/norm<<"\n";
    MPI_Finalize();
    return 0;
}
