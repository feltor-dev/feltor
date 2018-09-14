#include <iostream>
#include "json/json.h"

#include "mpi.h"

#include "dg/backend/timer.h"
#include "dg/backend/mpi_init.h"
#include "dg/geometry/functions.h"
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
    unsigned n, Nx, Ny, Nz, mx[2];
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( rank == 0)
    {
        std::cout << "Type mx (10) and my (10)\n";
        std::cin >> mx[0] >> mx[1];
        std::cout << "You typed "<<mx[0]<<" "<<mx[1]<<std::endl;
    }
    MPI_Bcast( mx, 2, MPI_INT, 0, MPI_COMM_WORLD);
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint.js");
        is >> js;
    }
    else
    {
        std::ifstream is(argv[1]);
        is >> js;
    }
    dg::geo::solovev::Parameters gp(js);
    if(rank==0)std::cout << "Start DS test on flux grid!"<<std::endl;
    dg::geo::TokamakMagneticField mag = dg::geo::createSolovevField( gp);
    dg::Timer t;
    t.tic();
    double psi_0 = -20, psi_1 = -4;
    dg::geo::FluxGenerator flux( mag.get_psip(), mag.get_ipol(), psi_0, psi_1, gp.R_0, 0., 1);
    if(rank==0)std::cout << "Constructing Grid...\n";
    dg::geo::CurvilinearProductMPIGrid3d g3d(flux, n, Nx, Ny,Nz, dg::DIR, dg::PER, dg::PER, comm);
    if(rank==0)std::cout << "Constructing Fieldlines...\n";
    dg::geo::DS<dg::aProductMPIGeometry3d, dg::MIHMatrix, dg::MHMatrix, dg::MHVec> ds( mag, g3d, dg::NEU, dg::PER, dg::geo::FullLimiter(), dg::normed, dg::centered, 1e-8, mx[0], mx[1], false, true);

    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s\n";
    dg::MHVec B = dg::pullback( dg::geo::InvB(mag), g3d), divB(B);
    dg::MHVec lnB = dg::pullback( dg::geo::LnB(mag), g3d), gradB(B);
    const dg::MHVec gradLnB = dg::pullback( dg::geo::GradLnB(mag), g3d);
    dg::MHVec ones3d = dg::evaluate( dg::one, g3d);
    dg::MHVec vol3d = dg::create::volume( g3d);
    dg::blas1::pointwiseDivide( ones3d, B, B);

    const double sol_norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    ds.centeredDiv( 1., ones3d, 0., divB);
    dg::blas1::axpby( 1., gradLnB, 1, divB);
    double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    if(rank==0)std::cout << "TEST ADJOINT derivatives!\n";
    if(rank==0)std::cout << "Error Centered Divergence of b is "<<norm/sol_norm<<"\n";
    ds.forwardDiv( 1., ones3d, 0., divB);
    dg::blas1::axpby( 1., gradLnB, 1, divB);
    norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    if(rank==0)std::cout << "Error Forward  Divergence of b is "<<norm/sol_norm<<"\n";
    ds.backwardDiv( 1., ones3d, 0., divB);
    dg::blas1::axpby( 1., gradLnB, 1, divB);
    norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    if(rank==0)std::cout << "Error Backward Divergence of b is "<<norm/sol_norm<<"\n";

    ds.centered( 1., lnB, 0., gradB);
    dg::blas1::axpby( 1., gradLnB, -1., gradB);
    norm = sqrt(dg::blas2::dot(gradB, vol3d, gradB));
    if(rank==0)std::cout << "TEST DIRECT derivatives!\n";
    if(rank==0)std::cout << "rel. error of lnB is    "<<norm/sol_norm<<"\n";
    MPI_Finalize();
    return 0;
}
