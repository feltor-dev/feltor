#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include <mpi.h>

#include "dg/algorithm.h"
#include "mpi_curvilinear.h"
//#include "guenter.h"
#include "solovev.h"
#include "ribeiro.h"
#include "simple_orthogonal.h"
//#include "ds.h"

#include "dg/file/file.h"

double sineX( double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY( double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}

//should be the same as conformal_t.cu, except for the periodify
int main( int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz;
    MPI_Comm comm;
    dg::mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( comm, 3, dims, periods, coords);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    if( dims[2] != 1)
    {
        // because of netcdf output
        if(rank==0) std::cout << "Please do not parallelize in z!\n";
        MPI_Finalize();
        return 0;
    }
    auto js = dg::file::file2Json( argc == 1 ? "geometry_params_Xpoint.json" : argv[1]);
    dg::geo::solovev::Parameters gp(js);
    dg::geo::CylindricalFunctorsLvl2 psip = dg::geo::solovev::createPsip( gp);
    if(rank==0)std::cout << "Psi min "<<psip.f()(gp.R_0, 0)<<"\n";
    if(rank==0)std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    double psi_0, psi_1;
    if(rank==0)std::cin >> psi_0>> psi_1;
    MPI_Bcast( &psi_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &psi_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank==0)gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    if(rank==0)std::cout << "Constructing grid ... \n";
    t.tic();
    dg::geo::Ribeiro ribeiro( psip, psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::CurvilinearProductMPIGrid3d g3d(ribeiro, n, Nx, Ny,Nz, dg::DIR,dg::PER, dg::PER,comm);
    dg::ClonePtr<dg::aMPIGeometry2d> g2d = g3d.perp_grid();
    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    dg::file::NcFile file( "test_mpi.nc", dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"}}, g2d->abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"}}, g2d->abscissas(1));

    dg::MHVec psi_p = dg::pullback( psip.f(), *g2d);
    //g.display();
    file.defput_var( "psi", {"y", "x"}, {}, {*g2d}, psi_p);
    file.defput_var( "xc", {"y", "x"}, {}, {*g2d}, g2d->map()[0]);
    file.defput_var( "yc", {"y", "x"}, {}, {*g2d}, g2d->map()[1]);

    dg::MHVec temp0( dg::evaluate(dg::zero, *g2d)), temp1(temp0);
    dg::MHVec w2d = dg::create::weights( *g2d);

    dg::SparseTensor<dg::MHVec> metric = g2d->metric();
    dg::MHVec g_xx = metric.value(0,0), g_xy = metric.value(0,1), g_yy=metric.value(1,1);
    dg::MHVec vol = dg::tensor::volume(metric);
    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    const dg::MHVec ones = dg::evaluate( dg::one, *g2d);
    dg::blas1::axpby( 1., ones, -1., temp0, temp0);
    file.defput_var( "deformation", {"y", "x"}, {}, {*g2d}, temp0);

    if(rank==0)std::cout << "Construction successful!\n";

    //compute error in volume element
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::assign( g_xx,  temp1);
    dg::blas1::pointwiseDot( temp1, temp1, temp1);
    dg::blas1::axpby( 1., temp1, -1., temp0, temp0);
    double error = sqrt( dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( temp1, w2d, temp1));
    if(rank==0)std::cout<< "Rel Error in Determinant is "<<error<<"\n";

    //compute error in determinant vs volume form
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    file.defput_var( "volume", {"y", "x"}, {}, {*g2d}, temp0);
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error = sqrt(dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( vol, w2d, vol));
    if(rank==0)std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //compare g^xx to volume form
    dg::assign( g_xx, temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error=sqrt(dg::blas2::dot( temp0, w2d, temp0))/sqrt( dg::blas2::dot(vol, w2d, vol));
    if(rank==0)std::cout << "Difference vol - 1/g_xx "<<error<<"\n";

    vol = dg::create::volume( g3d);
    dg::MHVec ones3d = dg::evaluate( dg::one, g3d);
    double volume = dg::blas1::dot( vol, ones3d);

    if(rank==0)std::cout << "TEST VOLUME IS:\n";
    double psipmin, psipmax;
    if( psi_0 < psi_1) psipmax = psi_1, psipmin = psi_0;
    else               psipmax = psi_0, psipmin = psi_1;
    auto iris = dg::compose( dg::Iris(psipmin, psipmax), psip.f());
    //dg::CylindricalGrid3d<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
    dg::CartesianMPIGrid2d g2dC( gp.R_0 -2.*gp.a, gp.R_0 + 2.*gp.a, -2.*gp.a, 2.*gp.a, 1, 2e3, 2e3, dg::DIR, dg::PER, g2d->communicator());
    dg::MHVec vec  = dg::evaluate( iris, g2dC);
    dg::MHVec R  = dg::evaluate( dg::cooX2d, g2dC);
    dg::MHVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    if(rank==0)std::cout << "volumeXYP is "<< volume<<std::endl;
    if(rank==0)std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    if(rank==0)std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    if(rank==0)std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    file.close();
    MPI_Finalize();


    return 0;
}
