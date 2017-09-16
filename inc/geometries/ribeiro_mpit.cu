#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include <mpi.h>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"

#include "dg/backend/timer.cuh"
#include "dg/backend/mpi_init.h"
#include "mpi_curvilinear.h"
//#include "guenther.h"
#include "solovev.h"
#include "ribeiro.h"
#include "simple_orthogonal.h"
//#include "ds.h"
#include "init.h"

#include <netcdf_par.h>
#include "file/nc_utilities.h"

using namespace dg::geo::solovev;
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
    GeomParameters gp(js);
    dg::geo::BinaryFunctorsLvl2 psip = createPsip( gp);
    if(rank==0)std::cout << "Psi min "<<psip.f()(gp.R_0, 0)<<"\n";
    if(rank==0)std::cout << "Type psi_0 and psi_1\n";
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
    dg::CurvilinearProductMPIGrid3d g3d(ribeiro, n, Nx, Ny,Nz, dg::DIR,dg::PER, dg::PER,comm);
    dg::CurvilinearMPIGrid2d g2d = g3d.perp_grid();
    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    MPI_Info info = MPI_INFO_NULL;
    err = nc_create_par( "test_mpi.nc", NC_NETCDF4|NC_MPIIO|NC_CLOBBER, g2d.communicator(), info, &ncid); //MPI ON
    int dim3d[2];
    err = file::define_dimensions(  ncid, dim3d, g2d.global());
    int coordsID[2], onesID, defID,confID, volID, divBID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);
    err = nc_def_var( ncid, "conformal", NC_DOUBLE, 2, dim3d, &confID);
    err = nc_def_var( ncid, "volume", NC_DOUBLE, 2, dim3d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim3d, &divBID);

    int dims[2], periods[2],  coords[2];
    MPI_Cart_get( g2d.communicator(), 2, dims, periods, coords);
    size_t count[2] = {g2d.n()*g2d.Ny(), g2d.n()*g2d.Nx()};
    size_t start[2] = {coords[1]*count[0], coords[0]*count[1]};
    err = nc_var_par_access( ncid, coordsID[0], NC_COLLECTIVE);
    err = nc_var_par_access( ncid, coordsID[1], NC_COLLECTIVE);
    err = nc_var_par_access( ncid, onesID, NC_COLLECTIVE);
    err = nc_var_par_access( ncid, defID, NC_COLLECTIVE);
    err = nc_var_par_access( ncid, divBID, NC_COLLECTIVE);

    dg::MHVec psi_p = dg::pullback( psip.f(), g2d);
    //g.display();
    err = nc_put_vara_double( ncid, onesID, start, count, psi_p.data().data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.map()[0].data()[i];
        Y[i] = g2d.map()[0].data()[i];
    }

    dg::MHVec temp0( dg::evaluate(dg::zero, g2d)), temp1(temp0);
    dg::MHVec w2d = dg::create::weights( g2d);

    err = nc_put_vara_double( ncid, coordsID[0], start,count, X.data());
    err = nc_put_vara_double( ncid, coordsID[1], start,count, Y.data());

    dg::SparseTensor<dg::MHVec> metric = g2d.metric();
    dg::MHVec g_xx = metric.value(0,0), g_xy = metric.value(0,1), g_yy=metric.value(1,1);
    dg::SparseElement<dg::MHVec> vol_ = dg::tensor::volume(metric);
    dg::MHVec vol = vol_.value();
    //err = nc_put_vara_double( ncid, coordsID[2], g.z().data());
    //dg::blas1::pointwiseDivide( g2d.g_xy(), g2d.g_xx(), temp0);
    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    const dg::MHVec ones = dg::evaluate( dg::one, g2d);
    dg::blas1::axpby( 1., ones, -1., temp0, temp0);
    dg::blas1::transfer( temp0.data(), X);
    err = nc_put_vara_double( ncid, defID, start,count, X.data());

    if(rank==0)std::cout << "Construction successful!\n";

    //compute error in volume element
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transfer( g_xx,  temp1);
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
    dg::blas1::transfer( temp0.data(), X);
    err = nc_put_var_double( ncid, volID, X.data());
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error = sqrt(dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( vol, w2d, vol));
    if(rank==0)std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //compare g^xx to volume form
    dg::blas1::transfer( g_xx, temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error=sqrt(dg::blas2::dot( temp0, w2d, temp0))/sqrt( dg::blas2::dot(vol, w2d, vol));
    if(rank==0)std::cout << "Rel Error of volume form is "<<error<<"\n";

    vol = dg::create::volume( g3d);
    dg::MHVec ones3d = dg::evaluate( dg::one, g3d);
    double volume = dg::blas1::dot( vol, ones3d);

    if(rank==0)std::cout << "TEST VOLUME IS:\n";
    if( psi_0 < psi_1) gp.psipmax = psi_1, gp.psipmin = psi_0;
    else               gp.psipmax = psi_0, gp.psipmin = psi_1;
    dg::geo::Iris iris(psip.f(), gp.psipmin, gp.psipmax);
    //dg::CylindricalGrid3d<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
    dg::CartesianMPIGrid2d g2dC( gp.R_0 -2.*gp.a, gp.R_0 + 2.*gp.a, -2.*gp.a, 2.*gp.a, 1, 2e3, 2e3, dg::DIR, dg::PER, g2d.communicator());
    dg::MHVec vec  = dg::evaluate( iris, g2dC);
    dg::MHVec R  = dg::evaluate( dg::cooX2d, g2dC);
    dg::MHVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    if(rank==0)std::cout << "volumeXYP is "<< volume<<std::endl;
    if(rank==0)std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    if(rank==0)std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    if(rank==0)std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    /////////////////////////TEST 3d grid//////////////////////////////////////
    //if(rank==0)std::cout << "Start DS test!"<<std::endl;
    //const dg::MHVec vol3d = dg::create::volume( g3d);
    //t.tic();
    ////DFA fieldaligned( CurvilinearField( gp, g3d.x(), g3d.f_x()), g3d, gp.rk4eps, dg::NoLimiter()); 
    //DFA fieldaligned( OrthogonalField( gp, g2d.global(), g2d.f2_xy()), g3d, gp.rk4eps, dg::NoLimiter()); 

    ////dg::DS<DFA, dg::MHMatrix, dg::MHVec> ds( fieldaligned, CurvilinearField(gp, g3d.x(), g3d.f_x()), dg::normed, dg::centered);
    //dg::DS<DFA, dg::MHMatrix, dg::MHVec> ds( fieldaligned, OrthogonalField(gp, g2d.global(), g2d.f2_xy()), dg::normed, dg::centered);
    //t.toc();
    //if(rank==0)std::cout << "Construction took "<<t.diff()<<"s\n";
    //dg::MHVec B = dg::pullback( dg::geo::InvB(gp), g3d), divB(B);
    //dg::MHVec lnB = dg::pullback( dg::geo::LnB(gp), g3d), gradB(B); //dg::MHVec gradLnB = dg::pullback( dg::geo::GradLnB(gp), g3d);
    //dg::blas1::pointwiseDivide( ones3d, B, B);
    //dg::MHVec function = dg::pullback( dg::geo::FuncNeu(gp), g3d), derivative(function);
    //ds( function, derivative);

    //ds.centeredT( B, divB);
    //double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    //if(rank==0)std::cout << "Divergence of B is "<<norm<<"\n";

    //ds.centered( lnB, gradB);
    //norm = sqrt(dg::blas2::dot(gradB,vol3d,gradB) );
    //if(rank==0)std::cout << "num. norm of gradLnB is "<<norm<<"\n";
    //norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    //if(rank==0)std::cout << "ana. norm of gradLnB is "<<norm<<"\n";
    //dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    //X = divB.data();
    //err = nc_put_vara_double( ncid, divBID, start,count, X.data());
    //double norm2 = sqrt(dg::blas2::dot(gradLnB, vol3d,gradLnB));
    //if(rank==0)std::cout << "rel. error of lnB is    "<<norm2/norm<<"\n";
    err = nc_close( ncid);
    MPI_Finalize();


    return 0;
}
