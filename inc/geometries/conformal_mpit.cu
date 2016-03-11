#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include <mpi.h>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"

#include "dg/backend/timer.cuh"
#include "dg/backend/mpi_init.h"
//#include "guenther.h"
#include "solovev.h"
#include "mpi_conformal.h"
//#include "mpi_orthogonal.h"
#include "dg/ds.h"
#include "init.h"

#include <netcdf_par.h>
#include "file/nc_utilities.h"

double sineX( double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY( double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}
typedef dg::MPI_FieldAligned< conformal::MPIRingGrid3d<dg::HVec> , dg::IHMatrix, dg::BijectiveComm<dg::iHVec, dg::HVec>, dg::HVec> DFA;
//typedef dg::FieldAligned< orthogonal::RingGrid3d<dg::HVec> , dg::IHMatrix, dg::HVec> DFA;

//should be the same as conformal_t.cu, except for the periodify
int main( int argc, char* argv[])
{
    MPI_Init( &argc, &argv);
    int rank;
    unsigned n, Nx, Ny, Nz; 
    MPI_Comm comm;
    mpi_init3d( dg::DIR, dg::PER, dg::PER, n, Nx, Ny, Nz, comm);
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);

    std::vector<double> v, v2;
    try{ 
        if( argc==1)
        {
            v = file::read_input( "geometry_params_Xpoint.txt"); 
        }
        else
        {
            v = file::read_input( argv[1]); 
        }
    }
    catch (toefl::Message& m) {  
        m.display(); 
        for( unsigned i = 0; i<v.size(); i++)
            if(rank==0)std::cout << v[i] << " ";
            if(rank==0)std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    solovev::GeomParameters gp(v);
    solovev::Psip psip( gp); 
    if(rank==0)std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    if(rank==0)std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    if(rank==0)std::cin >> psi_0>> psi_1;
    MPI_Bcast( &psi_0, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast( &psi_1, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    if(rank==0)gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    if(rank==0)std::cout << "Constructing conformal grid ... \n";
    t.tic();
    conformal::MPIRingGrid3d<dg::HVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR,comm);
    conformal::MPIRingGrid2d<dg::HVec> g2d = g3d.perp_grid();
    //orthogonal::RingGrid3d<dg::HVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    //orthogonal::RingGrid2d<dg::HVec> g2d = g3d.perp_grid();
    //
    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    MPI_Info info = MPI_INFO_NULL;
    err = nc_create_par( "test_mpi.nc", NC_NETCDF4|NC_MPIIO|NC_CLOBBER, comm, info, &ncid); //MPI ON
    int dim3d[2];
    err = file::define_dimensions(  ncid, dim3d, g2d.global());
    int coordsID[2], onesID, defID, divBID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim3d, &divBID);

    dg::MHVec psi_p = dg::pullback( psip, g2d);
    //g.display();
    err = nc_put_var_double( ncid, onesID, psi_p.data().data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.r()[i];
        Y[i] = g2d.z()[i];
    }

    dg::MHVec temp0( dg::evaluate(dg::zero, g2d)), temp1(temp0);
    dg::MHVec w3d = dg::create::weights( g2d);

    err = nc_put_var_double( ncid, coordsID[0], X.data());
    err = nc_put_var_double( ncid, coordsID[1], Y.data());
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());

    //dg::blas1::pointwiseDivide( g2d.g_xy(), g2d.g_xx(), temp0);
    dg::blas1::pointwiseDivide( g2d.g_yy(), g2d.g_xx(), temp0);
    const dg::MHVec ones = dg::evaluate( dg::one, g2d);
    dg::blas1::axpby( 1., ones, -1., temp0, temp0);
    X=temp0.data();
    err = nc_put_var_double( ncid, defID, X.data());

    if(rank==0)std::cout << "Construction successful!\n";

    //compute error in volume element
    const dg::MHVec f_ = g2d.f();
    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    //dg::blas1::pointwiseDot( f_, f_, temp1);
    temp1 = ones;
    dg::blas1::axpby( 0.0, temp1, 1.0, g2d.g_xx(),  temp1);
    dg::blas1::pointwiseDot( temp1, temp1, temp1);
    dg::blas1::axpby( 1., temp1, -1., temp0, temp0);
    double error = sqrt( dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( temp1, w3d, temp1));
    if(rank==0)std::cout<< "Rel Error in Determinant is "<<error<<"\n";

    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::pointwiseDot( temp0, g.g_pp(), temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0);
    error = sqrt(dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( g2d.vol(), w3d, g2d.vol()));
    if(rank==0)std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //temp0=g.r();
    //dg::blas1::pointwiseDivide( temp0, g.g_xx(), temp0);
    dg::blas1::pointwiseDot( f_, f_, temp0);
    dg::blas1::axpby( 0.0,temp0 , 1.0, g2d.g_xx(), temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    //dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0);
    dg::blas1::axpby( 1., ones, -1., g2d.vol(), temp0);
    error=sqrt(dg::blas2::dot( temp0, w3d, temp0))/sqrt( dg::blas2::dot(g2d.vol(), w3d, g2d.vol()));
    if(rank==0)std::cout << "Rel Error of volume form is "<<error<<"\n";

    solovev::conformal::FieldY fieldY(gp);
    //solovev::ConformalField fieldY(gp);
    dg::MHVec fby = dg::pullback( fieldY, g2d);
    dg::blas1::pointwiseDot( fby, f_, fby);
    dg::blas1::pointwiseDot( fby, f_, fby);
    //for( unsigned k=0; k<Nz; k++)
        //for( unsigned i=0; i<n*Ny; i++)
        //    for( unsigned j=0; j<n*Nx; j++)
        //        //by[k*n*n*Nx*Ny + i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
        //        fby[i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
    //dg::HVec fby_device = fby;
    dg::blas1::scal( fby, 1./gp.R_0);
    temp0=g2d.r();
    dg::blas1::pointwiseDot( temp0, fby, fby);
    dg::blas1::pointwiseDivide( ones, g2d.vol(), temp0);
    dg::blas1::axpby( 1., temp0, -1., fby, temp1);
    error= dg::blas2::dot( temp1, w3d, temp1)/dg::blas2::dot(fby,w3d,fby);
    if(rank==0)std::cout << "Rel Error of g.g_xx() is "<<sqrt(error)<<"\n";
    const dg::MHVec vol = dg::create::volume( g3d);
    dg::MHVec ones3d = dg::evaluate( dg::one, g3d);
    double volume = dg::blas1::dot( vol, ones3d);

    if(rank==0)std::cout << "TEST VOLUME IS:\n";
    if( psi_0 < psi_1) gp.psipmax = psi_1, gp.psipmin = psi_0;
    else               gp.psipmax = psi_0, gp.psipmin = psi_1;
    solovev::Iris iris( gp);
    //dg::CylindricalGrid<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
        MPI_Comm planeComm;
        int remain_dims[] = {true,true,false}; //true true false
        MPI_Cart_sub( comm, remain_dims, &planeComm);
    dg::CartesianMPIGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, -1.2*gp.a, 1.2*gp.a, 1, 1e3, 1e3, dg::DIR, dg::PER, planeComm);
    dg::MHVec vec  = dg::evaluate( iris, g2dC);
    dg::MHVec R  = dg::evaluate( dg::coo1, g2dC);
    dg::MHVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    if(rank==0)std::cout << "volumeXYP is "<< volume<<std::endl;
    if(rank==0)std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    if(rank==0)std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    if(rank==0)std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    /////////////////////////TEST 3d grid//////////////////////////////////////
    if(rank==0)std::cout << "Start DS test!"<<std::endl;
    const dg::MHVec vol3d = dg::create::volume( g3d);
    t.tic();
    DFA fieldaligned( conformal::Field( gp, g3d.x(), g3d.f_x()), g3d, gp.rk4eps, dg::NoLimiter()); 
    //DFA fieldaligned( orthogonal::Field( gp, g2d, g2d.g()), g3d, gp.rk4eps, dg::NoLimiter()); 

    dg::DS<DFA, dg::MDMatrix, dg::MHVec> ds( fieldaligned, conformal::Field(gp, g3d.x(), g3d.f_x()), dg::normed, dg::centered);
    //dg::DS<DFA, dg::DMatrix, dg::HVec> ds( fieldaligned, orthogonal::Field(gp, g2d, g2d.g()), dg::normed, dg::centered);
    t.toc();
    if(rank==0)std::cout << "Construction took "<<t.diff()<<"s\n";
    dg::MHVec B = dg::pullback( solovev::InvB(gp), g3d), divB(B);
    dg::MHVec lnB = dg::pullback( solovev::LnB(gp), g3d), gradB(B);
    dg::MHVec gradLnB = dg::pullback( solovev::GradLnB(gp), g3d);
    dg::blas1::pointwiseDivide( ones3d, B, B);
    dg::MHVec function = dg::pullback( solovev::FuncNeu(gp), g3d), derivative(function);
    ds( function, derivative);

    ds.centeredT( B, divB);
    double norm =  sqrt( dg::blas2::dot(divB, vol3d, divB));
    if(rank==0)std::cout << "Divergence of B is "<<norm<<"\n";

    ds.centered( lnB, gradB);
    norm = sqrt(dg::blas2::dot(gradB,vol3d,gradB) );
    if(rank==0)std::cout << "num. norm of gradLnB is "<<norm<<"\n";
    norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    if(rank==0)std::cout << "ana. norm of gradLnB is "<<norm<<"\n";
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    X = divB.data();
    err = nc_put_var_double( ncid, divBID, X.data());
    double norm2 = sqrt(dg::blas2::dot(gradLnB, vol3d,gradLnB));
    if(rank==0)std::cout << "rel. error of lnB is    "<<norm2/norm<<"\n";
    err = nc_close( ncid);
    MPI_Finalize();


    return 0;
}
