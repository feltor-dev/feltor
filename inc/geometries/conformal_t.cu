#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"

#include "dg/backend/timer.cuh"
//#include "guenther.h"
#include "solovev.h"
//#include "conformal.h"
#include "orthogonal.h"
#include "dg/ds.h"
#include "init.h"

#include "file/nc_utilities.h"

double sineX( double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY( double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}
//typedef dg::FieldAligned< solovev::ConformalRingGrid3d<dg::DVec> , dg::IDMatrix, dg::DVec> DFA;
typedef dg::FieldAligned< solovev::OrthogonalRingGrid3d<dg::DVec> , dg::IDMatrix, dg::DVec> DFA;

int main( int argc, char* argv[])
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
    //solovev::detail::Fpsi fpsi( gp, -10);
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Constructing conformal grid ... \n";
    t.tic();
    //solovev::ConformalRingGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    //solovev::ConformalRingGrid2d<dg::DVec> g2d = g3d.perp_grid();
    solovev::OrthogonalRingGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    solovev::OrthogonalRingGrid2d<dg::DVec> g2d = g3d.perp_grid();
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[2];
    err = file::define_dimensions(  ncid, dim3d, g2d);
    int coordsID[2], onesID, defID, divBID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim3d, &divBID);

    thrust::host_vector<double> psi_p = dg::pullback( psip, g2d);
    //g.display();
    err = nc_put_var_double( ncid, onesID, psi_p.data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        //X[i] = g.r()[i]*cos(P[i]);
        //Y[i] = g.r()[i]*sin(P[i]);
        X[i] = g2d.r()[i];
        Y[i] = g2d.z()[i];
    }

    const dg::DVec ones = dg::evaluate( dg::one, g2d);
    dg::DVec temp0( g2d.size()), temp1(temp0);
    dg::DVec w3d = dg::create::weights( g2d);

    err = nc_put_var_double( ncid, coordsID[0], X.data());
    err = nc_put_var_double( ncid, coordsID[1], Y.data());
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());

    dg::blas1::pointwiseDivide( g2d.g_xy(), g2d.g_xx(), temp0);
    X=temp0;
    err = nc_put_var_double( ncid, defID, X.data());

    std::cout << "Construction successful!\n";

    //compute error in volume element
    const dg::DVec f_ = g2d.f();
    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    //dg::blas1::pointwiseDot( f_, f_, temp1);
    temp1 = ones;
    //dg::blas1::axpby( 1.0, temp1, 0.0, g2d.g_xx(),  temp1);
    dg::blas1::pointwiseDot( temp1, temp1, temp1);
    dg::blas1::axpby( 1., temp1, -1., temp0, temp0);
    std::cout<< "Rel Error in Determinant is "<<sqrt( dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( temp1, w3d, temp1))<<"\n";

    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::pointwiseDot( temp0, g.g_pp(), temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0);
    std::cout << "Rel Consistency  of volume is "<<sqrt(dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( g2d.vol(), w3d, g2d.vol()))<<"\n";

    //temp0=g.r();
    //dg::blas1::pointwiseDivide( temp0, g.g_xx(), temp0);
    dg::blas1::pointwiseDot( f_, f_, temp0);
    dg::blas1::axpby( 1.0,temp0 , 0.0, g2d.g_xx(), temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    //dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0);
    dg::blas1::axpby( 1., ones, -1., g2d.vol(), temp0);
    std::cout << "Rel Error of volume form is "<<sqrt(dg::blas2::dot( temp0, w3d, temp0))/sqrt( dg::blas2::dot(g2d.vol(), w3d, g2d.vol()))<<"\n";

    solovev::FieldY fieldY(gp);
    //solovev::ConformalField fieldY(gp);
    dg::DVec fby = dg::pullback( fieldY, g2d);
    dg::blas1::pointwiseDot( fby, f_, fby);
    dg::blas1::pointwiseDot( fby, f_, fby);
    //for( unsigned k=0; k<Nz; k++)
        //for( unsigned i=0; i<n*Ny; i++)
        //    for( unsigned j=0; j<n*Nx; j++)
        //        //by[k*n*n*Nx*Ny + i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
        //        fby[i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
    //dg::DVec fby_device = fby;
    dg::blas1::scal( fby, 1./gp.R_0);
    temp0=g2d.r();
    dg::blas1::pointwiseDot( temp0, fby, fby);
    dg::blas1::pointwiseDivide( ones, g2d.vol(), temp0);
    dg::blas1::axpby( 1., temp0, -1., fby, temp1);
    double error= dg::blas2::dot( temp1, w3d, temp1);
    std::cout << "Rel Error of g.g_xx() is "<<sqrt(error/dg::blas2::dot( fby, w3d, fby))<<"\n";
    const dg::DVec vol = dg::create::volume( g3d);
    dg::DVec ones3d = dg::evaluate( dg::one, g3d);
    double volume = dg::blas1::dot( vol, ones3d);

    std::cout << "TEST VOLUME IS:\n";
    gp.psipmax = psi_1, gp.psipmin = psi_0;
    solovev::Iris iris( gp);
    //dg::CylindricalGrid<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
    dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, -1.2*gp.a, 1.2*gp.a, 1, 1e4, 1e4, dg::PER, dg::PER);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec R  = dg::evaluate( dg::coo1, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    /////////////////////////TEST 3d grid//////////////////////////////////////
    /*
    std::cout << "Start DS test!"<<std::endl;
    const dg::DVec vol3d = dg::create::volume( g3d);
    //DFA fieldaligned( solovev::ConformalField( gp, g3d.x(), g3d.f_x()), g3d, gp.rk4eps, dg::NoLimiter()); 
    DFA fieldaligned( solovev::OrthogonalField( gp, g3d.x(), g3d.f_x()), g3d, gp.rk4eps, dg::NoLimiter()); 

    //dg::DS<DFA, dg::DMatrix, dg::DVec> ds( fieldaligned, solovev::ConformalField(gp, g3d.x(), g3d.f_x()), dg::normed, dg::centered);
    dg::DS<DFA, dg::DMatrix, dg::DVec> ds( fieldaligned, solovev::OrthogonalField(gp, g3d.x(), g3d.f_x()), dg::normed, dg::centered);
    dg::DVec B = dg::pullback( solovev::InvB(gp), g3d), divB(B);
    dg::DVec lnB = dg::pullback( solovev::LnB(gp), g3d), gradB(B);
    dg::DVec gradLnB = dg::pullback( solovev::GradLnB(gp), g3d);
    dg::blas1::pointwiseDivide( ones, B, B);
        dg::DVec function = dg::pullback( solovev::FuncNeu(gp), g3d), derivative(function);
        ds( function, derivative);

    ds.centeredT( B, divB);
    std::cout << "Divergence of B is "<<sqrt( dg::blas2::dot( divB, vol3d, divB))<<"\n";

    ds.centered( lnB, gradB);
    std::cout << "num. norm of gradLnB is "<<sqrt( dg::blas2::dot( gradB,vol3d, gradB))<<"\n";
    double norm = sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB) );
    std::cout << "ana. norm of gradLnB is "<<norm<<"\n";
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    X = divB;
    err = nc_put_var_double( ncid, divBID, X.data());
    std::cout << "Error of lnB is    "<<sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB))/norm<<"\n";
    */
    err = nc_close( ncid);


    return 0;
}
