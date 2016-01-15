#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "file/read_input.h"
#include "draw/host_window.h"

#include "dg/backend/timer.cuh"
#include "guenther.h"
//#include "solovev.h"
#include "conformal.h"
#include "dg/ds.h"
#include "init.h"

#include "file/nc_utilities.h"

double sineX( double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY( double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}
typedef dg::FieldAligned< solovev::ConformalRingGrid3d<dg::DVec> , dg::IDMatrix, dg::DVec> DFA;

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
    solovev::ConformalRingGrid3d<dg::DVec> g3d(gp, psi_0, psi_1, n, Nx, Ny,Nz, dg::DIR);
    solovev::ConformalRingGrid2d<dg::DVec> g = g3d.perp_grid();
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "test.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[2];
    err = file::define_dimensions(  ncid, dim3d, g);
    int coordsID[2], onesID, defID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);

    thrust::host_vector<double> psi_p = dg::pullback( psip, g);
    g.display();
    err = nc_put_var_double( ncid, onesID, psi_p.data());
    dg::HVec X( g.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g.size(); i++)
    {
        //X[i] = g.r()[i]*cos(P[i]);
        //Y[i] = g.r()[i]*sin(P[i]);
        X[i] = g.r()[i];
        Y[i] = g.z()[i];
    }

    dg::DVec ones = dg::evaluate( dg::one, g);
    dg::DVec temp0( g.size()), temp1(temp0);
    dg::DVec w3d = dg::create::weights( g);

    err = nc_put_var_double( ncid, coordsID[0], X.data());
    err = nc_put_var_double( ncid, coordsID[1], Y.data());
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());

    dg::blas1::pointwiseDivide( g.g_xy(), g.g_xx(), temp0);
    X=temp0;
    err = nc_put_var_double( ncid, defID, X.data());
    err = nc_close( ncid);

    std::cout << "Construction successful!\n";

    //compute error in volume element
    dg::blas1::pointwiseDot( g.g_xx(), g.g_yy(), temp0);
    dg::blas1::pointwiseDot( g.g_xy(), g.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDot( g.g_xx(), g.g_xx(), temp1);
    dg::blas1::axpby( 1., temp1, -1., temp0, temp0);
    std::cout<< "Rel Error in Determinant is "<<sqrt( dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( temp1, w3d, temp1))<<"\n";

    dg::blas1::pointwiseDot( g.g_xx(), g.g_yy(), temp0);
    dg::blas1::pointwiseDot( g.g_xy(), g.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::pointwiseDot( temp0, g.g_pp(), temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., g.vol(), temp0);
    std::cout << "Rel Consistency  of volume is "<<sqrt(dg::blas2::dot( temp0, w3d, temp0)/dg::blas2::dot( g.vol(), w3d, g.vol()))<<"\n";

    //temp0=g.r();
    //dg::blas1::pointwiseDivide( temp0, g.g_xx(), temp0);
    dg::blas1::pointwiseDivide( ones, g.g_xx(), temp0);
    dg::blas1::axpby( 1., temp0, -1., g.vol(), temp0);
    std::cout << "Rel Error of volume form is "<<sqrt(dg::blas2::dot( temp0, w3d, temp0))/sqrt( dg::blas2::dot(g.vol(), w3d, g.vol()))<<"\n";

    solovev::FieldY fieldY(gp);
    //solovev::ConformalField fieldY(gp);
    dg::HVec by = dg::pullback( fieldY, g);
    //for( unsigned k=0; k<Nz; k++)
        for( unsigned i=0; i<n*Ny; i++)
            for( unsigned j=0; j<n*Nx; j++)
                //by[k*n*n*Nx*Ny + i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
                by[i*n*Nx + j] *= g.f_x()[j]*g.f_x()[j];
    dg::DVec by_device = by;
    dg::blas1::scal( by_device, 1./gp.R_0);
    temp0=g.r();
    dg::blas1::pointwiseDivide( g.g_xx(), temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., by_device, temp1);
    double error= dg::blas2::dot( temp1, w3d, temp1);
    std::cout << "Rel Error of g.g_xx() is "<<sqrt(error/dg::blas2::dot( by_device, w3d, by_device))<<"\n";
    double volume = dg::blas1::dot( g.vol(), w3d);

    std::cout << "TEST VOLUME IS:\n";
    gp.psipmax = psi_1, gp.psipmin = psi_0;
    solovev::Iris iris( gp);
    //dg::CylindricalGrid<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
    dg::CartesianGrid2d g2d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 3, 2200, 2200, dg::PER, dg::PER);
    dg::DVec vec  = dg::evaluate( iris, g2d);
    dg::DVec g2d_weights = dg::create::volume( g2d);
    double volumeRZP = dg::blas1::dot( vec, g2d_weights);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n";

    /////////////////////////TEST 3d grid//////////////////////////////////////
    std::cout << "Start DS test!"<<std::endl;
    DFA fieldaligned( solovev::ConformalField( gp, g.x(), g.f_x()), g3d, gp.rk4eps, dg::NoLimiter()); 

    dg::DS<DFA, dg::DMatrix, dg::DVec> ds( fieldaligned, solovev::ConformalField(gp, g3d.x(), g3d.f_x()), dg::normed, dg::centered);
    dg::DVec B = dg::pullback( solovev::InvB(gp), g3d), divB(B);
    dg::DVec lnB = dg::pullback( solovev::LnB(gp), g3d), gradB(B);
    dg::DVec gradLnB = dg::pullback( solovev::GradLnB(gp), g3d);
    dg::DVec vol = dg::create::volume( g3d);
    dg::blas1::pointwiseDivide( ones, B, B);

    ds.centeredT( B, divB);
    std::cout << "Divergence of B is "<<sqrt( dg::blas2::dot( divB, vol, divB))<<"\n";
    ds.centered( lnB, gradB);
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    std::cout << "Error of lnB is    "<<sqrt( dg::blas2::dot( gradLnB, vol, gradLnB))<<"\n";


    return 0;
}
