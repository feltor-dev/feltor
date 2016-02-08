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
#include "solovev.h"
#include "conformalX.h"
#include "orthogonalX.h"
#include "dg/ds.h"
#include "init.h"

#include "file/nc_utilities.h"

//typedef dg::FieldAligned< solovev::ConformalXGrid3d<dg::DVec> , dg::IDMatrix, dg::DVec> DFA;
double sine( double x) {return sin(x);}
double cosine( double x) {return cos(x);}
typedef dg::FieldAligned< orthogonal::GridX3d<dg::DVec> , dg::IDMatrix, dg::DVec> DFA;

thrust::host_vector<double> periodify( const thrust::host_vector<double>& in, const dg::GridX2d& g)
{
    thrust::host_vector<double> out(g.size());
    for( unsigned i=0; i<g.Ny()-1; i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n()+l] = 
            in[((i*g.n() + k)*g.Nx() + j)*g.n()+l];
    for( unsigned i=g.Ny()-1; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.inner_Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n()+l] = 
            in[((0*g.n() + k)*g.Nx() + j)*g.n()+l];
    //////////////////////////////////
    for( unsigned i=g.Ny()-1; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=g.inner_Nx(); j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n()+l] = 
            in[(((i-1)*g.n() + k)*g.Nx() + j)*g.n()+l];
    return out;
}

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny, Nz (Nx must be divided by 4 and Ny by 10) \n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
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
            std::cout << v[i] << " ";
            std::cout << std::endl;
        return -1;}
    //write parameters from file into variables
    solovev::GeomParameters gp(v);
    dg::Timer t;
    solovev::Psip psip( gp); 
    std::cout << "Psi min "<<psip(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 \n";
    double psi_0;
    std::cin >> psi_0;
    std::cout << "Type fx and fy \n";
    double fx_0, fy_0;
    std::cin >> fx_0>> fy_0;
    gp.display( std::cout);
    std::cout << "Constructing conformal grid ... \n";
    t.tic();
    orthogonal::GridX3d<dg::DVec> g3d(gp, psi_0, fx_0, fy_0, n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    orthogonal::GridX2d<dg::DVec> g2d = g3d.perp_grid();
    t.toc();
    //dg::GridX2d g2d_periodic(g2d.x0(), g2d.x1(), g2d.y0(), g2d.y1(), g2d.fx(), g2d.fy(), g2d.n(), g2d.Nx(), g2d.Ny()+1); 
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    dg::Grid1d<double> g1d( g2d.x0(), g2d.x1(), g2d.n(), g2d.Nx());
    dg::HVec x_left = dg::evaluate( sine, g1d), x_right(x_left);
    dg::HVec y_left = dg::evaluate( cosine, g1d);
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "testX.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[2], dim1d[1];
    //err = file::define_dimensions(  ncid, dim3d, g2d_periodic.grid());
    err = file::define_dimensions(  ncid, dim3d, g2d.grid());
    err = file::define_dimension(  ncid, "i", dim1d, g1d);
    int coordsID[2], onesID, defID, volID, divBID;
    int coord1D[5];
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    err = nc_def_var( ncid, "x_left", NC_DOUBLE, 1, dim1d, &coord1D[0]);
    err = nc_def_var( ncid, "y_left", NC_DOUBLE, 1, dim1d, &coord1D[1]);
    err = nc_def_var( ncid, "x_right", NC_DOUBLE, 1, dim1d, &coord1D[2]);
    err = nc_def_var( ncid, "y_right", NC_DOUBLE, 1, dim1d, &coord1D[3]);
    err = nc_def_var( ncid, "f_x", NC_DOUBLE, 1, dim1d, &coord1D[4]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);
    err = nc_def_var( ncid, "volume", NC_DOUBLE, 2, dim3d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim3d, &divBID);

    thrust::host_vector<double> psi_p = dg::pullback( psip, g2d);
    g2d.display();
    //err = nc_put_var_double( ncid, onesID, periodify(psi_p, g2d_periodic).data());
    err = nc_put_var_double( ncid, onesID, g2d.g().data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        //X[i] = g.r()[i]*cos(P[i]);
        //Y[i] = g.r()[i]*sin(P[i]);
        X[i] = g2d.r()[i];
        Y[i] = g2d.z()[i];
    }

    dg::DVec ones = dg::evaluate( dg::one, g2d);
    dg::DVec temp0( g2d.size()), temp1(temp0);
    dg::DVec w3d = dg::create::weights( g2d);

    //err = nc_put_var_double( ncid, coordsID[0], periodify(X, g2d_periodic).data());
    //err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g2d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[0], X.data());
    err = nc_put_var_double( ncid, coordsID[1], Y.data());
    //err = nc_put_var_double( ncid, coord1D[0], g3d.rx0().data());
    //err = nc_put_var_double( ncid, coord1D[1], g3d.zx0().data());
    //err = nc_put_var_double( ncid, coord1D[2], g3d.rx1().data());
    //err = nc_put_var_double( ncid, coord1D[3], g3d.zx1().data());
    //err = nc_put_var_double( ncid, coord1D[4], periodify(g3d.f_x(), g2d_periodic).data());
    err = nc_put_var_double( ncid, coord1D[4], g3d.f_x().data());
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());

    dg::blas1::pointwiseDivide( g2d.g_yy(), g2d.g_xx(), temp0);
    dg::blas1::axpby( 1., ones, -1., temp0, temp0);
    X=temp0;
    //err = nc_put_var_double( ncid, defID, periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, defID, X.data());
    X = g2d.vol();
    //err = nc_put_var_double( ncid, volID, periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, volID, X.data());

    std::cout << "Construction successful!\n";

    //compute error in volume element
    const dg::DVec f_ = g2d.f();
    dg::blas1::pointwiseDot( g2d.g_xx(), g2d.g_yy(), temp0);
    dg::blas1::pointwiseDot( g2d.g_xy(), g2d.g_xy(), temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    //dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDot( f_, f_, temp1);
    dg::blas1::axpby( 1., temp1, 0.001, g2d.g_xx(),  temp1);
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
    dg::blas1::axpby( 1.0,temp0 , 0.001, g2d.g_xx(), temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., g2d.vol(), temp0);
    std::cout << "Rel Error of volume form is "<<sqrt(dg::blas2::dot( temp0, w3d, temp0))/sqrt( dg::blas2::dot(g2d.vol(), w3d, g2d.vol()))<<"\n";
    solovev::conformal::FieldY fieldY(gp);
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

    std::cout << "TEST VOLUME IS:\n";
    gp.psipmax = 0., gp.psipmin = psi_0;
    solovev::Iris iris( gp);
    //dg::CylindricalGrid<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
    dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, -1.1*gp.a*gp.elongation, 1.1*gp.a*gp.elongation, 1, 5e3, 5e3, dg::PER, dg::PER);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::DVec cutter = dg::pullback( iris, g2d), vol( cutter);
    dg::blas1::pointwiseDot(cutter, w3d, vol);
    double volume = dg::blas1::dot( g2d.vol(), vol);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = dg::blas1::dot( vec, g2d_weights);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n";

    /////////////////////////TEST 3d grid//////////////////////////////////////
    std::cout << "Start DS test!"<<std::endl;
    const dg::DVec vol3d = dg::create::volume( g3d);
    DFA fieldaligned( orthogonal::XField( gp, g2d, g2d.g()), g3d, gp.rk4eps, dg::NoLimiter(), dg::NEU); 

    dg::DS<DFA, dg::Composite<dg::DMatrix>, dg::DVec> ds( fieldaligned, orthogonal::XField(gp, g2d, g2d.g()), dg::normed, dg::centered, false);
    dg::DVec B = dg::pullback( solovev::InvB(gp), g3d), divB(B);
    dg::DVec lnB = dg::pullback( solovev::LnB(gp), g3d), gradB(B);
    dg::DVec gradLnB = dg::pullback( solovev::GradLnB(gp), g3d);
    dg::blas1::pointwiseDivide( ones, B, B);

    ds.centeredT( B, divB);
    std::cout << "Divergence of B is "<<sqrt( dg::blas2::dot( divB, vol3d, divB))<<"\n";
    ds.centered( lnB, gradB);
    dg::blas1::axpby( 1., gradB, -1., gradLnB, gradLnB);
    //test if topological shift was correct!!
    //dg::blas1::pointwiseDot(cutter, gradLnB, gradLnB);
    double norm = sqrt( dg::blas2::dot( gradB, vol3d, gradB) );
    std::cout << "rel. error of lnB is    "<<sqrt( dg::blas2::dot( gradLnB, vol3d, gradLnB))/norm<<" (doesn't fullfill boundary conditions so it was cut before separatrix)\n";

    const dg::DVec function = dg::pullback(solovev::FuncNeu(gp), g3d);
    dg::DVec temp(function);
    const dg::DVec derivative = dg::pullback(solovev::DeriNeu(gp), g3d);
    ds( function, temp);
    dg::blas1::axpby( 1., temp, -1., derivative, temp);
    norm = sqrt( dg::blas2::dot( derivative, vol3d, derivative) );
    std::cout << "rel. error of DS  is    "<<sqrt( dg::blas2::dot( temp, vol3d, temp))/norm<<"\n";
    X = gradB;
    //err = nc_put_var_double( ncid, divBID, periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, divBID, X.data());
    err = nc_close( ncid);


    return 0;
}
