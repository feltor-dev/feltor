#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "json/json.h"

#include "dg/algorithm.h"

#include "curvilinear.h"
//#include "guenther.h"
#include "solovev.h"
#include "ribeiro.h"
//#include "ds.h"

#include "dg/file/nc_utilities.h"

thrust::host_vector<double> periodify( const thrust::host_vector<double>& in, const dg::Grid2d& g)
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
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[((i*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[((0*g.n() + k)*g.Nx() + j)*g.n()+l];
    return out;
}

double sineX( double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY( double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny, Nz\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    Json::Value js;
    if( argc==1)
    {
        std::ifstream is("geometry_params_Xpoint.json");
        is >> js;
    }
    else
    {
        std::ifstream is(argv[1]);
        is >> js;
    }
    //write parameters from file into variables
    dg::geo::solovev::Parameters gp(js);
    dg::geo::CylindricalFunctorsLvl2 psip = dg::geo::solovev::createPsip( gp);
    std::cout << "Psi min "<<psip.f()(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    std::cout << "Constructing ribeiro grid ... \n";
    t.tic();
    dg::geo::Ribeiro ribeiro( psip, psi_0, psi_1, gp.R_0, 0., 1);
    dg::geo::CurvilinearProductGrid3d g3d(ribeiro, n, Nx, Ny,Nz, dg::DIR);
    std::unique_ptr<dg::aGeometry2d> g2d( g3d.perp_grid());
    dg::Grid2d g2d_periodic(g2d->x0(), g2d->x1(), g2d->y0(), g2d->y1(), g2d->n(), g2d->Nx(), g2d->Ny()+1);
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    int ncid;
    dg::file::NC_Error_Handle err;
    err = nc_create( "ribeiro.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[2];
    err = dg::file::define_dimensions(  ncid, dim3d, g2d_periodic);
    int coordsID[2], onesID, defID, confID, volID,divBID;
    err = nc_def_var( ncid, "xc", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "yc", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "zc", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);
    err = nc_def_var( ncid, "conformal", NC_DOUBLE, 2, dim3d, &confID);
    err = nc_def_var( ncid, "volume", NC_DOUBLE, 2, dim3d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim3d, &divBID);

    thrust::host_vector<double> psi_p = dg::pullback( psip.f(), *g2d);
    //g.display();
    err = nc_put_var_double( ncid, onesID, periodify(psi_p, g2d_periodic).data());
    dg::HVec X( g2d->map()[0]), Y(g2d->map()[1]);
    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g2d_periodic).data());

    dg::HVec temp0( g2d->size()), temp1(temp0);
    dg::HVec w2d = dg::create::weights( *g2d);

    dg::SparseTensor<dg::HVec> metric = g2d->metric();
    dg::HVec g_xx = metric.value(0,0), g_xy = metric.value(0,1), g_yy=metric.value(1,1);
    dg::HVec vol = dg::tensor::volume(metric);
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());
    //compute and write deformation into netcdf
    dg::blas1::pointwiseDivide( g_xy, g_xx, temp0);
    const dg::HVec ones = dg::evaluate( dg::one, *g2d);
    X=g_yy;
    err = nc_put_var_double( ncid, defID, periodify(X, g2d_periodic).data());
    //compute and write ribeiroratio into netcdf
    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    X=temp0;

    err = nc_put_var_double( ncid, confID, periodify(X, g2d_periodic).data());
    std::cout << "Construction successful!\n";

    //compute error in volume element (in ribeiro grid g^xx is the volume element)
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transfer( g_xx,  temp1);
    dg::blas1::pointwiseDot( temp1, temp1, temp1);
    dg::blas1::axpby( 1., temp1, -1., temp0, temp0);
    double error = sqrt( dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( temp1, w2d, temp1));
    std::cout<< "Rel Error in Determinant is "<<error<<"\n";

    //compute error in determinant vs volume form
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::transfer( temp0, X);
    err = nc_put_var_double( ncid, volID, periodify(X, g2d_periodic).data());
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error = sqrt(dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( vol, w2d, vol));
    std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //compare g^xx to volume form
    dg::blas1::transfer( g_xx, temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error=sqrt(dg::blas2::dot( temp0, w2d, temp0))/sqrt( dg::blas2::dot(vol, w2d, vol));
    std::cout << "Rel Error of volume form is "<<error<<"\n";

    vol = dg::create::volume( g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    double volume = dg::blas1::dot( vol, ones3d);

    std::cout << "TEST VOLUME IS:\n";
    double psipmin, psipmax;
    if( psi_0 < psi_1) psipmax = psi_1, psipmin = psi_0;
    else               psipmax = psi_0, psipmin = psi_1;
    auto iris = dg::compose( dg::Iris(psipmin, psipmax), psip.f());
    //dg::CylindricalGrid3d<dg::HVec> g3d( gp.R_0 -2.*gp.a, gp.R_0 + 2*gp.a, -2*gp.a, 2*gp.a, 0, 2*M_PI, 3, 2200, 2200, 1, dg::PER, dg::PER, dg::PER);
//     dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, -1.2*gp.a, 1.2*gp.a, 1, 1e3, 1e3, dg::PER, dg::PER);
    dg::CartesianGrid2d g2dC( gp.R_0 -2.0*gp.a, gp.R_0 + 2.0*gp.a, -2.0*gp.a, 2.0*gp.a, 1, 2e3, 2e3, dg::PER, dg::PER);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec R  = dg::evaluate( dg::cooX2d, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    return 0;
}
