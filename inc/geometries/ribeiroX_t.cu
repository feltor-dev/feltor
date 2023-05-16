#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>
#include "json/json.h"

#include "dg/algorithm.h"
#include "dg/file/nc_utilities.h"

#include "solovev.h"
#include "taylor.h"
//#include "guenter.h"
#include "curvilinearX.h"
#include "ribeiroX.h"
#include "ds.h"

double sine( double x) {return sin(x);}
double cosine( double x) {return cos(x);}

thrust::host_vector<double> periodify( const thrust::host_vector<double>& in, const dg::GridX3d& g)
{
    assert( g.Nz() == 2);
    thrust::host_vector<double> out(g.size());
    for( unsigned s=0; s<g.Nz(); s++)
    for( unsigned i=0; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((s*g.Ny()+i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[((i*g.n() + k)*g.Nx() + j)*g.n()+l];

    //exchange two segments
    for( unsigned i=g.outer_Ny(); i<2*g.outer_Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((i+g.inner_Ny())*g.n() + k)*g.Nx() + j)*g.n()+l];
    for( unsigned i=g.inner_Ny()+g.outer_Ny(); i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((i-g.inner_Ny())*g.n() + k)*g.Nx() + j)*g.n()+l];
    if( g.outer_Ny() == 0)
    {
    //exchange two segments
    for( unsigned i=0; i<g.Ny()-1; i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((i+1)*g.n() + k)*g.Nx() + j)*g.n()+l];
    for( unsigned i=g.Ny()-1; i<g.Ny(); i++)
    for( unsigned k=0; k<g.n(); k++)
    for( unsigned j=0; j<g.Nx(); j++)
    for( unsigned l=0; l<g.n(); l++)
        out[(((1*g.Ny() + i)*g.n() + k)*g.Nx() + j)*g.n()+l] =
            in[(((0)*g.n() + k)*g.Nx() + j)*g.n()+l];
    }


    return out;
}

int main( int argc, char* argv[])
{
    std::cout << "Type n, Nx, Ny, Nz (Nx must be divided by 4 and Ny by 10) \n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    Json::Value js;
    if( argc==1)
    {
        //std::ifstream is("geometry_params_Xpoint_taylor.json");
        std::ifstream is("geometry_params_Xpoint.json");
        is >> js;
    }
    else
    {
        std::ifstream is(argv[1]);
        is >> js;
    }
    dg::geo::solovev::Parameters gp(js);
    dg::Timer t;
    std::cout << "Type psi_0 \n";
    double psi_0 = -16;
    std::cin >> psi_0;
    std::cout << "Type fx and fy ( fx*Nx and fy*Ny must be integer) \n";
    double fx_0=1./4., fy_0=1./22.;
    std::cin >> fx_0>> fy_0;
    gp.display( std::cout);
    std::cout << "Constructing orthogonal grid ... \n";
    t.tic();
    dg::geo::CylindricalFunctorsLvl2 psip = dg::geo::solovev::createPsip(gp);
    std::cout << "Psi min "<<psip.f()(gp.R_0, 0)<<"\n";
    double R_X = gp.R_0-1.1*gp.triangularity*gp.a;
    double Z_X = -1.1*gp.elongation*gp.a;
    dg::geo::findXpoint( psip, R_X, Z_X);

    double R0 = gp.R_0, Z0 = 0;
    dg::geo::RibeiroX generator(psip, psi_0, fx_0, R_X,Z_X, R0, Z0);
    dg::geo::CurvilinearProductGridX3d g3d(generator, fx_0, fy_0, n, Nx, Ny,Nz, dg::DIR, dg::NEU);
    dg::geo::CurvilinearGridX2d g2d(generator, fx_0, fy_0, n, Nx, Ny, dg::DIR, dg::NEU);
    t.toc();
    dg::GridX3d g3d_periodic(g3d.x0(), g3d.x1(), g3d.y0(), g3d.y1(), g3d.z0(), g3d.z1(), g3d.fx(), g3d.fy(), g3d.n(), g3d.Nx(), g3d.Ny(), 2);
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    dg::Grid1d g1d( g2d.x0(), g2d.x1(), g2d.n(), g2d.Nx());
    dg::HVec x_left = dg::evaluate( sine, g1d), x_right(x_left);
    dg::HVec y_left = dg::evaluate( cosine, g1d);
    int ncid;
    dg::file::NC_Error_Handle err;
    err = nc_create( "ribeiroX.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[3], dim1d[1];
    err = dg::file::define_dimensions(  ncid, dim3d, g3d_periodic.grid());
    //err = dg::file::define_dimensions(  ncid, dim3d, g2d.grid());
    err = dg::file::define_dimension(  ncid, dim1d, g1d, "i");
    int coordsID[2], onesID, defID, volID, divBID;
    int coord1D[5];
    err = nc_def_var( ncid, "xc", NC_DOUBLE, 3, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "yc", NC_DOUBLE, 3, dim3d, &coordsID[1]);
    err = nc_def_var( ncid, "x_left", NC_DOUBLE, 1, dim1d, &coord1D[0]);
    err = nc_def_var( ncid, "y_left", NC_DOUBLE, 1, dim1d, &coord1D[1]);
    err = nc_def_var( ncid, "x_right", NC_DOUBLE, 1, dim1d, &coord1D[2]);
    err = nc_def_var( ncid, "y_right", NC_DOUBLE, 1, dim1d, &coord1D[3]);
    err = nc_def_var( ncid, "f_x", NC_DOUBLE, 1, dim1d, &coord1D[4]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 3, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 3, dim3d, &defID);
    err = nc_def_var( ncid, "volume", NC_DOUBLE, 3, dim3d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 3, dim3d, &divBID);

    thrust::host_vector<double> psi_p = dg::pullback( psip.f(), g2d);
    g2d.display();
    err = nc_put_var_double( ncid, onesID, periodify(psi_p, g3d_periodic).data());
    //err = nc_put_var_double( ncid, onesID, periodify(g2d.g(), g3d_periodic).data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.map()[0][i];
        Y[i] = g2d.map()[1][i];
    }

    dg::DVec ones = dg::evaluate( dg::one, g2d);
    dg::DVec temp0( g2d.size()), temp1(temp0);
    dg::DVec w2d = dg::create::weights( g2d);

    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g3d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g3d_periodic).data());

    dg::SparseTensor<dg::DVec> metric = g2d.metric();
    dg::DVec g_xx = metric.value(0,0), g_xy = metric.value(0,1), g_yy=metric.value(1,1);
    dg::DVec vol = dg::tensor::volume(metric);

    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    dg::blas1::axpby( 1., ones, -1., temp0, temp0);
    dg::assign( temp0, X);
    err = nc_put_var_double( ncid, defID, periodify(X, g3d_periodic).data());
    //err = nc_put_var_double( ncid, defID, X.data());
    dg::assign( vol, X);
    dg::assign( g_yy,Y);
    dg::blas1::pointwiseDot( Y, X, X);
    err = nc_put_var_double( ncid, volID, periodify(X, g3d_periodic).data());
    //err = nc_put_var_double( ncid, volID, X.data());

    std::cout << "Construction successful!\n";

    //compute error in volume element (in conformal grid g^xx is the volume element)
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::pointwiseDot( g_xy, g_xy, temp1);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::assign( g_xx,  temp1);
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
    dg::assign( temp0, X);
    err = nc_put_var_double( ncid, volID, periodify(X, g3d_periodic).data());
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error = sqrt(dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( vol, w2d, vol));
    std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    //compare g^xx to volume form
    dg::assign( g_xx, temp0);
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::axpby( 1., temp0, -1., vol, temp0);
    error=sqrt(dg::blas2::dot( temp0, w2d, temp0))/sqrt( dg::blas2::dot(vol, w2d, vol));
    std::cout << "Rel Error of volume form is "<<error<<"\n";

    std::cout << "TEST VOLUME IS:\n";
    dg::CartesianGrid2d g2dC( gp.R_0 -1.2*gp.a, gp.R_0 + 1.2*gp.a, -2.0*gp.a*gp.elongation, 1.2*gp.a*gp.elongation, 1, 5e3, 1e4, dg::PER, dg::PER);
    double psipmax = 0., psipmin = psi_0;
    auto iris = dg::compose( dg::Iris(  psipmin, psipmax), psip.f());
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::DVec cutter = dg::pullback( iris, g2d), cut_vol( cutter);
    dg::blas1::pointwiseDot(cutter, w2d, cut_vol);
    double volume = dg::blas1::dot( vol, cut_vol);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = dg::blas1::dot( vec, g2d_weights);
    std::cout << "volumeXYP is "<< volume<<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeRZP - volume)/volume<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n";

    err = nc_close( ncid);
    return 0;
}
