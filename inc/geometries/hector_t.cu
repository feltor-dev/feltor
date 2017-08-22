#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/backend/xspacelib.cuh"
#include "dg/functors.h"
#include "curvilinear.h"

#include "dg/backend/timer.cuh"
//#include "guenther.h"
#include "solovev.h"
#include "hector.h"
//#include "refined_conformal.h"
#include "init.h"

#include "file/nc_utilities.h"

using namespace dg::geo;

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

int main( int argc, char* argv[])
{
    std::cout << "Type nHector, NxHector, NyHector ( 13 2 10)\n";
    unsigned nGrid, NxGrid, NyGrid;
    std::cin >> nGrid>> NxGrid>>NyGrid;   
    std::cout << "Type epsHector (1e-10)\n";
    double epsHector;
    std::cin >> epsHector;
    std::cout << "Type n, Nx, Ny, Nz ( 3 4 40 1)\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;   
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
    //write parameters from file into variables
    dg::geo::solovev::GeomParameters gp(js);
    dg::geo::BinaryFunctorsLvl2 psip = dg::geo::solovev::createPsip( gp); 
    std::cout << "Psi min "<<psip.f()(gp.R_0, 0)<<"\n";
    std::cout << "Type psi_0 and psi_1\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    std::cout << "Constructing conformal grid ... \n";
    t.tic();
    Hector<dg::IDMatrix, dg::DMatrix, dg::DVec>* hector;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    int construction = 0;
    if( construction == 0)
    {
        hector = new Hector<dg::IDMatrix, dg::DMatrix, dg::DVec>( psip, psi_0, psi_1, gp.R_0, 0., nGrid, NxGrid, NyGrid, epsHector, true);
    }
    else if( construction == 1)
    {
        dg::geo::BinaryFunctorsLvl1 nc = dg::geo::make_NablaPsiInvCollective( psip);
        hector = new Hector<dg::IDMatrix, dg::DMatrix, dg::DVec>( psip, nc, psi_0, psi_1, gp.R_0, 0., nGrid, NxGrid, NyGrid, epsHector, true);
    }
    else
    {
        dg::geo::BinarySymmTensorLvl1 lc = dg::geo::make_LiseikinCollective( psip, 0.1, 0.001);
        hector = new Hector<dg::IDMatrix, dg::DMatrix, dg::DVec>( psip,lc, psi_0, psi_1, gp.R_0, 0., nGrid, NxGrid, NyGrid, epsHector, true);
    }
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dg::CurvilinearProductGrid3d g3d(*hector, n, Nx, Ny,Nz, dg::DIR);
    dg::CurvilinearGrid2d g2d = g3d.perp_grid();

    dg::Grid2d g2d_periodic(g2d.x0(), g2d.x1(), g2d.y0(), g2d.y1(), g2d.n(), g2d.Nx(), g2d.Ny()+1); 
    t.toc();
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    std::cout << "Length in u is    "<<hector->width()<<std::endl;
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "conformal.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim3d[2];
    err = file::define_dimensions(  ncid, dim3d, g2d_periodic);
    int coordsID[2], onesID, defID, confID,volID,divBID;
    err = nc_def_var( ncid, "x_XYP", NC_DOUBLE, 2, dim3d, &coordsID[0]);
    err = nc_def_var( ncid, "y_XYP", NC_DOUBLE, 2, dim3d, &coordsID[1]);
    //err = nc_def_var( ncid, "z_XYP", NC_DOUBLE, 3, dim3d, &coordsID[2]);
    err = nc_def_var( ncid, "psi", NC_DOUBLE, 2, dim3d, &onesID);
    err = nc_def_var( ncid, "deformation", NC_DOUBLE, 2, dim3d, &defID);
    err = nc_def_var( ncid, "error", NC_DOUBLE, 2, dim3d, &confID);
    err = nc_def_var( ncid, "volume", NC_DOUBLE, 2, dim3d, &volID);
    err = nc_def_var( ncid, "divB", NC_DOUBLE, 2, dim3d, &divBID);

    thrust::host_vector<double> psi_p = dg::pullback( psip.f(), g2d);
    //g.display();
    err = nc_put_var_double( ncid, onesID, periodify(psi_p, g2d_periodic).data());
    dg::HVec X( g2d.size()), Y(X); //P = dg::pullback( dg::coo3, g);
    for( unsigned i=0; i<g2d.size(); i++)
    {
        X[i] = g2d.map()[0][i];
        Y[i] = g2d.map()[1][i];
    }

    dg::HVec temp0( g2d.size()), temp1(temp0);
    dg::HVec w2d = dg::create::weights( g2d);

    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g2d_periodic).data());
    //err = nc_put_var_double( ncid, coordsID[2], g.z().data());

    //compute and write deformation into netcdf
    dg::SparseTensor<dg::HVec> metric = g2d.metric();
    dg::HVec g_xx = metric.value(0,0), g_yy=metric.value(1,1);
    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    const dg::HVec ones = dg::evaluate( dg::one, g2d);
    X=temp0;
    err = nc_put_var_double( ncid, defID, periodify(X, g2d_periodic).data());
    //compute and write conformalratio into netcdf
    dg::blas1::pointwiseDivide( g_yy, g_xx, temp0);
    X=temp0;
    err = nc_put_var_double( ncid, confID, periodify(X, g2d_periodic).data());

    std::cout << "Construction successful!\n";

    //compare determinant vs volume form
    dg::blas1::pointwiseDot( g_xx, g_yy, temp0);
    dg::blas1::axpby( 1., temp0, -1., temp1, temp0);
    dg::blas1::transform( temp0, temp0, dg::SQRT<double>());
    dg::blas1::pointwiseDivide( ones, temp0, temp0);
    dg::blas1::transfer( temp0, X);
    err = nc_put_var_double( ncid, volID, periodify(X, g2d_periodic).data());
    dg::SparseElement<dg::HVec> vol = dg::tensor::volume(metric);
    dg::blas1::axpby( 1., temp0, -1., vol.value(), temp0);
    double error = sqrt(dg::blas2::dot( temp0, w2d, temp0)/dg::blas2::dot( vol.value(), w2d, vol.value()));
    std::cout << "Rel Consistency  of volume is "<<error<<"\n";

    std::cout << "TEST VOLUME IS:\n";
    dg::HVec volume = dg::create::volume( g2d);
    dg::HVec ones2d = dg::evaluate( dg::one, g2d);
    double volumeUV = dg::blas1::dot( vol.value(), ones2d);

    volume = dg::create::volume( hector->internal_grid());
    ones2d = dg::evaluate( dg::one, hector->internal_grid());
    double volumeZE = dg::blas1::dot( vol.value(), ones2d);
    std::cout << "volumeUV is "<< volumeUV<<std::endl;
    std::cout << "volumeZE is "<< volumeZE<<std::endl;
    std::cout << "relative difference in volume is "<<fabs(volumeUV - volumeZE)/volumeZE<<std::endl;
    err = nc_close( ncid);
    delete hector;
    return 0;
}
