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
#include "testfunctors.h"
#include "solovev.h"
#include "flux.h"
#include "fieldaligned.h"
#include "ds.h"
#include "init.h"

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

double sineX(   double x, double y) {return sin(x)*sin(y);}
double cosineX( double x, double y) {return cos(x)*sin(y);}
double sineY(   double x, double y) {return sin(x)*sin(y);}
double cosineY( double x, double y) {return sin(x)*cos(y);}

int main( int argc, char* argv[])
{
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
    //write parameters from file into variables
    dg::geo::solovev::Parameters gp(js);
    {dg::geo::TokamakMagneticField c = dg::geo::createSolovevField( gp);
    std::cout << "Psi min "<<c.psip()(gp.R_0, 0)<<"\n";}
    std::cout << "Type n(3), Nx(8), Ny(80), Nz(20)\n";
    unsigned n, Nx, Ny, Nz;
    std::cin >> n>> Nx>>Ny>>Nz;
    std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    double psi_0, psi_1;
    std::cin >> psi_0>> psi_1;
    gp.display( std::cout);
    dg::Timer t;
    //solovev::detail::Fpsi fpsi( gp, -10);
    t.tic();
    //![doxygen]
    //create the magnetic field
    dg::geo::TokamakMagneticField c = dg::geo::createSolovevField( gp);
    //create a grid generator
    dg::geo::FluxGenerator flux( c.get_psip(), c.get_ipol(), psi_0, psi_1, gp.R_0, 0., 0, false);
    //create a grid
    dg::geo::CurvilinearGrid2d g2d(flux, n, Nx,Ny, dg::NEU);
    //![doxygen]
    dg::geo::CurvilinearProductGrid3d g3d(flux, n, Nx, Ny,Nz, dg::DIR);
    dg::Grid2d g2d_periodic(g2d.x0(), g2d.x1(), g2d.y0(), g2d.y1(), g2d.n(), g2d.Nx(), g2d.Ny()+1);
    t.toc();
    std::cout << "Construction successful!\n";
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    //////////////////////////////setup and write netcdf//////////////////
    int ncid;
    file::NC_Error_Handle err;
    err = nc_create( "flux.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    err = file::define_dimensions(  ncid, dim2d, g2d_periodic);
    int coordsID[2];
    err = nc_def_var( ncid, "xc", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    err = nc_def_var( ncid, "yc", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    dg::HVec X=dg::pullback(dg::cooX2d, g2d), Y=dg::pullback(dg::cooY2d, g2d); //P = dg::pullback( dg::coo3, g);
    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g2d_periodic).data());

    std::map< std::string, std::function< void( dg::HVec&, dg::geo::CurvilinearGrid2d&, dg::geo::solovev::Parameters&, dg::geo::TokamakMagneticField&)> > output = {
        { "Psip", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psip(), g2d);
        }},
        { "PsipR", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psipR(), g2d);
        }},
        { "PsipZ", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psipZ(), g2d);
        }},
        { "g_xx", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result=g2d.metric().value(0,0);
        }},
        { "g_xy", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result=g2d.metric().value(0,1);
        }},
        { "g_yy", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result=g2d.metric().value(1,1);
        }},
        { "g_xy_g_xx", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            dg::blas1::pointwiseDivide( g2d.metric().value(0,1),
                g2d.metric().value(0,0), result);
        }},
        { "g_yy_g_xx", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            dg::blas1::pointwiseDivide( g2d.metric().value(1,1),
                g2d.metric().value(0,0), result);
        }},
        { "vol", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result=dg::tensor::volume(g2d.metric());
        }},
        { "Bzeta", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result=Bzeta;
        }},
        { "Beta", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result=Beta;
        }},
        { "Bphi", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( dg::geo::BFieldP(mag), g2d);
        }},
        { "q-profile", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result = dg::pullback( dg::geo::BFieldP(mag), g2d);
            dg::blas1::pointwiseDivide( result, Beta, result); //Bphi / Beta

        }},
        { "Ipol", []( dg::HVec& result, dg::geo::CurvilinearGrid2d& g2d,
            dg::geo::solovev::Parameters& gp, dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.ipol(), g2d);
        }}
    };
    dg::HVec temp( g2d.size());
    for( auto pair : output)
    {
        int varID;
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 2, dim2d, &varID);
        pair.second( temp, g2d, gp, c);
        err = nc_put_var_double( ncid, varID, periodify(temp, g2d_periodic).data());
    }
    err = nc_close( ncid);
    ///////////////////////some further testing//////////////////////

    const dg::HVec vol3d = dg::create::volume( g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    double volume3d = dg::blas1::dot( vol3d, ones3d);
    const dg::HVec vol2d = dg::create::volume( g2d);
    dg::HVec ones2d = dg::evaluate( dg::one, g2d);
    double volume2d = dg::blas1::dot( vol2d, ones2d);

    std::cout << "TEST VOLUME IS:\n";
    if( psi_0 < psi_1) gp.psipmax = psi_1, gp.psipmin = psi_0;
    else               gp.psipmax = psi_0, gp.psipmin = psi_1;
    dg::geo::Iris iris( c.psip(), gp.psipmin, gp.psipmax);
    dg::CartesianGrid2d g2dC( gp.R_0 -2.0*gp.a, gp.R_0 + 2.0*gp.a, -2.0*gp.a,2.0*gp.a,3, 2e2, 2e2, dg::PER, dg::PER);
    dg::HVec vec  = dg::evaluate( iris, g2dC);
    dg::HVec R  = dg::evaluate( dg::cooX2d, g2dC);
    dg::HVec onesC = dg::evaluate( dg::one, g2dC);
    dg::HVec g2d_weights = dg::create::volume( g2dC);
    double volumeRZP = 2.*M_PI*dg::blas2::dot( vec, g2d_weights, R);
    double volumeRZ = dg::blas2::dot( vec, g2d_weights, onesC);
    std::cout << "volumeXYP is "<< volume3d  <<std::endl;
    std::cout << "volumeRZP is "<< volumeRZP <<std::endl;
    std::cout << "volumeXY  is "<< volume2d  <<std::endl;
    std::cout << "volumeRZ  is "<< volumeRZ  <<std::endl;
    std::cout << "relative difference in volume3d is "<<fabs(volumeRZP - volume3d)/volume3d<<std::endl;
    std::cout << "relative difference in volume2d is "<<fabs(volumeRZ - volume2d)/volume2d<<std::endl;
    std::cout << "Note that the error might also come from the volume in RZP!\n"; //since integration of jacobian is fairly good probably

    return 0;
}
