#include <iostream>
#include <iomanip>
#include <vector>
#include <fstream>
#include <sstream>
#include <cmath>

#include "dg/file/file.h"

#include "dg/algorithm.h"

#include "curvilinear.h"
#include "testfunctors.h"
#include "make_field.h"
#include "flux.h"
#include "simple_orthogonal.h"
#include "ribeiro.h"


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
    dg::file::WrappedJsonValue js( dg::file::error::is_throw);
    std::string input = argc==1 ? "flux.json" : argv[1];
    dg::file::file2Json( input, js.asJson(), dg::file::comments::are_discarded);

    if( js["magnetic_field"][ "input"].asString() == "external")
    {
        std::string path = js["magnetic_field"]["path"].asString();
        dg::file::file2Json( path, js.asJson()["magnetic_field"]["params"],
                dg::file::comments::are_discarded);
    }
    //write parameters from file into variables
    //std::cout << "Type n(3), Nx(8), Ny(80), Nz(20)\n";
    dg::file::WrappedJsonValue grid = js["grid"];
    unsigned  n = grid ["n"].asUInt(), Nx = grid["Nx"].asUInt(),
             Ny = grid["Ny"].asUInt(), Nz = grid["Nz"].asUInt();
    //std::cout << "Type psi_0 (-20) and psi_1 (-4)\n";
    double psi_0 = grid["psi"][0].asDouble(), psi_1 = grid["psi"][1].asDouble();
    dg::Timer t;
    t.tic();
    //![doxygen]
    std::unique_ptr<dg::geo::aGenerator2d> generator;
    //create the magnetic field
    dg::geo::TokamakMagneticField mag = dg::geo::createMagneticField(
            js["magnetic_field"]["params"]);
    //create a grid generator
    std::string type = js["grid"]["type"].asString();
    std::cout << "Constructing "<<type<<" grid ... \n";
    if( type == "flux")
        generator = std::make_unique<dg::geo::FluxGenerator>( mag.get_psip(),
                mag.get_ipol(), psi_0, psi_1, mag.R0(), 0., 0, false);
    else if( type == "orthogonal")
        generator = std::make_unique<dg::geo::SimpleOrthogonal>( mag.get_psip(),
                psi_0, psi_1, mag.R0(), 0., 0);
    else if( type == "ribeiro-flux")
        generator = std::make_unique<dg::geo::RibeiroFluxGenerator>( mag.get_psip(),
                psi_0, psi_1, mag.R0(), 0., 0, false);
    else if( type == "ribeiro")
        generator = std::make_unique<dg::geo::Ribeiro>( mag.get_psip(),
                psi_0, psi_1, mag.R0(), 0., 0, false);
    else
    {
        std::cerr << "Error: Unknown grid type '"<<type<<"'. Exit now!\n";
        return -1;
    }
    //create a 3d and a 2d grid
    dg::geo::CurvilinearProductGrid3d g3d(*generator, n, Nx, Ny,Nz, dg::DIR);
    std::unique_ptr<dg::aGeometry2d> g2d( g3d.perp_grid());
    //![doxygen]
    //Find O-point
    double R_O = mag.R0(), Z_O = 0.;
    if( mag.params().getDescription() != dg::geo::description::none )
        dg::geo::findOpoint( mag.get_psip(), R_O, Z_O);
    std::cout << "O-point "<<R_O<<" "<<Z_O<<" with Psip = "<<mag.psip()(R_O,
            Z_O)<<std::endl;
    dg::Grid2d g2d_periodic(g2d->x0(), g2d->x1(), g2d->y0(), g2d->y1(), g2d->n(), g2d->Nx(), g2d->Ny()+1);
    t.toc();
    std::cout << "Construction successful!\n";
    std::cout << "Construction took "<<t.diff()<<"s"<<std::endl;
    //////////////////////////////setup and write netcdf//////////////////
    int ncid;
    dg::file::NC_Error_Handle err;
    std::string outfile = argc<2 ? "flux.nc" : argv[2];
    err = nc_create( outfile.c_str(), NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim2d[2];
    err = dg::file::define_dimensions(  ncid, dim2d, g2d_periodic);
    int coordsID[2];
    err = nc_def_var( ncid, "xc", NC_DOUBLE, 2, dim2d, &coordsID[0]);
    err = nc_def_var( ncid, "yc", NC_DOUBLE, 2, dim2d, &coordsID[1]);
    dg::HVec X=g2d->map()[0], Y=g2d->map()[1];
    err = nc_put_var_double( ncid, coordsID[0], periodify(X, g2d_periodic).data());
    err = nc_put_var_double( ncid, coordsID[1], periodify(Y, g2d_periodic).data());

    std::map< std::string, std::function< void( dg::HVec&, const
            dg::aGeometry2d&, const dg::geo::TokamakMagneticField&)>
        >
        output = {
        { "Psip", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psip(), g2d);
        }},
        { "PsipR", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psipR(), g2d);
        }},
        { "PsipZ", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psipZ(), g2d);
        }},
        { "g_xx", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result=g2d.metric().value(0,0);
        }},
        { "g_xy", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result=g2d.metric().value(0,1);
        }},
        { "g_yy", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result=g2d.metric().value(1,1);
        }},
        { "g_xy_g_xx", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            // deformation
            dg::blas1::pointwiseDivide( g2d.metric().value(0,1),
                g2d.metric().value(0,0), result);
        }},
        { "g_yy_g_xx", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            //ribeiro ratio
            dg::blas1::pointwiseDivide( g2d.metric().value(1,1),
                g2d.metric().value(0,0), result);
        }},
        { "vol", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result=dg::tensor::volume(g2d.metric());
        }},
        { "Bzeta", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result=Bzeta;
        }},
        { "Beta", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result=Beta;
        }},
        { "Bphi", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( dg::geo::BFieldP(mag), g2d);
        }},
        { "q-profile", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result = dg::pullback( dg::geo::BFieldP(mag), g2d);
            dg::blas1::pointwiseDivide( result, Beta, result); //Bphi / Beta

        }},
        { "Ipol", []( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.ipol(), g2d);
        }}
    };
    dg::HVec temp( g2d->size());
    for( auto pair : output)
    {
        int varID;
        err = nc_def_var( ncid, pair.first.data(), NC_DOUBLE, 2, dim2d, &varID);
        pair.second( temp, *g2d, mag);
        err = nc_put_var_double( ncid, varID, periodify(temp, g2d_periodic).data());
    }
    err = nc_close( ncid);
    ///////////////////////some further testing//////////////////////

    const dg::HVec vol3d = dg::create::volume( g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    double volume3d = dg::blas1::dot( vol3d, ones3d);
    const dg::HVec vol2d = dg::create::volume( *g2d);
    dg::HVec ones2d = dg::evaluate( dg::one, *g2d);
    double volume2d = dg::blas1::dot( vol2d, ones2d);

    std::cout << "TEST VOLUME IS:\n";
    double psipmin, psipmax;
    if( psi_0 < psi_1) psipmax = psi_1, psipmin = psi_0;
    else               psipmax = psi_0, psipmin = psi_1;
    auto iris = dg::compose( dg::Iris(psipmin, psipmax), mag.psip());
    dg::CartesianGrid2d g2dC( mag.R0() -2.0*mag.params().a(), mag.R0() +
            2.0*mag.params().a(), -2.0*mag.params().a(),2.0*mag.params().a(),
            1, 2e3, 2e3, dg::PER, dg::PER);
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
