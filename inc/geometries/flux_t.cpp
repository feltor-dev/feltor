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
#include "separatrix_orthogonal.h"
#include "ds_generator.h"
#include "ribeiro.h"
#include "hector.h"


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
    std::string input = argc==1 ? "flux.json" : argv[1];
    dg::file::WrappedJsonValue js = dg::file::file2Json(input);

    std::string geometry_params = js["magnetic_field"]["input"].asString();
    if( geometry_params == "file")
    {
        std::string path = js["magnetic_field"]["file"].asString();
        js.asJson()["magnetic_field"]["params"] = dg::file::file2Json( path);
    }
    else if( geometry_params != "params")
    {
        std::cerr << "Error: Unknown magnetic field input '"<<geometry_params<<"'. Exit now!\n";
        return -1;
    }
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
    std::string type = js["grid"]["generator"]["type"].asString();
    int mode = 0;
    if( type != "dsp")
        mode = js["grid"]["generator"]["mode"].asInt();
    std::cout << "Constructing "<<type<<" grid ... \n";
    if( type == "flux")
        generator = std::make_unique<dg::geo::FluxGenerator>( mag.get_psip(),
                mag.get_ipol(), psi_0, psi_1, mag.R0(), 0., mode, false);
    else if( type == "orthogonal")
    {
        double psi_init = js["grid"]["generator"]["firstline"].asDouble();
        if( mode == 0 || mode == 1)
            generator = std::make_unique<dg::geo::SimpleOrthogonal>(
                mag.get_psip(), psi_0, psi_1, mag.R0(), 0., psi_init, mode);
        if( mode > 1)
        {
            dg::geo::CylindricalSymmTensorLvl1 lc =
                dg::geo::make_LiseikinCollective( mag.get_psip(), 0.1, 0.001);
            generator = std::make_unique<dg::geo::SimpleOrthogonal>(
                mag.get_psip(), lc, psi_0, psi_1, mag.R0(), 0., psi_init,
                mode%2);
        }
    }
    else if( type == "separatrix-orthogonal")
    {
        double RX = mag.R0()-1.1*mag.params().triangularity()*mag.params().a();
        double ZX = -1.1*mag.params().elongation()*mag.params().a();
        dg::geo::findXpoint( mag.get_psip(), RX, ZX);
        //dg::geo::CylindricalSymmTensorLvl1 monitor_chi;
        dg::geo::CylindricalSymmTensorLvl1 monitor_chi = dg::geo::make_Xconst_monitor( mag.get_psip(), RX, ZX) ;
        double fx = js["grid"]["generator"]["fx"].asDouble();
        generator = std::make_unique<dg::geo::SeparatrixOrthogonalAdaptor>(
            mag.get_psip(), monitor_chi, psi_0, RX, ZX, mag.R0(), 0., mode, false, fx);
        //psi_1 = -fx/(1.-fx)*psi_0;
    }
    else if ( type == "dsp")
    {
        double boxscaleRm  = js["grid"][ "scaleR"].get( 0u, 1.05).asDouble();
        double boxscaleRp  = js["grid"][ "scaleR"].get( 1u, 1.05).asDouble();
        double boxscaleZm  = js["grid"][ "scaleZ"].get( 0u, 1.05).asDouble();
        double boxscaleZp  = js["grid"][ "scaleZ"].get( 1u, 1.05).asDouble();
        const double Rmin=mag.R0()-boxscaleRm*mag.params().a();
        const double Zmin=-boxscaleZm*mag.params().a();
        const double Rmax=mag.R0()+boxscaleRp*mag.params().a();
        const double Zmax=boxscaleZp*mag.params().a();
        generator = std::make_unique<dg::geo::DSPGenerator>( mag,
            Rmin, Rmax, Zmin, Zmax, 2.*M_PI/(double)Nz);
    }
    else if( type == "ribeiro-flux")
        generator = std::make_unique<dg::geo::RibeiroFluxGenerator>( mag.get_psip(),
                psi_0, psi_1, mag.R0(), 0., mode, false);
    else if( type == "ribeiro")
        generator = std::make_unique<dg::geo::Ribeiro>( mag.get_psip(),
                psi_0, psi_1, mag.R0(), 0., mode, false);
    //![doxygen]
    else if( type == "hector")
    {
        //![hector]
        unsigned nGrid = js["grid"]["generator"]["initial"]["n"].asUInt();
        unsigned NxGrid = js["grid"]["generator"]["initial"]["Nx"].asUInt();
        unsigned NyGrid = js["grid"]["generator"]["initial"]["Ny"].asUInt();
        double epsHector = js["grid"]["generator"]["eps"].asDouble();
        if( mode == 0)
        {
            std::cout << " ... of type conformal ...\n";
            generator = std::make_unique< dg::geo::Hector<dg::IDMatrix,
                      dg::DMatrix, dg::DVec>>( mag.get_psip(), psi_0, psi_1,
                              mag.R0(), 0., nGrid, NxGrid, NyGrid, epsHector,
                              true);
        }
        else if( mode == 1)
        {
            std::cout << " ... of type adaption ...\n";
            dg::geo::CylindricalFunctorsLvl1 nc =
                dg::geo::make_NablaPsiInvCollective( mag.get_psip());
            generator = std::make_unique< dg::geo::Hector<dg::IDMatrix,
                      dg::DMatrix, dg::DVec>>( mag.get_psip(), nc, psi_0,
                              psi_1, mag.R0(), 0., nGrid, NxGrid, NyGrid,
                              epsHector, true);
        }
        else
        {
            std::cout << " ... of type monitor metric ...\n";
            dg::geo::CylindricalSymmTensorLvl1 lc =
                dg::geo::make_LiseikinCollective( mag.get_psip(), 0.1, 0.001);
            generator = std::make_unique< dg::geo::Hector<dg::IDMatrix,
                      dg::DMatrix, dg::DVec>>( mag.get_psip(), lc, psi_0,
                              psi_1, mag.R0(), 0., nGrid, NxGrid, NyGrid,
                              epsHector, true);
        }
        //![hector]
    }
    else
    {
        std::cerr << "Error: Unknown grid type '"<<type<<"'. Exit now!\n";
        return -1;
    }
    //create a 3d and a 2d grid
    dg::geo::CurvilinearProductGrid3d g3d(*generator, n, Nx, Ny,Nz, dg::DIR);
    std::unique_ptr<dg::aGeometry2d> g2d( g3d.perp_grid());
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
    dg::file::NcFile file( argc < 3 ?"flux.nc" : argv[2], dg::file::nc_clobber);
    file.defput_dim( "x", {{"axis", "X"}}, g2d_periodic.abscissas(0));
    file.defput_dim( "y", {{"axis", "Y"}}, g2d_periodic.abscissas(1));

    dg::SparseTensor<dg::HVec > metric = g2d->metric();
    std::map< std::string, std::function< void( dg::HVec&, const
            dg::aGeometry2d&, const dg::geo::TokamakMagneticField&)>
        >
        output = {
        { "xc", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField&){
            result = g2d.map()[0];
        }},
        { "yc", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField&){
            result = g2d.map()[1];
        }},
        { "Psip", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psip(), g2d);
        }},
        { "PsipR", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psipR(), g2d);
        }},
        { "PsipZ", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.psipZ(), g2d);
        }},
        { "g_xx", [&]( dg::HVec& result, const dg::aGeometry2d&,
            const dg::geo::TokamakMagneticField&){
            result=metric.value(0,0);
        }},
        { "g_xy", [&]( dg::HVec& result, const dg::aGeometry2d&,
            const dg::geo::TokamakMagneticField&){
            result=metric.value(0,1);
        }},
        { "g_yy", [&]( dg::HVec& result, const dg::aGeometry2d&,
            const dg::geo::TokamakMagneticField&){
            result=metric.value(1,1);
        }},
        { "g_xy_g_xx", [&]( dg::HVec& result, const dg::aGeometry2d&,
            const dg::geo::TokamakMagneticField&){
            // deformation
            dg::blas1::pointwiseDivide( metric.value(0,1),
                metric.value(0,0), result);
        }},
        { "g_yy_g_xx", [&]( dg::HVec& result, const dg::aGeometry2d&,
            const dg::geo::TokamakMagneticField&){
            //ribeiro ratio
            dg::blas1::pointwiseDivide( metric.value(1,1),
                metric.value(0,0), result);
        }},
        { "vol", [&]( dg::HVec& result, const dg::aGeometry2d&,
            const dg::geo::TokamakMagneticField&){
            result=dg::tensor::volume(metric);
        }},
        { "Bzeta", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result=Bzeta;
        }},
        { "Beta", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result=Beta;
        }},
        { "Bphi", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( dg::geo::BFieldP(mag), g2d);
        }},
        { "q-profile", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            dg::HVec Bzeta, Beta;
            dg::pushForwardPerp( dg::geo::BFieldR(mag), dg::geo::BFieldZ(mag), Bzeta, Beta, g2d);
            result = dg::pullback( dg::geo::BFieldP(mag), g2d);
            dg::blas1::pointwiseDivide( result, Beta, result); //Bphi / Beta

        }},
        { "Ipol", [&]( dg::HVec& result, const dg::aGeometry2d& g2d,
            const dg::geo::TokamakMagneticField& mag){
            result = dg::pullback( mag.ipol(), g2d);
        }}
    };
    dg::HVec temp( g2d->size());
    for( auto pair : output)
    {
        pair.second( temp, *g2d, mag);
        file.def_var_as<double>( pair.first, {"y", "x"});
        file.put_var( pair.first, {g2d_periodic}, periodify( temp, g2d_periodic));
    }
    file.close();
    ///////////////////////some further testing//////////////////////

    const dg::HVec vol3d = dg::create::volume( g3d);
    dg::HVec ones3d = dg::evaluate( dg::one, g3d);
    double volume3d = dg::blas1::dot( vol3d, ones3d);
    const dg::HVec vol2d = dg::create::volume( *g2d);
    double volume2d = dg::blas1::dot( vol2d, 1.0);
    //
    if( type == "ribeiro")
    {
        dg::HVec err = dg::tensor::volume2d( metric);
        dg::blas1::pointwiseDivide( 1., 1., metric.value(0,0), -1., err);
        double error = dg::blas2::dot( err, vol2d, err);
        std::cout << "Error 1/g_xx - vol: "<<error<< "\n";
    }
    if( type == "orthogonal")
    {
        //basically just tests if g_xy is really 0
        dg::HVec err = dg::tensor::volume2d( metric);
        dg::blas1::evaluate( err, dg::minus_equals(),
                []DG_DEVICE(double g_xx, double g_yy){ return 1./sqrt(g_xx*g_yy); },
                metric.value(0,0), metric.value(1,1));
        double error = dg::blas2::dot( err, vol2d, err);
        std::cout << "Error 1/sqrt( g_xx g_yy) - vol: "<<error<< "\n";
    }

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
    if ( type == "hector")
    {
        dg::HVec volume = dg::create::volume(
                dynamic_cast<dg::geo::Hector<dg::IDMatrix, dg::DMatrix,
                dg::DVec>*>( generator.get())->internal_grid());
        double volumeZE = dg::blas1::dot( volume, 1.);
        std::cout << "relative difference in volumeZE is "<<fabs(volumeZE - volume2d)/volume2d<<std::endl;
    }
    std::cout << "Note that the error might also come from the volume in RZP!\n";
    //since integration of jacobian is fairly good probably

    return 0;
}
