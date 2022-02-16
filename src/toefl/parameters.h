#pragma once
#include <string>
#include "dg/algorithm.h"
#include "dg/file/json_utilities.h"
#include "json/json.h"

namespace toefl{

struct Parameters
{
    unsigned n, Nx, Ny;
    double lx, ly;
    dg::bc bcx, bcy;

    std::vector<double> eps_pol, eps_gamma;
    enum dg::direction pol_dir, diff_dir;
    unsigned num_stages;

    double amp, sigma, posX, posY;

    std::string model;
    double tau, kappa, friction, nu;
    bool boussinesq;
    Parameters() = default;

    Parameters( const dg::file::WrappedJsonValue& js) {
        n  = js["grid"]["n"].asUInt();
        Nx = js["grid"]["Nx"].asUInt();
        Ny = js["grid"]["Ny"].asUInt();
        lx = js["grid"]["lx"].asDouble();
        ly = js["grid"]["ly"].asDouble();

        num_stages = js["elliptic"]["stages"].asUInt();
        eps_pol.resize(num_stages);
        eps_gamma.resize(num_stages);
        eps_pol[0] = js["elliptic"]["eps_pol"][0].asDouble();
        eps_gamma[0] = js["elliptic"]["eps_gamma"][0].asDouble();
        for( unsigned u=1;u<num_stages; u++)
        {
            eps_pol[u] = js["elliptic"][ "eps_pol"][u].asDouble();
            eps_gamma[u] = js["elliptic"]["eps_gamma"][u].asDouble();
            eps_pol[u]*=eps_pol[0];
            eps_gamma[u]*=eps_gamma[0];
        }
        pol_dir =  dg::str2direction( js["elliptic"]["direction"].asString());
        diff_dir = dg::centered;

        amp = js["init"]["amplitude"].asDouble();
        sigma = js["init"]["sigma"].asDouble();
        posX = js["init"]["posX"].asDouble();
        posY = js["init"]["posY"].asDouble();
        bcx = dg::str2bc(js["bc"][0].asString());
        bcy = dg::str2bc(js["bc"][1].asString());
        model = js["model"].get("type", "global").asString();
        boussinesq = false;
        tau = friction = 0.;
        nu = js["model"]["nu"].asDouble();
        if( "local" == model)
        {
            kappa = js["model"]["curvature"].asDouble();
            tau = js["model"]["tau"].asDouble();
        }
        else if( "global" == model)
        {
            kappa = js["model"]["curvature"].asDouble();
            tau = js["model"]["tau"].asDouble();
            boussinesq = js["model"]["boussinesq"].asBool();
        }
        else if( "gravity-local" == model)
        {
            friction = js["model"]["friction"].asDouble();
        }
        else if( "gravity-global" == model)
        {
            friction = js["model"]["friction"].asDouble();
        }
        else if( "drift-global" == model)
        {
            boussinesq = js["model"]["boussinesq"].asBool();
            kappa = js["model"]["curvature"].asDouble();
        }
        else
            throw dg::Error( dg::Message(_ping_) << "Model : type `"<<model<<"` not recognized!\n");
    }

    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"    Friction:        = "<<friction<<"\n"
            <<"    Ion-temperature: = "<<tau<<"\n";
        os << "Equation parameters are: \n"
            <<"    "<<model<<"\n"
            <<"    boussinesq  "<<boussinesq<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n";
        os << "Boundary conditions in x are: \n"
            <<"    "<<bc2str(bcx)<<"\n";  //Curious! dg:: is not needed due to ADL!
        os << "Boundary conditions in y are: \n"
            <<"    "<<bc2str(bcy)<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n";
        os  <<"Blob parameters are: \n"
            << "    width:        "<<sigma<<"\n"
            << "    amplitude:    "<<amp<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Stopping for CG:         "<<eps_pol[0]<<"\n"
            <<"Stopping for Gamma CG:   "<<eps_gamma[0]<<std::endl;
    }
};
}//namespace toefl
