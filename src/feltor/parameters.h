#pragma once
#include <map>
#include <array>
#include <string>
#include "dg/enums.h"
#include "json/json.h"

namespace feltor{
struct Parameters
{
    unsigned n, Nx, Ny, Nz;
    unsigned n_out, Nx_out, Ny_out, Nz_out;
    double dt;
    unsigned cx, cy;
    unsigned itstp;
    unsigned maxout;

    double eps_pol;
    double jfactor;
    double eps_gamma;
    double eps_time;
    double rtol;
    double eps_hat;
    unsigned stages;
    unsigned mx, my;

    std::array<double,2> mu; // mu[0] = mu_e, m[1] = mu_i
    std::array<double,2> tau; // tau[0] = -1, tau[1] = tau_i

    double nu_perp, nu_parallel;
    double c;

    double amp;
    double sigma;
    double posX, posY;
    double sigma_z;
    double k_psi;

    double omega_source;
    double nprofileamp;
    double bgprofamp;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;

    enum dg::bc bcxN, bcyN, bcxU, bcyU, bcxP, bcyP;
    std::string initni, initphi, curvmode;
    Parameters( const Json::Value& js) {
        n       = js["n"].asUInt();
        Nx      = js["Nx"].asUInt();
        Ny      = js["Ny"].asUInt();
        Nz      = js["Nz"].asUInt();
        dt      = js["dt"].asDouble();
        cx      = js.get("cx",1).asUInt();
        cy      = js.get("cy",1).asUInt();
        n_out = n, Nx_out = Nx/cx, Ny_out = Ny/cy, Nz_out = Nz;
        itstp   = js["itstp"].asUInt();
        maxout  = js["maxout"].asUInt();

        eps_pol     = js["eps_pol"].asDouble();
        jfactor     = js.get("jumpfactor",1).asDouble();
        eps_gamma   = js["eps_gamma"].asDouble();
        eps_time    = js["eps_time"].asDouble();
        rtol        = js["rtol"].asDouble();
        eps_hat     = 1.;
        stages      = js.get( "stages", 3).asUInt();
        mx          = js.get( "multiplyX", 10).asUInt();
        my          = js.get( "multiplyY", 10).asUInt();

        mu[0]       = js["mu"].asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = js["tau"].asDouble();
        nu_perp     = js["nu_perp"].asDouble();
        nu_parallel = js["nu_parallel"].asDouble();
        c           = js["resistivity"].asDouble();

        amp         = js["amplitude"].asDouble();
        sigma       = js["sigma"].asDouble();
        posX        = js["posX"].asDouble();
        posY        = js["posY"].asDouble();
        sigma_z     = js["sigma_z"].asDouble();
        k_psi       = js["k_psi"].asDouble();

        nprofileamp  = js["nprofileamp"].asDouble();
        bgprofamp    = js["bgprofamp"].asDouble();
        omega_source = js["source"].asDouble();

        bcxN = dg::str2bc(js["bc"]["density"][0].asString());
        bcyN = dg::str2bc(js["bc"]["density"][1].asString());
        bcxU = dg::str2bc(js["bc"]["velocity"][0].asString());
        bcyU = dg::str2bc(js["bc"]["velocity"][1].asString());
        bcxP = dg::str2bc(js["bc"]["potential"][0].asString());
        bcyP = dg::str2bc(js["bc"]["potential"][1].asString());

        boxscaleRp  = js.get("boxscaleRp",1.05).asDouble();
        boxscaleRm  = js.get("boxscaleRm",1.05).asDouble();
        boxscaleZp  = js.get("boxscaleZp",1.05).asDouble();
        boxscaleZm  = js.get("boxscaleZm",1.05).asDouble();

        initni      = js.get( "initni", "blob").asString();
        initphi     = js.get( "initphi", "zero").asString();
        curvmode    = js.get( "curvmode", "toroidal").asString();
    }
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"     mu_e              = "<<mu[0]<<"\n"
            <<"     mu_i              = "<<mu[1]<<"\n"
            <<"     El.-temperature:  = "<<tau[0]<<"\n"
            <<"     Ion-temperature:  = "<<tau[1]<<"\n"
            <<"     perp. Viscosity:  = "<<nu_perp<<"\n"
            <<"     par. Resistivity: = "<<c<<"\n"
            <<"     par. Viscosity:   = "<<nu_parallel<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n"
            << "    sigma_z:      "<<sigma_z<<"\n";
        os << "Profile parameters are: \n"
            <<"     omega_source:                 "<<omega_source<<"\n"
            <<"     density profile amplitude:    "<<nprofileamp<<"\n"
            <<"     background profile amplitude: "<<bgprofamp<<"\n"
            <<"     boxscale R+:                  "<<boxscaleRp<<"\n"
            <<"     boxscale R-:                  "<<boxscaleRm<<"\n"
            <<"     boxscale Z+:                  "<<boxscaleZp<<"\n"
            <<"     boxscale Z-:                  "<<boxscaleZm<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     Nz = "<<Nz<<"\n"
            <<"     dt = "<<dt<<"\n";
        os << "     Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"     Jump scale factor:   "<<jfactor<<"\n"
            <<"     Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"     Stopping for Time  CG:   "<<eps_time<<"\n";
        os << "Output parameters are: \n"
            <<"     n_out  =              "<<n_out<<"\n"
            <<"     Nx_out =              "<<Nx_out<<"\n"
            <<"     Ny_out =              "<<Ny_out<<"\n"
            <<"     Nz_out =              "<<Nz_out<<"\n"
            <<"     Steps between output: "<<itstp<<"\n"
            <<"     Number of outputs:    "<<maxout<<"\n";
        os << "Boundary condition is: \n"
            <<"     bc density x   = "<<dg::bc2str(bcxN)<<"\n"
            <<"     bc density y   = "<<dg::bc2str(bcyN)<<"\n"
            <<"     bc velocity x  = "<<dg::bc2str(bcxU)<<"\n"
            <<"     bc velocity y  = "<<dg::bc2str(bcyU)<<"\n"
            <<"     bc potential x = "<<dg::bc2str(bcxP)<<"\n"
            <<"     bc potential y = "<<dg::bc2str(bcyP)<<"\n"
            <<"     init N_i       = "<<initni<<"\n"
            <<"     init Phi       = "<<initphi<<"\n"
            <<"     curvature mode = "<<curvmode<<"\n";
        os << std::flush;
    }
};

}//namespace feltor
