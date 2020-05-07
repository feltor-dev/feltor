#pragma once
#include <map>
#include <array>
#include <string>
#include "dg/enums.h"
#include "json/json.h"

namespace feltor{
/// If you need more parameters, just go ahead and extend the list
struct Parameters
{
    unsigned n, Nx, Ny, Nz;
    unsigned n_out, Nx_out, Ny_out, Nz_out;
    double dt;
    unsigned cx, cy;
    unsigned inner_loop;
    unsigned itstp;
    unsigned maxout;

    double eps_pol;
    double jfactor;
    double eps_gamma;
    double eps_time;
    double rtol;
    unsigned stages;
    unsigned mx, my;
    double rk4eps;

    std::array<double,2> mu; // mu[0] = mu_e, m[1] = mu_i
    std::array<double,2> tau; // tau[0] = -1, tau[1] = tau_i

    double nu_perp, nu_parallel;
    double eta, beta;

    double amp;
    double sigma;
    double posX, posY;
    double sigma_z;
    double k_psi;

    double source_rate, damping_rate;
    double damping_alpha, source_alpha, profile_alpha;
    double source_boundary, damping_boundary;
    double nprofamp;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;

    enum dg::bc bcxN, bcyN, bcxU, bcyU, bcxP, bcyP;
    std::string initne, initphi, curvmode, perp_diff;
    std::string source_type;
    bool symmetric, periodify;
    Parameters() = default;
    Parameters( const Json::Value& js) {
        n       = js["n"].asUInt();
        Nx      = js["Nx"].asUInt();
        Ny      = js["Ny"].asUInt();
        Nz      = js["Nz"].asUInt();
        dt      = js["dt"].asDouble();
        cx      = js["compression"].get(0u,1).asUInt();
        cy      = js["compression"].get(1u,1).asUInt();
        n_out = n, Nx_out = Nx/cx, Ny_out = Ny/cy, Nz_out = Nz;
        inner_loop = js.get("inner_loop",1).asUInt();
        itstp   = js["itstp"].asUInt();
        maxout  = js["maxout"].asUInt();
        eps_time    = js["eps_time"].asDouble();
        rtol        = js["rtol"].asDouble();

        eps_pol     = js["eps_pol"].asDouble();
        jfactor     = js.get("jumpfactor",1).asDouble();

        eps_gamma   = js["eps_gamma"].asDouble();
        stages      = js.get( "stages", 3).asUInt();
        mx          = js["FCI"]["refine"].get( 0u, 1).asUInt();
        my          = js["FCI"]["refine"].get( 1u, 1).asUInt();
        rk4eps      = js["FCI"].get( "rk4eps", 1e-6).asDouble();
        periodify   = js["FCI"].get( "periodify", true).asBool();

        mu[0]       = js["mu"].asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = js["tau"].asDouble();
        beta        = js.get("beta",0.).asDouble();
        nu_perp     = js["nu_perp"].asDouble();
        perp_diff   = js.get("perp_diff", "viscous").asString();
        nu_parallel = js["nu_parallel"].asDouble();
        eta         = js["resistivity"].asDouble();

        initne      = js.get( "initne", "blob").asString();
        initphi     = js.get( "initphi", "zero").asString();
        amp         = js["amp"].asDouble();
        sigma       = js["sigma"].asDouble();
        posX        = js["posX"].asDouble();
        posY        = js["posY"].asDouble();
        sigma_z     = js["sigma_z"].asDouble();
        k_psi       = js["k_psi"].asDouble();

        nprofamp   = js["profile"]["amp"].asDouble();
        profile_alpha = js["profile"]["alpha"].asDouble();

        source_rate     = js["source"].get("rate", 0.).asDouble();
        source_type     = js["source"].get("type", "profile").asString();
        source_boundary = js["source"].get("boundary", 0.2).asDouble();
        source_alpha    = js["source"].get("alpha", 0.2).asDouble();
        damping_rate = js["damping"].get("rate", 0.).asDouble();
        damping_alpha= js["damping"].get("alpha", 0.05).asDouble();
        damping_boundary = js["damping"].get("boundary", 1.2).asDouble();

        bcxN = dg::str2bc(js["bc"]["density"][0].asString());
        bcyN = dg::str2bc(js["bc"]["density"][1].asString());
        bcxU = dg::str2bc(js["bc"]["velocity"][0].asString());
        bcyU = dg::str2bc(js["bc"]["velocity"][1].asString());
        bcxP = dg::str2bc(js["bc"]["potential"][0].asString());
        bcyP = dg::str2bc(js["bc"]["potential"][1].asString());

        boxscaleRm  = js["box"]["scaleR"].get(0u,1.05).asDouble();
        boxscaleRp  = js["box"]["scaleR"].get(1u,1.05).asDouble();
        boxscaleZm  = js["box"]["scaleZ"].get(0u,1.05).asDouble();
        boxscaleZp  = js["box"]["scaleZ"].get(1u,1.05).asDouble();

        curvmode    = js.get( "curvmode", "toroidal").asString();
        symmetric   = js.get( "symmetric", false).asBool();
    }
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"     mu_e              = "<<mu[0]<<"\n"
            <<"     mu_i              = "<<mu[1]<<"\n"
            <<"     El.-temperature   = "<<tau[0]<<"\n"
            <<"     Ion-temperature   = "<<tau[1]<<"\n"
            <<"     perp. Viscosity   = "<<nu_perp<<"\n"
            <<"     perp. Viscosity   = "<<perp_diff<<"\n"
            <<"     par. Resistivity  = "<<eta<<"\n"
            <<"     beta              = "<<beta<<"\n"
            <<"     par. Viscosity    = "<<nu_parallel<<"\n"
            <<"     curvature mode    = "<<curvmode<<"\n"
            <<"     Symmetry in phi   = "<<std::boolalpha<<symmetric<<"\n";
        os  <<"Initial parameters are: \n"
            <<"    amplitude:    "<<amp<<"\n"
            <<"    width:        "<<sigma<<"\n"
            <<"    posX:         "<<posX<<"\n"
            <<"    posY:         "<<posY<<"\n"
            <<"    sigma_z:      "<<sigma_z<<"\n"
            <<"    k_psi:        "<<k_psi<<"\n"
            <<"    init n_e:     "<<initne<<"\n"
            <<"    init Phi:     "<<initphi<<"\n";
        os << "Profile parameters are: \n"
            <<"     source_rate:                  "<<source_rate<<"\n"
            <<"     source_boundary:              "<<source_boundary<<"\n"
            <<"     source_alpha:                 "<<source_alpha<<"\n"
            <<"     source_type:                  "<<source_type<<"\n"
            <<"     damping_rate:                 "<<damping_rate<<"\n"
            <<"     damping_boundary:             "<<damping_boundary<<"\n"
            <<"     damping_alpha:                "<<damping_alpha<<"\n"
            <<"     density profile amplitude:    "<<nprofamp<<"\n"
            <<"     boxscale R+:                  "<<boxscaleRp<<"\n"
            <<"     boxscale R-:                  "<<boxscaleRm<<"\n"
            <<"     boxscale Z+:                  "<<boxscaleZp<<"\n"
            <<"     boxscale Z-:                  "<<boxscaleZm<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     Nz = "<<Nz<<"\n"
            <<"     dt = "<<dt<<"\n"
            <<"     Accuracy Polar CG:    "<<eps_pol<<"\n"
            <<"     Jump scale factor:    "<<jfactor<<"\n"
            <<"     Accuracy Gamma CG:    "<<eps_gamma<<"\n"
            <<"     Accuracy Time  CG:    "<<eps_time<<"\n"
            <<"     Accuracy Time Stepper "<<rtol<<"\n"
            <<"     Accuracy Fieldline    "<<rk4eps<<"\n"
            <<"     Periodify FCI         "<<std::boolalpha<< periodify<<"\n"
            <<"     Refined FCI           "<<mx<<" "<<my<<"\n";
        os << "Output parameters are: \n"
            <<"     n_out  =                 "<<n_out<<"\n"
            <<"     Nx_out =                 "<<Nx_out<<"\n"
            <<"     Ny_out =                 "<<Ny_out<<"\n"
            <<"     Nz_out =                 "<<Nz_out<<"\n"
            <<"     Steps between energies:  "<<inner_loop<<"\n"
            <<"     Energies between output: "<<itstp<<"\n"
            <<"     Number of outputs:       "<<maxout<<"\n";
        os << "Boundary conditions are: \n"
            <<"     bc density x   = "<<dg::bc2str(bcxN)<<"\n"
            <<"     bc density y   = "<<dg::bc2str(bcyN)<<"\n"
            <<"     bc velocity x  = "<<dg::bc2str(bcxU)<<"\n"
            <<"     bc velocity y  = "<<dg::bc2str(bcyU)<<"\n"
            <<"     bc potential x = "<<dg::bc2str(bcxP)<<"\n"
            <<"     bc potential y = "<<dg::bc2str(bcyP)<<"\n";
        os << std::flush;
    }
};

}//namespace feltor
