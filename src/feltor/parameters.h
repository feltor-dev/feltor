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
    double alpha_mag, alpha, beta;
    double rho_source, rho_damping;

    double nu_perp, nu_parallel;
    double eta;

    double amp;
    double sigma;
    double posX, posY;
    double sigma_z;
    double k_psi;

    double omega_source, omega_damping;
    double nprofamp;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;

    enum dg::bc bcxN, bcyN, bcxU, bcyU, bcxP, bcyP;
    std::string initne, initphi, curvmode, perp_diff;
    bool symmetric;
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
        mx          = js["refineDS"].get( 0u, 10).asUInt();
        my          = js["refineDS"].get( 1u, 10).asUInt();
        rk4eps      = js.get( "rk4eps", 1e-5).asDouble();

        mu[0]       = js["mu"].asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = js["tau"].asDouble();
        beta        = js.get("beta",0.).asDouble();
        nu_perp     = js["nu_perp"].asDouble();
        perp_diff   = js.get("perp_diff", "viscous").asString();
        nu_parallel = js["nu_parallel"].asDouble();
        eta         = js["resistivity"].asDouble();

        amp         = js["amplitude"].asDouble();
        sigma       = js["sigma"].asDouble();
        posX        = js["posX"].asDouble();
        posY        = js["posY"].asDouble();
        sigma_z     = js["sigma_z"].asDouble();
        k_psi       = js["k_psi"].asDouble();

        nprofamp  = js["nprofileamp"].asDouble();
        omega_source  = js.get("source", 0.).asDouble();
        omega_damping = js.get("damping", 0.).asDouble();
        alpha_mag    = js.get("alpha_mag", 0.05).asDouble();
        alpha        = js.get("alpha", 0.2).asDouble();
        rho_source   = js.get("rho_source", 0.2).asDouble();
        rho_damping  = js.get("rho_damping", 1.2).asDouble();

        bcxN = dg::str2bc(js["bc"]["density"][0].asString());
        bcyN = dg::str2bc(js["bc"]["density"][1].asString());
        bcxU = dg::str2bc(js["bc"]["velocity"][0].asString());
        bcyU = dg::str2bc(js["bc"]["velocity"][1].asString());
        bcxP = dg::str2bc(js["bc"]["potential"][0].asString());
        bcyP = dg::str2bc(js["bc"]["potential"][1].asString());

        boxscaleRm  = js["boxscaleR"].get(0u,1.05).asDouble();
        boxscaleRp  = js["boxscaleR"].get(1u,1.05).asDouble();
        boxscaleZm  = js["boxscaleZ"].get(0u,1.05).asDouble();
        boxscaleZp  = js["boxscaleZ"].get(1u,1.05).asDouble();

        initne      = js.get( "initne", "blob").asString();
        initphi     = js.get( "initphi", "zero").asString();
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
            <<"     omega_source:                 "<<omega_source<<"\n"
            <<"     rho_source:                   "<<rho_source<<"\n"
            <<"     omega_damping:                "<<omega_damping<<"\n"
            <<"     rho_damping:                  "<<rho_damping<<"\n"
            <<"     alpha_mag:                    "<<alpha_mag<<"\n"
            <<"     alpha:                        "<<alpha<<"\n"
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
            <<"     Refined DS            "<<mx<<" "<<my<<"\n";
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
