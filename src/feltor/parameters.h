#pragma once
#include <map>
#include <array>
#include <string>
#include "dg/enums.h"
#include "json/json.h"
#include "dg/file/json_utilities.h"

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

    std::vector<double> eps_pol;
    double jfactor;
    double eps_gamma;
    double eps_time;
    double rtol;
    unsigned stages;
    unsigned mx, my;
    double rk4eps;

    std::array<double,2> mu; // mu[0] = mu_e, m[1] = mu_i
    std::array<double,2> tau; // tau[0] = -1, tau[1] = tau_i
    std::array<double,2> nu_parallel;

    double nu_perp;
    double eta, beta;

    double amp;
    double sigma;
    double posX, posY;
    double sigma_z;
    double k_psi;

    double source_rate, damping_rate;
    double source_alpha, profile_alpha;
    double source_boundary;
    double nprofamp;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;

    enum dg::bc bcxN, bcyN, bcxU, bcyU, bcxP, bcyP;
    std::string initne, initphi, curvmode, perp_diff;
    std::string source_type;
    bool symmetric, periodify;
    Parameters() = default;
    Parameters( const Json::Value& js, enum file::error mode = file::error::is_warning ) {
        //We need to check if a member is present
        n       = file::get(mode, js,"n", 3).asUInt();
        Nx      = file::get(mode, js,"Nx", 0).asUInt();
        Ny      = file::get(mode, js,"Ny", 0).asUInt();
        Nz      = file::get(mode, js,"Nz", 0).asUInt();
        dt      = file::get(mode, js,"dt", 0.).asDouble();
        cx      = file::get_idx(mode, js,"compression",0u,1).asUInt();
        cy      = file::get_idx(mode, js,"compression",1u,1).asUInt();
        n_out = n, Nx_out = Nx/cx, Ny_out = Ny/cy, Nz_out = Nz;
        inner_loop = file::get(mode, js, "inner_loop",1).asUInt();
        itstp   = file::get( mode, js, "itstp", 0).asUInt();
        maxout  = file::get( mode, js, "maxout", 0).asUInt();
        eps_time    = file::get( mode, js, "eps_time", 1e-10).asDouble();
        rtol        = file::get( mode, js, "rtol", 1e-5).asDouble();

        stages      = file::get( mode, js, "stages", 3).asUInt();
        eps_pol.resize(stages);
        eps_pol[0] = file::get_idx( mode, js, "eps_pol", 0, 1e-6).asDouble();
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = file::get_idx( mode, js, "eps_pol", i, 1).asDouble();
            eps_pol[i]*=eps_pol[0];
        }
        jfactor     = file::get( mode, js, "jumpfactor", 1).asDouble();

        eps_gamma   = file::get( mode, js, "eps_gamma", 1e-6).asDouble();
        mx          = file::get_idx( mode, js,"FCI","refine", 0u, 1).asUInt();
        my          = file::get_idx( mode, js,"FCI","refine", 1u, 1).asUInt();
        rk4eps      = file::get( mode, js,"FCI", "rk4eps", 1e-6).asDouble();
        periodify   = file::get( mode, js,"FCI", "periodify", true).asBool();

        mu[0]       = file::get( mode, js, "mu", -0.000272121).asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = file::get( mode, js, "tau", 0.).asDouble();
        beta        = file::get( mode, js, "beta", 0.).asDouble();
        eta         = file::get( mode, js, "resistivity", 0.).asDouble();
        nu_perp     = file::get( mode, js, "nu_perp", 0.).asDouble();
        perp_diff   = file::get( mode, js, "perp_diff", "viscous").asString();
        //nu_parallel = file::get( mode, js, "nu_parallel", 0.).asDouble();
        //Init after reading in eta and mu[0]
        nu_parallel[0] = 0.73/eta;
        nu_parallel[1] = sqrt(fabs(mu[0]))*1.36/eta;

        initne      = file::get( mode, js, "initne", "blob").asString();
        initphi     = file::get( mode, js, "initphi", "zero").asString();
        amp         = file::get( mode, js, "amplitude", 0.).asDouble();
        sigma       = file::get( mode, js, "sigma", 0.).asDouble();
        posX        = file::get( mode, js, "posX", 0.).asDouble();
        posY        = file::get( mode, js, "posY", 0.).asDouble();
        sigma_z     = file::get( mode, js, "sigma_z", 0.).asDouble();
        k_psi       = file::get( mode, js, "k_psi", 0.).asDouble();

        nprofamp   = file::get( mode, js, "profile", "amp", 0.).asDouble();
        profile_alpha = file::get( mode, js, "profile", "alpha", 0.2).asDouble();

        source_rate     = file::get( mode, js, "source", "rate", 0.).asDouble();
        source_type     = file::get( mode, js, "source", "type", "profile").asString();
        source_boundary = file::get( mode, js, "source", "boundary", 0.5).asDouble();
        source_alpha    = file::get( mode, js, "source", "alpha", 0.2).asDouble();
        damping_rate = file::get( mode, js, "damping", "rate", 0.).asDouble();

        bcxN = dg::str2bc(file::get_idx( mode, js, "bc", "density", 0, "").asString());
        bcyN = dg::str2bc(file::get_idx( mode, js, "bc", "density", 1, "").asString());
        bcxU = dg::str2bc(file::get_idx( mode, js, "bc", "velocity", 0, "").asString());
        bcyU = dg::str2bc(file::get_idx( mode, js, "bc", "velocity", 1, "").asString());
        bcxP = dg::str2bc(file::get_idx( mode, js, "bc", "potential", 0, "").asString());
        bcyP = dg::str2bc(file::get_idx( mode, js, "bc", "potential", 1, "").asString());

        boxscaleRm  = file::get_idx( mode, js, "box", "scaleR", 0u, 1.05).asDouble();
        boxscaleRp  = file::get_idx( mode, js, "box", "scaleR", 1u, 1.05).asDouble();
        boxscaleZm  = file::get_idx( mode, js, "box", "scaleZ", 0u, 1.05).asDouble();
        boxscaleZp  = file::get_idx( mode, js, "box", "scaleZ", 1u, 1.05).asDouble();

        curvmode    = file::get( mode, js, "curvmode", "toroidal").asString();
        symmetric   = file::get( mode, js, "symmetric", false).asBool();
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
            <<"     par. Viscosity e  = "<<nu_parallel[0]<<"\n"
            <<"     par. Viscosity i  = "<<nu_parallel[1]<<"\n"
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
            <<"     Accuracy Polar CG:    "<<eps_pol[0]<<"\n"
            <<"     Jump scale factor:    "<<jfactor<<"\n"
            <<"     Accuracy Gamma CG:    "<<eps_gamma<<"\n"
            <<"     Accuracy Time  CG:    "<<eps_time<<"\n"
            <<"     Accuracy Time Stepper "<<rtol<<"\n"
            <<"     Accuracy Fieldline    "<<rk4eps<<"\n"
            <<"     Periodify FCI         "<<std::boolalpha<< periodify<<"\n"
            <<"     Refined FCI           "<<mx<<" "<<my<<"\n";
        for( unsigned i=1; i<stages; i++)
            os <<"     Factors for Multigrid "<<i<<" "<<eps_pol[i]<<"\n";
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
