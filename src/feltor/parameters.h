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

    double source_rate, wall_rate, sheath_rate;
    double source_alpha, profile_alpha;
    double source_boundary;
    double nprofamp;
    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;

    enum dg::bc bcxN, bcyN, bcxU, bcyU, bcxP, bcyP;
    std::string initne, initphi, curvmode, perp_diff;
    std::string source_type, sheath_bc;
    bool symmetric, periodify, explicit_diffusion ;
    Parameters() = default;
    Parameters( const Json::Value& js, enum dg::file::error mode = dg::file::error::is_warning ) {
        //We need to check if a member is present
        n       = dg::file::get(mode, js,"n", 3).asUInt();
        Nx      = dg::file::get(mode, js,"Nx", 0).asUInt();
        Ny      = dg::file::get(mode, js,"Ny", 0).asUInt();
        Nz      = dg::file::get(mode, js,"Nz", 0).asUInt();
        dt      = dg::file::get(mode, js,"dt", 0.).asDouble();
        cx      = dg::file::get_idx(mode, js,"compression",0u,1).asUInt();
        cy      = dg::file::get_idx(mode, js,"compression",1u,1).asUInt();
        n_out = n, Nx_out = Nx/cx, Ny_out = Ny/cy, Nz_out = Nz;
        inner_loop = dg::file::get(mode, js, "inner_loop",1).asUInt();
        itstp   = dg::file::get( mode, js, "itstp", 0).asUInt();
        maxout  = dg::file::get( mode, js, "maxout", 0).asUInt();
        eps_time    = dg::file::get( mode, js, "eps_time", 1e-10).asDouble();

        stages      = dg::file::get( mode, js, "stages", 3).asUInt();
        eps_pol.resize(stages);
        eps_pol[0] = dg::file::get_idx( mode, js, "eps_pol", 0, 1e-6).asDouble();
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = dg::file::get_idx( mode, js, "eps_pol", i, 1).asDouble();
            eps_pol[i]*=eps_pol[0];
        }
        jfactor     = dg::file::get( mode, js, "jumpfactor", 1).asDouble();

        eps_gamma   = dg::file::get( mode, js, "eps_gamma", 1e-6).asDouble();
        mx          = dg::file::get_idx( mode, js,"FCI","refine", 0u, 1).asUInt();
        my          = dg::file::get_idx( mode, js,"FCI","refine", 1u, 1).asUInt();
        rk4eps      = dg::file::get( mode, js,"FCI", "rk4eps", 1e-6).asDouble();
        periodify   = dg::file::get( mode, js,"FCI", "periodify", true).asBool();

        mu[0]       = dg::file::get( mode, js, "mu", -0.000272121).asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = dg::file::get( mode, js, "tau", 0.).asDouble();
        beta        = dg::file::get( mode, js, "beta", 0.).asDouble();
        eta         = dg::file::get( mode, js, "resistivity", 0.).asDouble();
        nu_perp     = dg::file::get( mode, js, "nu_perp", 0.).asDouble();
        perp_diff   = dg::file::get_idx( mode, js, "perp_diff", 0, "viscous").asString();
        std::string temp = dg::file::get_idx( mode, js, "perp_diff", 1, "").asString();
        explicit_diffusion = true;
        if(temp == "implicit")
            explicit_diffusion = false;
        else if(temp == "explicit")
            explicit_diffusion = true;
        else
        {
            if( dg::file::error::is_throw == mode)
                throw std::runtime_error( "Value "+temp+" for perp_diff[1] is invalid! Must be either explicit or implicit\n");
            else if ( dg::file::error::is_warning == mode)
                std::cerr << "Value "+temp+" for perp_diff[1] is invalid!\n";
            else
                ;
        }

        //nu_parallel = dg::file::get( mode, js, "nu_parallel", 0.).asDouble();
        //Init after reading in eta and mu[0]
        nu_parallel[0] = 0.73/eta;
        nu_parallel[1] = sqrt(fabs(mu[0]))*1.36/eta;

        initne      = dg::file::get( mode, js, "initne", "blob").asString();
        initphi     = dg::file::get( mode, js, "initphi", "zero").asString();
        amp         = dg::file::get( mode, js, "amplitude", 0.).asDouble();
        sigma       = dg::file::get( mode, js, "sigma", 0.).asDouble();
        posX        = dg::file::get( mode, js, "posX", 0.).asDouble();
        posY        = dg::file::get( mode, js, "posY", 0.).asDouble();
        sigma_z     = dg::file::get( mode, js, "sigma_z", 0.).asDouble();
        k_psi       = dg::file::get( mode, js, "k_psi", 0.).asDouble();

        nprofamp   = dg::file::get( mode, js, "profile", "amp", 0.).asDouble();
        profile_alpha = dg::file::get( mode, js, "profile", "alpha", 0.2).asDouble();

        source_rate     = dg::file::get( mode, js, "source", "rate", 0.).asDouble();
        source_type     = dg::file::get( mode, js, "source", "type", "profile").asString();
        sheath_bc       = dg::file::get( mode, js, "sheath", "bc", "bohm").asString();
        source_boundary = dg::file::get( mode, js, "source", "boundary", 0.5).asDouble();
        source_alpha    = dg::file::get( mode, js, "source", "alpha", 0.2).asDouble();
        wall_rate = dg::file::get( mode, js, "wall", "penalization", 0.).asDouble();
        sheath_rate  = dg::file::get( mode, js, "sheath", "penalization", 0.).asDouble();

        bcxN = dg::str2bc(dg::file::get_idx( mode, js, "bc", "density", 0, "").asString());
        bcyN = dg::str2bc(dg::file::get_idx( mode, js, "bc", "density", 1, "").asString());
        bcxU = dg::str2bc(dg::file::get_idx( mode, js, "bc", "velocity", 0, "").asString());
        bcyU = dg::str2bc(dg::file::get_idx( mode, js, "bc", "velocity", 1, "").asString());
        bcxP = dg::str2bc(dg::file::get_idx( mode, js, "bc", "potential", 0, "").asString());
        bcyP = dg::str2bc(dg::file::get_idx( mode, js, "bc", "potential", 1, "").asString());

        boxscaleRm  = dg::file::get_idx( mode, js, "box", "scaleR", 0u, 1.05).asDouble();
        boxscaleRp  = dg::file::get_idx( mode, js, "box", "scaleR", 1u, 1.05).asDouble();
        boxscaleZm  = dg::file::get_idx( mode, js, "box", "scaleZ", 0u, 1.05).asDouble();
        boxscaleZp  = dg::file::get_idx( mode, js, "box", "scaleZ", 1u, 1.05).asDouble();

        curvmode    = dg::file::get( mode, js, "curvmode", "toroidal").asString();
        symmetric   = dg::file::get( mode, js, "symmetric", false).asBool();
    }
};

}//namespace feltor
