#pragma once
#include "dg/enums.h"
#include "json/json.h"

namespace asela{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n; //!< \# of polynomial coefficients in R and Z
    unsigned Nx; //!< \# of cells in x -direction
    unsigned Ny; //!< \# of cells in y -direction

    double dt;  //!< timestep
    double lxhalf, lyhalf;
    unsigned stages;
    std::vector<double> eps_pol;  //!< accuracy of polarization
    double jfactor; //jump factor € [1,0.01]
    double eps_maxwell; //!< accuracy of induction equation
    double eps_gamma; //!< accuracy of gamma operator
    std::string direction_diff, direction_ell;
    std::string advection;

    double mu[2]; //!< mu[0] = mu_e, m[1] = mu_i
    double tau[2]; //!< tau[0] = -1, tau[1] = tau_i
    double beta; //!< plasma beta

    double nu_perp;  //!< perpendicular diffusion

    Parameters( const Json::Value& js, enum dg::file::error mode = dg::file::error::is_warning ) {
        n       = dg::file::get( mode, js, "grid", "n", 3).asUInt();
        Nx      = dg::file::get( mode, js, "grid", "Nx", 48).asUInt();
        Ny      = dg::file::get( mode, js, "grid", "Ny", 48).asUInt();
        lxhalf  = dg::file::get( mode, js, "grid", "lxhalf", 80).asDouble();
        lyhalf  = dg::file::get( mode, js, "grid", "lyhalf", 80).asDouble();

        dt      = dg::file::get( mode, js, "timestepper", "dt", 20).asDouble();

        advection = dg::file::get( mode, js, "advection", "type", "arakawa").asString();

        stages      = dg::file::get( mode, js, "elliptic", "stages", 3).asUInt();
        eps_pol.resize(stages);
        eps_pol[0] = dg::file::get_idx( mode, js, "elliptic", "eps_pol", 0, 1e-6).asDouble();
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = dg::file::get_idx( mode, js, "elliptic", "eps_pol", i, 1).asDouble();
            eps_pol[i]*=eps_pol[0];
        }
        jfactor     = dg::file::get( mode, js, "elliptic", "jumpfactor", 1).asDouble();
        direction_ell = dg::file::get( mode, js, "elliptic", "direction", "forward").asString();
        eps_maxwell = dg::file::get( mode, js, "elliptic", "eps_maxwell", 1e-7).asDouble();
        eps_gamma   = dg::file::get( mode, js, "elliptic", "eps_gamma", 1e-10).asDouble();

        mu[0]    = dg::file::get( mode, js, "physical", "mu", -0.000544617 ).asDouble();
        mu[1]    = +1.;
        tau[0]   = -1.;
        tau[1]   = dg::file::get( mode, js, "physical", "tau",  0.0 ).asDouble();
        beta     = dg::file::get( mode, js, "physical", "beta", 1e-4).asDouble();
        nu_perp  = dg::file::get( mode, js, "regularization", "nu_perp", 1e-4).asDouble();
        direction_diff = dg::file::get( mode, js, "regularization", "direction", "centered").asString();


    }
};

}//namespace eule

