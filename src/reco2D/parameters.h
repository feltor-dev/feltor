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

    double lxhalf, lyhalf;
    unsigned stages;
    std::vector<double> eps_pol;  //!< accuracy of polarization
    double jfactor; //jump factor € [1,0.01]
    double eps_maxwell; //!< accuracy of induction equation
    double eps_gamma; //!< accuracy of gamma operator
    std::string direction_diff, direction_ell;
    std::string advection, timestepper, viscosity;

    double mu[2]; //!< mu[0] = mu_e, m[1] = mu_i
    double tau[2]; //!< tau[0] = -1, tau[1] = tau_i
    double beta; //!< plasma beta

    double nu_perp;  //!< perpendicular diffusion

    Parameters( const dg::file::WrappedJsonValue& ws ) {
        n       = ws["grid"].get("n", 3).asUInt();
        Nx      = ws["grid"].get("Nx", 48).asUInt();
        Ny      = ws["grid"].get("Ny", 48).asUInt();
        lxhalf  = ws["grid"].get("lxhalf", 80).asDouble();
        lyhalf  = ws["grid"].get("lyhalf", 80).asDouble();

        timestepper = ws["timestepper"].get("type", "multistep").asString();

        advection = ws["advection"].get("type", "arakawa").asString();

        auto ell = ws["elliptic"];
        stages      = ell.get( "stages",3).asUInt();
        eps_pol.resize(stages);
        eps_pol[0] = ell[ "eps_pol"].get(0, 1e-6).asDouble();
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = ell[ "eps_pol"].get(i, 1.0).asDouble();
            eps_pol[i]*=eps_pol[0];
        }
        jfactor     = ell.get( "jumpfactor", 1.0).asDouble( );
        direction_ell = ell.get("direction", "forward").asString( );
        eps_maxwell = ell.get( "eps_maxwell", 1e-7).asDouble( );
        eps_gamma   = ell.get( "eps_gamma", 1e-10).asDouble( );

        mu[0]    = ws["physical"].get("mu", -0.000544617 ).asDouble( );
        mu[1]    = +1.;
        tau[0]   = -1.;
        tau[1]    = ws["physical"].get("tau", 0.0).asDouble( );
        beta      = ws["physical"].get("beta", 1e-4).asDouble();
        viscosity = ws["regularization"].get("type", "velocity-viscosity").asString();
        nu_perp   = ws["regularization"].get("nu_perp",1e-4).asDouble();
        direction_diff = ws["regularization"].get("direction", "centered").asString();


    }
};

}//namespace eule

