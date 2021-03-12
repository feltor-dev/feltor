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
        n       = ws["grid"]["n"].asUInt(3);
        Nx      = ws["grid"]["Nx"].asUInt(48);
        Ny      = ws["grid"]["Ny"].asUInt( 48);
        lxhalf  = ws["grid"]["lxhalf"].asDouble( 80);
        lyhalf  = ws["grid"]["lyhalf"].asDouble( 80);

        timestepper = ws["timestepper"]["type"].asString("multistep");

        advection = ws["advection"]["type"].asString("arakawa");

        auto ell = ws["elliptic"];
        stages      = ell[ "stages"].asUInt(3);
        eps_pol.resize(stages);
        eps_pol[0] = ell[ "eps_pol"][0].asDouble( 1e-6);
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = ell[ "eps_pol"][i].asDouble( 1);
            eps_pol[i]*=eps_pol[0];
        }
        jfactor     = ell[ "jumpfactor"].asDouble( 1);
        direction_ell = ell["direction"].asString( "forward");
        eps_maxwell = ell[ "eps_maxwell"].asDouble( 1e-7);
        eps_gamma   = ell[ "eps_gamma"].asDouble( 1e-10);

        mu[0]    = ws["physical"]["mu"].asDouble( -0.000544617 );
        mu[1]    = +1.;
        tau[0]   = -1.;
        tau[1]    = ws["physical"]["tau"].asDouble(  0.0 );
        beta      = ws["physical"]["beta"].asDouble( 1e-4);
        viscosity = ws["regularization"]["type"].asString( "velocity-viscosity");
        nu_perp   = ws["regularization"]["nu_perp"].asDouble( 1e-4);
        direction_diff = ws["regularization"]["direction"].asString( "centered");


    }
};

}//namespace eule

