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
    double dt;
    std::string tableau;

    unsigned inner_loop;
    unsigned itstp;

    std::vector<double> eps_pol;
    double jfactor;
    double eps_gamma;
    unsigned stages;
    unsigned mx, my;
    double rk4eps;

    std::array<double,2> mu; // mu[0] = mu_e, m[1] = mu_i
    std::array<double,2> tau; // tau[0] = -1, tau[1] = tau_i
    std::array<double,2> nu_parallel;
    double eta, beta;

    unsigned diff_order;
    double nu_perp_n, nu_perp_u;
    enum dg::direction diff_dir;

    double source_rate, wall_rate, sheath_rate;

    double boxscaleRm, boxscaleRp;
    double boxscaleZm, boxscaleZp;

    enum dg::bc bcxN, bcyN, bcxU, bcyU, bcxP, bcyP, bcxA, bcyA;
    enum dg::direction pol_dir;
    std::string curvmode;
    std::string sheath_bc;
    std::string fci_bc;
    std::string output;
    bool symmetric, calibrate, periodify;
    bool penalize_wall, penalize_sheath, modify_B;

    //

    Parameters() = default;
    Parameters( const dg::file::WrappedJsonValue& js) {
        //We need to check if a member is present
        n           = js["grid"].get("n", 3).asUInt();
        Nx          = js["grid"].get("Nx", 0).asUInt();
        Ny          = js["grid"].get("Ny", 0).asUInt();
        Nz          = js["grid"].get("Nz", 0).asUInt();
        boxscaleRm  = js["grid"][ "scaleR"].get( 0u, 1.05).asDouble();
        boxscaleRp  = js["grid"][ "scaleR"].get( 1u, 1.05).asDouble();
        boxscaleZm  = js["grid"][ "scaleZ"].get( 0u, 1.05).asDouble();
        boxscaleZp  = js["grid"][ "scaleZ"].get( 1u, 1.05).asDouble();
        dt      = js["timestepper"].get("dt", 0.).asDouble();
        tableau = js["timestepper"].get("tableau", "TVB-3-3").asString();
        inner_loop  = js["output"].get("inner_loop",1).asUInt();
        itstp       = js["output"].get("itstp", 0).asUInt();
        output      = js["output"].get( "type", "netcdf").asString();
        if( !("netcdf" == output) && !("glfw" == output))
            throw std::runtime_error( "Output type "+output+" not
                    recognized!\n");
#ifdef WITHOUT_GLFW
        if( "glfw" == output)
            throw std::runtime_error( "Output type glfw not possible without glfw compiled!\n");
#endif

        stages      = js["elliptic"].get( "stages", 3).asUInt();
        eps_pol.resize(stages);
        eps_pol[0] = js["elliptic"]["eps_pol"].get( 0, 1e-6).asDouble();
        for( unsigned i=1;i<stages; i++)
        {
            eps_pol[i] = js["elliptic"][ "eps_pol"].get( i, 1).asDouble();
            eps_pol[i]*=eps_pol[0];
        }
        jfactor     = js["elliptic"].get( "jumpfactor", 1).asDouble();
        eps_gamma   = js["elliptic"].get( "eps_gamma", 1e-6).asDouble();
        eps_maxwell = js["elliptic"].get( "eps_maxwell", 1e-6).asDouble();
        dir_pol = dg::str2direction(
                js["elliptic"].get("direction", "centered").asString() );


        mx          = js["FCI"]["refine"].get( 0u, 1).asUInt();
        my          = js["FCI"]["refine"].get( 1u, 1).asUInt();
        rk4eps      = js["FCI"].get( "rk4eps", 1e-6).asDouble();
        periodify   = js["FCI"].get( "periodify", true).asBool();
        fci_bc      = js["FCI"].get( "bc", "along_field").asString();

        diff_order  = js["regularization"].get( "order", 2).asUInt();
        diff_dir    = dg::str2direction(
                js["regularization"].get( "direction", "centered").asString() );
        nu_perp_n   = js["regularization"].get( "nu_perp_n", 0.).asDouble();
        nu_perp_u   = js["regularization"].get( "nu_perp_u", 0.).asDouble();

        mu[0]       = js["physical"].get( "mu", -0.000272121).asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = js["physical"].get( "tau", 0.).asDouble();
        beta        = js["physical"].get( "beta", 0.).asDouble();
        eta         = js["physical"].get( "resistivity", 0.).asDouble();
        //Init after reading in eta and mu[0]
        std::string viscosity = js["physical"].get( "viscosity",
                "braginskii").asString();
        if( viscosity == "braginskii")
        {
            nu_parallel[0] = 0.73/eta;
            nu_parallel[1] = sqrt(fabs(mu[0]))*1.36/eta;
        }
        else if ( viscosity == "value")
        {
            nu_parallel[0] = js["physical"].get("nu_parallel", 1.0).asDouble();
            nu_parallel[1] = nu_parallel[1];
        }
        else
            throw std::runtime_error( "physical viscosity "+viscosity+" not
                    recognized!\n");


        source_rate = 0.;
        if( js["source"].get("type", "zero").asString() != "zero")
            source_rate = js[ "source"].get( "rate", 0.).asDouble();
        sheath_bc       = js["sheath"].get("bc", "bohm").asString();

        bcxN = dg::str2bc(js["boundary"]["bc"][  "density"].get( 0, "").asString());
        bcyN = dg::str2bc(js["boundary"]["bc"][  "density"].get( 1, "").asString());
        bcxU = dg::str2bc(js["boundary"]["bc"][ "velocity"].get( 0, "").asString());
        bcyU = dg::str2bc(js["boundary"]["bc"][ "velocity"].get( 1, "").asString());
        bcxP = dg::str2bc(js["boundary"]["bc"]["potential"].get( 0, "").asString());
        bcyP = dg::str2bc(js["boundary"]["bc"]["potential"].get( 1, "").asString());
        bcxA = dg::str2bc(js["boundary"]["bc"]["aparallel"].get( 0, "").asString());
        bcyA = dg::str2bc(js["boundary"]["bc"]["aparallel"].get( 1, "").asString());

        curvmode    = js["magnetic_field"].get( "curvmode", "toroidal").asString();
        modify_B = penalize_wall = penalize_sheath = false;
        wall_rate = sheath_rate = 0.;
        if( js["boundary"]["wall"].get("type","none") != "none")
        {
            modify_B = js["boundary"]["wall"].get( "modify-B", false).asBool();
            penalize_wall = js["boundary"]["wall"].get( "penalize-rhs",
                    false).asBool();
            wall_rate = js ["boundary"]["wall"].get( "penalization",
                    0.).asDouble();
        }
        if( js["boundary"]["sheath"].get("type","none") != "none")
        {
            penalize_sheath = js["boundary"]["sheath"].get( "penalize-rhs",
                    false).asBool();
            sheath_rate = js ["boundary"]["wall"].get( "penalization",
                    0.).asDouble();
        }

        // Computing flags
        symmetric = calibrate = false;
        for( unsigned i=0; i<js["flags"].size(); i++)
        {
            std::string flag = js["flags"].get(i,"symmetric").asString();
            if( flag  == "symmetric")
                symmetric = true;
            else if( flag == "calibrate" )
            {
                if( output == "glfw")
                    throw std::runtime_error(
                            "Calibrate not possible with glfw output!\n");
                calibrate = true;
            }
            else
                throw std::runtime_error( "Flag "+flag+" not recognized!\n");
        }
    }
};

}//namespace feltor
