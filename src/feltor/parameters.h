#pragma once
#include <map>
#include <array>
#include <string>
#include <cmath>

#include "dg/file/json_utilities.h"

namespace feltor{
/// If you need more parameters, just go ahead and extend the list
struct Parameters
{
    unsigned n, Nx, Ny, Nz;
    std::string tableau, timestepper;

    unsigned itstp;

    std::vector<double> eps_pol;
    double jfactor;
    double eps_gamma, eps_ampere;
    unsigned stages;
    unsigned mx, my;
    double rk4eps;
    std::string interpolation_method;
    double nbc;

    std::array<double,2> mu; // mu[0] = mu_e, m[1] = mu_i
    std::array<double,2> tau; // tau[0] = -1, tau[1] = tau_i
    std::array<double,2> nu_parallel_u;
    double eta, beta;

    unsigned diff_order;
    double nu_perp_n, nu_perp_u, nu_parallel_n;
    enum dg::direction diff_dir;
    std::string slope_limiter;

    double source_rate, nwall, uwall, wall_rate;
    double sheath_rate, sheath_max_angle;
    std::string sheath_coord;

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
    bool partitioned;
    //bool mass_conserv, energy_theorem, toroidal_mom, parallel_mom, parallel_e_force, zonal_flow, COCE_GF, COCE_fluid; //To define which variable to be saved in the output (from input)
    bool probes;
    unsigned num_pins;
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
        tableau     = js["timestepper"].get("tableau", "TVB-3-3").asString();
        timestepper = js["timestepper"].get("type", "multistep").asString();
        partitioned = false;
        itstp       = js["output"].get("itstp", 0).asUInt();
        output      = js["output"].get( "type", "netcdf").asString();
        if( !("netcdf" == output) && !("glfw" == output))
            throw std::runtime_error( "Output type "+output+" not recognized!\n");
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
        eps_ampere  = js["elliptic"].get( "eps_ampere", 1e-6).asDouble();
        pol_dir = dg::str2direction(
                js["elliptic"].get("direction", "centered").asString() );


        mx          = js["FCI"]["refine"].get( 0u, 1).asUInt();
        my          = js["FCI"]["refine"].get( 1u, 1).asUInt();
        rk4eps      = js["FCI"].get( "rk4eps", 1e-6).asDouble();
        interpolation_method = js["FCI"].get("interpolation-method", "dg").asString();
        periodify   = js["FCI"].get( "periodify", true).asBool();
        fci_bc      = js["FCI"].get( "bc", "along_field").asString();

        diff_order  = js["regularization"].get( "order", 2).asUInt();
        diff_dir    = dg::str2direction(
                js["regularization"].get( "direction", "centered").asString() );
        nu_perp_n   = js["regularization"].get( "nu_perp_n", 0.).asDouble();
        nu_perp_u   = js["regularization"].get( "nu_perp_u", 0.).asDouble();
        nu_parallel_n = js["regularization"].get( "nu_parallel_n", 0.).asDouble();
        slope_limiter = js["advection"].get("slope-limiter", "none").asString();
        if( (slope_limiter != "none") && (slope_limiter != "minmod")
             && (slope_limiter != "vanLeer")
                )
            throw std::runtime_error( "ERROR: advection : slope-limiter "+slope_limiter+" not recognized!\n");

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
            nu_parallel_u[0] = 0.37/eta;
            nu_parallel_u[1] = sqrt(fabs(mu[0]))*pow(tau[1], 1.5)*0.69/eta;
        }
        else if ( viscosity == "value")
        {
            nu_parallel_u[0] = js["physical"]["nu_parallel"].get(0u, 1.0).asDouble();
            nu_parallel_u[1] = js["physical"]["nu_parallel"].get(1u, 1.0).asDouble();
        }
        else
            throw std::runtime_error( "ERROR: physical viscosity "+viscosity+" not recognized!\n");


        source_rate = 0.;
        if( js["source"].get("type", "zero").asString() != "zero")
            source_rate = js[ "source"].get( "rate", 0.).asDouble();
        sheath_bc = js["boundary"]["sheath"].get("type", "bohm").asString();
        if( (sheath_bc != "bohm") && (sheath_bc != "insulating") &&
                (sheath_bc != "none") && (sheath_bc != "wall"))
            throw std::runtime_error( "ERROR: Sheath bc "+sheath_bc+" not recognized!\n");

        bcxN = dg::str2bc(js["boundary"]["bc"][  "density"].get( 0, "").asString());
        bcyN = dg::str2bc(js["boundary"]["bc"][  "density"].get( 1, "").asString());
        nbc = 0.;
        if( bcxN == dg::DIR || bcxN == dg::DIR_NEU || bcxN == dg::NEU_DIR
            || bcyN == dg::DIR || bcyN == dg::DIR_NEU || bcyN == dg::NEU_DIR)
            nbc = js["boundary"]["bc"].get( "nbc", 1.0).asDouble();

        bcxU = dg::str2bc(js["boundary"]["bc"][ "velocity"].get( 0, "").asString());
        bcyU = dg::str2bc(js["boundary"]["bc"][ "velocity"].get( 1, "").asString());
        bcxP = dg::str2bc(js["boundary"]["bc"]["potential"].get( 0, "").asString());
        bcyP = dg::str2bc(js["boundary"]["bc"]["potential"].get( 1, "").asString());
        bcxA = dg::str2bc(js["boundary"]["bc"]["aparallel"].get( 0, "").asString());
        bcyA = dg::str2bc(js["boundary"]["bc"]["aparallel"].get( 1, "").asString());

        if( fci_bc == "along_field" || fci_bc == "perp")
        {
            if( bcxN != bcyN || bcxN == dg::DIR_NEU || bcxN == dg::NEU_DIR)
                throw std::runtime_error( "ERROR: density bc must be either dg::NEU or dg::DIR in both directions!\n");
            if( bcxU != bcyU || bcxU == dg::DIR_NEU || bcxU == dg::NEU_DIR)
                throw std::runtime_error( "ERROR: velocity bc must be either dg::NEU or dg::DIR in both directions!\n");
            if( bcxP != bcyP || bcxP == dg::DIR_NEU || bcxP == dg::NEU_DIR)
                throw std::runtime_error( "ERROR: potential bc must be either dg::NEU or dg::DIR in both directions!\n");
        }
        else if( fci_bc != "perp")
            throw std::runtime_error("Error! FCI bc '"+fci_bc+"' not recognized!\n");


        curvmode    = js["magnetic_field"].get( "curvmode", "toroidal").asString();
        modify_B = penalize_wall = penalize_sheath = false;
        nwall = uwall = wall_rate = sheath_rate = sheath_max_angle = 0.;
        sheath_coord = "s";
        if( js["boundary"]["wall"].get("type","none").asString() != "none")
        {
            modify_B = js["boundary"]["wall"].get( "modify-B", false).asBool();
            penalize_wall = js["boundary"]["wall"].get( "penalize-rhs",
                    false).asBool();
            wall_rate = js ["boundary"]["wall"].get( "penalization",
                    0.).asDouble();
            nwall = js["boundary"]["wall"].get( "nwall", 1.0).asDouble();
            uwall = js["boundary"]["wall"].get( "uwall", 0.0).asDouble();
        }
        if( js["boundary"]["sheath"].get("type","none").asString() != "none")
        {
            penalize_sheath = js["boundary"]["sheath"].get( "penalize-rhs",
                    false).asBool();
            sheath_rate = js ["boundary"]["sheath"].get( "penalization",
                    0.).asDouble();
            sheath_coord = js["boundary"]["sheath"].get( "coordinate", "s").asString();
            sheath_max_angle = js["boundary"]["sheath"].get( "max_angle", 4).asDouble()*2.*M_PI;
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

        //Probes
        probes = js.isMember("probes");
        if(probes)
        {
            //num_pins = js["probes"]["num_pins"].asUInt();
            num_pins = js["probes"]["R"].size();
            unsigned num_pinsZ = js["probes"]["Z"].size();
            unsigned num_pinsP = js["probes"]["P"].size();
            if( num_pins != num_pinsZ)
                throw std::runtime_error( "Size of Z probes array ("
                        +std::to_string(num_pinsZ)+") does not match that of R ("
                        +std::to_string(num_pins)+")!");
            if( num_pins != num_pinsP)
                throw std::runtime_error( "Size of P probes array ("
                        +std::to_string(num_pinsP)+") does not match that of R ("
                        +std::to_string(num_pins)+")!");
        }
        else
            num_pins = 0;
        if( js.isMember("probe"))
            throw std::runtime_error( "Field <probe> found! Did you mean <probes>?");
    }
};

}//namespace feltor
