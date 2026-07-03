#pragma once
#include <map>
#include <array>
#include <string>
#include <cmath>

#include "dg/file/json_utilities.h"

namespace thermal{
/// If you need more parameters, just go ahead and extend the list
struct Parameters
{
    unsigned n, Nx, Ny, Nz;

    unsigned itstp, maxout;
    double Tend, deltaT;
    unsigned cx, cy;

    std::vector<double> eps_pol;
    double jfactor;
    double eps_gamma, eps_ampere;
    unsigned stages;
    unsigned mx, my;
    double rk4eps;
    std::string interpolation_method;

    double nu_ref, beta;

    unsigned diff_order;
    std::array<double, 6> nu_perp, nu_parallel;
    enum dg::direction diff_dir;

    std::string wall_bc;
    std::vector<double> nwall;
    double uwall, twall, qwall, wall_rate;

    enum dg::bc bcx, bcy, bcxP, bcyP, bcxA, bcyA;
    enum dg::direction pol_dir;
    std::string curvmode;
    std::string sheath_bc;
    std::string fci_bc;
    std::string output;
    bool symmetric, calibrate, isothermal;
    bool penalize_wall, penalize_sheath;
    bool partitioned;
    //
    //
    unsigned num_species; // number of species
    double qlandau;
    std::vector<double> z;
    std::vector<double> mu;
    std::vector<double> pi, kappa; // prefactors for Braginskii viscosity and conductivity
    std::vector<std::string> name; // name of species s

    Parameters() = default;
    Parameters( const dg::file::WrappedJsonValue& js) {
        //We need to check if a member is present
        n           = js["grid"].get("n", 3).asUInt();
        Nx          = js["grid"].get("Nx", 0).asUInt();
        Ny          = js["grid"].get("Ny", 0).asUInt();
        Nz          = js["grid"].get("Nz", 0).asUInt();
        partitioned = false;
        output      = js["output"].get( "type", "netcdf").asString();
        if( !("netcdf" == output) && !("glfw" == output))
            throw std::runtime_error( "Output type "+output+" not recognized!\n");
        if( "glfw" == output)
            throw std::runtime_error( "Output type glfw not possible without glfw compiled!\n");
        cx = js["output"]["compression"].get(0u,1).asUInt();
        cy = js["output"]["compression"].get(1u,1).asUInt();

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
        fci_bc      = js["FCI"].get( "bc", "along_field").asString();

        diff_order  = js["regularization"].get( "order", 2).asUInt();
        diff_dir    = dg::str2direction(
                js["regularization"].get( "direction", "centered").asString() );
        for( unsigned u=0; u<6; u++)
        {
            nu_perp[u]   = js["regularization"]["nu_perp"].get( u, 0.).asDouble();
            nu_parallel[u] = js["regularization"]["nu_parallel"].get( u, 0.).asDouble();
        }

        num_species = js["species"].size();
        name.resize( num_species);
        mu.resize( num_species);
        z.resize( num_species);
        pi.resize( num_species);
        kappa.resize( num_species);
        for( unsigned s=0; s<num_species; s++)
        {
            name[s] = js["species"][s]["name"].asString();
            mu[s] = js["species"][s]["mu"].asDouble();
            z[s] = js["species"][s]["z"].asDouble();
            pi[s] = js["species"][s]["pi"].asDouble();
            kappa[s] = js["species"][s]["kappa"].asDouble();

        }
        if( fabs(mu[0] ) > 1e-3)
        {
            throw std::runtime_error( "ERROR: First species must be electrons with negligible mass "
                +std::to_string(mu[0])+" !\n");
        }
        beta        = js["physical"].get( "beta", 0.).asDouble();
        nu_ref      = js["physical"].get( "collisionality", 0.).asDouble();
        qlandau     = js["physical"].get( "qlandau", 0.).asDouble();


        sheath_bc = js["boundary"]["sheath"].get("type", "insulating").asString();
        if( (sheath_bc != "insulating") && // "bohm" is not valid
                (sheath_bc != "none") && (sheath_bc != "wall"))
            throw std::runtime_error( "ERROR: Sheath bc "+sheath_bc+" not recognized!\n");

        if( fci_bc != "along_field" and fci_bc != "perp")
            throw std::runtime_error("Error! FCI bc '"+fci_bc+"' not recognized!\n");
        bcx = dg::NEU, bcy = dg::NEU;
        bcxP = dg::DIR, bcyP = dg::DIR;
        bcxA = dg::NEU, bcyA = dg::NEU;


        curvmode    = js["magnetic_field"].get( "curvmode", "flutemode").asString();
        penalize_wall = penalize_sheath = false;
        nwall.resize( num_species, 0.);
        twall = uwall = qwall = wall_rate = 0.;
        if( js["boundary"]["wall"].get("type","none").asString() != "none")
        {
            penalize_wall = js["boundary"]["wall"].get( "penalize-rhs",
                    false).asBool();
            wall_rate = js ["boundary"]["wall"].get( "penalization",
                    0.).asDouble();
            wall_bc = js["boundary"]["wall"].get( "bc", "fixed").asString();
            if( wall_bc == "fixed")
            {
                double sum_n = 0;
                for( unsigned s=0; s<num_species; s++)
                {
                    nwall[s] = js["boundary"]["wall"]["nwall"].get( s, 1.0).asDouble();
                    sum_n += z[s] * nwall[s];
                }
                if( fabs( sum_n) > 1e-15)
                    throw std::runtime_error( "Sum of wall charge densities " + std::to_string( sum_n) +" is not zero \n");
                uwall = js["boundary"]["wall"].get( "uwall", 0.0).asDouble();
                qwall = js["boundary"]["wall"].get( "qwall", 0.0).asDouble();
                twall = js["boundary"]["wall"].get( "twall", 1.0).asDouble();
            }
            else if( wall_bc != "floating")
                throw std::runtime_error("Error! Wall bc '"+wall_bc+"' not recognized!\n");
        }
        if( sheath_bc != "none")
        {
            penalize_sheath = js["boundary"]["sheath"].get( "penalize-rhs",
                    false).asBool();
        }

        // Computing flags
        symmetric = calibrate = isothermal = false;
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
            else if( flag == "isothermal")
            {
                isothermal = true;
            }
            else
                throw std::runtime_error( "Flag "+flag+" not recognized!\n");
        }

        // output frequencies
        maxout = js["output"].get( "maxout", 0).asUInt();
        std::string output_mode = js["timestepper"].get(
                "output-mode", "Tend").asString();
        Tend = 0, deltaT = 0.;
        itstp       = js["output"].get("itstp", 0).asUInt();
        if( output_mode == "Tend")
        {
            Tend = js["timestepper"].get( "Tend", 1).asDouble();
            deltaT = Tend/(double)(maxout*itstp);
        }
        else if( output_mode == "deltaT")
        {
            deltaT = js["timestepper"].get( "deltaT", 1).asDouble()/(double)(itstp);
            Tend = deltaT*(double)(maxout*itstp);
        }
        else
        {
            throw std::runtime_error( "Error: Unrecognized timestepper: output-mode: '"
                               + output_mode);
        }
    }
};

}//namespace thermal
