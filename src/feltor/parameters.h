#pragma once
#include <string>
#include "dg/enums.h"
#include "json/json.h"

namespace feltor{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n; //!< \# of polynomial coefficients in R and Z
    unsigned Nx; //!< \# of cells in x -direction
    unsigned Ny; //!< \# of cells in y -direction
    unsigned Nz; //!< \# of cells in z -direction

    double dt;  //!< timestep
    unsigned n_out;  //!< \# of polynomial coefficients in output file
    unsigned Nx_out;  //!< \# of cells in x-direction in output file
    unsigned Ny_out; //!< \# of cells in y-direction in output file
    unsigned Nz_out; //!< \# of cells in z-direction in output file
    unsigned itstp; //!< \# of steps between outputs
    unsigned maxout; //!< \# of outputs excluding first

    double eps_pol;  //!< accuracy of polarization 
    double jfactor; //jump factor € [1,0.01]
    double eps_gamma; //!< accuracy of gamma operator
    double eps_time;//!< accuracy of implicit timestep
    double eps_hat;//!< 1
    unsigned stages; //!< # of stages in multigrid

    std::array<double,2> mu; //!< mu[0] = mu_e, m[1] = mu_i
    std::array<double,2> tau; //!< tau[0] = -1, tau[1] = tau_i

    double nu_perp;  //!< perpendicular diffusion
    double nu_parallel;  //!< parallel diffusion
    double c; //!< parallel resistivity

    double amp;  //!< blob amplitude
    double sigma; //!< perpendicular blob width
    double posX;  //!< perpendicular position relative to box width
    double posY; //!< perpendicular position relative to box height
    double sigma_z; //!< parallel blob width in units of pi
    double k_psi; //!< mode number

    double omega_source; //!< source amplitude 
    double nprofileamp; //!< amplitude of profile
    double bgprofamp; //!< background profile amplitude
    double boxscaleRp; //!< box can be larger
    double boxscaleRm;//!< box can be larger
    double boxscaleZp;//!< box can be larger
    double boxscaleZm;//!< box can be larger

    enum dg::bc bc; //!< global perpendicular boundary condition
    bool pollim; //!< 0= no poloidal limiter, 1 = poloidal limiter
    std::string initni; //!< "blob" = blob simulations (several rounds fieldaligned), "straight blob" = straight blob simulation( 1 round fieldaligned), "turbulence" = turbulence simulations ( 1 round fieldaligned),
    std::string initphi; //!< "zero" = 0 electric potential, "balance" = ExB vorticity equals ion diamagnetic vorticity
    std::string curvmode; //!< "low beta", "toroidal" toroidal field line approximation
    Parameters( const Json::Value& js) {
        n       = js["n"].asUInt();
        Nx      = js["Nx"].asUInt();
        Ny      = js["Ny"].asUInt();
        Nz      = js["Nz"].asUInt();
        dt      = js["dt"].asDouble();
        n_out   = js["n_out"].asUInt();
        Nx_out  = js["Nx_out"].asUInt();
        Ny_out  = js["Ny_out"].asUInt();
        Nz_out  = js["Nz_out"].asUInt();
        itstp   = js["itstp"].asUInt();
        maxout  = js["maxout"].asUInt();

        eps_pol     = js["eps_pol"].asDouble();
        jfactor     = js["jumpfactor"].asDouble();
        eps_gamma   = js["eps_gamma"].asDouble();
        eps_time    = js["eps_time"].asDouble();
        eps_hat     = 1.;
        stages      = js.get( "stages", 3).asUInt();

        mu[0]       = js["mu"].asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = js["tau"].asDouble();
        nu_perp     = js["nu_perp"].asDouble();
        nu_parallel = js["nu_parallel"].asDouble();
        c           = js["resistivity"].asDouble();

        amp         = js["amplitude"].asDouble();
        sigma       = js["sigma"].asDouble();
        posX        = js["posX"].asDouble();
        posY        = js["posY"].asDouble();
        sigma_z     = js["sigma_z"].asDouble();
        k_psi       = js["k_psi"].asDouble();
        omega_source = js["source"].asDouble();

        bc          = dg::str2bc(js["bc"].asString());
        nprofileamp = js["nprofileamp"].asDouble();
        bgprofamp   = js["bgprofamp"].asDouble();

        boxscaleRp  = js.get("boxscaleRp",1.05).asDouble();
        boxscaleRm  = js.get("boxscaleRm",1.05).asDouble();
        boxscaleZp  = js.get("boxscaleZp",1.05).asDouble();
        boxscaleZm  = js.get("boxscaleZm",1.05).asDouble();

        pollim      = js.get( "pollim", "false").asBool();
        initni      = js.get( "initni", "blob").asString();
        initphi     = js.get( "initphi", "zero").asString();
        curvmode    = js.get( "curvmode", "toroidal").asString();
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"     mu_e              = "<<mu[0]<<"\n"
            <<"     mu_i              = "<<mu[1]<<"\n"
            <<"     El.-temperature:  = "<<tau[0]<<"\n"
            <<"     Ion-temperature:  = "<<tau[1]<<"\n"
            <<"     perp. Viscosity:  = "<<nu_perp<<"\n"
            <<"     par. Resistivity: = "<<c<<"\n"
            <<"     par. Viscosity:   = "<<nu_parallel<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n"
            << "    sigma_z:      "<<sigma_z<<"\n";
        os << "Profile parameters are: \n"
            <<"     omega_source:                 "<<omega_source<<"\n"
            <<"     density profile amplitude:    "<<nprofileamp<<"\n"
            <<"     background profile amplitude: "<<bgprofamp<<"\n"
            <<"     boxscale R+:                  "<<boxscaleRp<<"\n"
            <<"     boxscale R-:                  "<<boxscaleRm<<"\n"
            <<"     boxscale Z+:                  "<<boxscaleZp<<"\n"
            <<"     boxscale Z-:                  "<<boxscaleZm<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     Nz = "<<Nz<<"\n"
            <<"     dt = "<<dt<<"\n";
        os << "     Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"     Jump scale factor:   "<<jfactor<<"\n"
            <<"     Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"     Stopping for Time  CG:   "<<eps_time<<"\n";
        os << "Output parameters are: \n"
            <<"     n_out  =              "<<n_out<<"\n"
            <<"     Nx_out =              "<<Nx_out<<"\n"
            <<"     Ny_out =              "<<Ny_out<<"\n"
            <<"     Nz_out =              "<<Nz_out<<"\n"
            <<"     Steps between output: "<<itstp<<"\n"
            <<"     Number of outputs:    "<<maxout<<"\n";
        os << "Boundary condition is: \n"
            <<"     global BC             =              "<<dg::bc2str(bc)<<"\n"
            <<"     Poloidal limiter      =              "<<std::boolalpha<<pollim<<"\n"
            <<"     init N_i              =              "<<initni<<"\n"
            <<"     init Phi              =              "<<initphi<<"\n"
            <<"     curvature mode        =              "<<curvmode<<"\n";
        os << std::flush;
    }
};

}//namespace feltor
