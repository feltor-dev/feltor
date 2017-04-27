#pragma once
#include "dg/enums.h"
#include "json/json.h"

namespace eule{
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
    double eps_maxwell; //!< accuracy of induction equation
    double eps_gamma; //!< accuracy of gamma operator
    double eps_time;//!< accuracy of implicit timestep
    double eps_hat;//!< 1

    double mu[2]; //!< mu[0] = mu_e, m[1] = mu_i
    double tau[2]; //!< tau[0] = -1, tau[1] = tau_i
    double beta; //!< plasma beta

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
    unsigned pollim; //!< 0= no poloidal limiter, 1 = poloidal limiter
    unsigned pardiss; //!< 0 = adjoint parallel dissipation, 1 = nonadjoint parallel dissipation
    unsigned mode; //!< 0 = blob simulations (several rounds fieldaligned), 1 = straight blob simulation( 1 round fieldaligned), 2 = turbulence simulations ( 1 round fieldaligned), 
    unsigned initcond; //!< 0 = zero electric potential, 1 = ExB vorticity equals ion diamagnetic vorticity
    unsigned curvmode; //!< 0 = low beta, 1 = toroidal field line 
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
        eps_maxwell = js["eps_maxwell"].asDouble();
        eps_gamma   = js["eps_gamma"].asDouble();
        eps_time    = js["eps_time"].asDouble();
        eps_hat     = 1.;

        mu[0]       = js["mu"].asDouble();
        mu[1]       = +1.;
        tau[0]      = -1.;
        tau[1]      = js["tau"].asDouble();
        beta        = js["beta"].asDouble();
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

        pollim      = js.get( "pollim", 0).asUInt();
        pardiss     = js.get( "pardiss", 0).asUInt();
        mode        = js.get( "mode", 0).asUInt();
        initcond    = js.get( "initial", 0).asUInt();
        curvmode    = js.get( "curvmode", 0).asUInt();
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
            <<"     beta              = "<<beta<<"\n"
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
            <<"     Stopping for Maxwell CG: "<<eps_maxwell<<"\n"
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
            <<"     Poloidal limiter      =              "<<pollim<<"\n"
            <<"     Parallel dissipation  =              "<<pardiss<<"\n"
            <<"     Computation mode      =              "<<mode<<"\n"
            <<"     init cond             =              "<<initcond<<"\n"
            <<"     curvature mode        =              "<<curvmode<<"\n";
        os << std::flush;
    }
};

}//namespace eule


    

