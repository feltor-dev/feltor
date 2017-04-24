#pragma once
//#include <string>
#include "dg/enums.h"
#include "json/json.h"

namespace imp
{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny;
    unsigned n_out, Nx_out, Ny_out;
    double dt; 

    double eps_time, eps_pol, eps_gamma;

    double lx, ly; 
    enum dg::bc bc_x, bc_y;

    double nu, kappa;

    double a[3], mu[3], tau[3];

    double amp, sigma, posX, posY;

    unsigned itstp; 
    unsigned maxout;

    unsigned vorticity;
    unsigned mode;
    double wall_pos, wall_amp, wall_sigma;
    /**
     * @brief constructor to make a const object
     *
     * @param js json object
     */
    Parameters( const Json::Value& js) {
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        dt = js["dt"].asDouble();
        n_out  = js["n_out"].asUInt();
        Nx_out = js["Nx_out"].asUInt();
        Ny_out = js["Ny_out"].asUInt();
        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();

        eps_pol = js["eps_pol"].asDouble();
        eps_gamma = js["eps_gamma"].asDouble();
        eps_time = js["eps_time"].asDouble();
        kappa = js["curvature"].asDouble();
        nu = js["nu_perp"].asDouble();
        amp = js["amplitude"].asDouble();
        sigma = js["sigma"].asDouble();
        posX = js["posX"].asDouble();
        posY = js["posY"].asDouble();
        lx = js["lx"].asDouble();
        ly = js["ly"].asDouble();
        bc_x = dg::str2bc(js["bc_x"].asString());
        bc_y = dg::str2bc(js["bc_y"].asString());
        tau[1] = js["tau"].asDouble();
        a[2] = js["a_z"].asDouble();
        mu[2] =  js["mu_z"].asDouble();
        tau[2] = js["tau_z"].asDouble();

        a[0] = -1, a[1] = 1-a[2];
        mu[0] = 0, mu[1] = 1;
        tau[0] = -1;
        vorticity = js["vorticity"].asDouble();
        mode = js["mode"].asUInt();
        wall_pos = js["wall_pos"].asDouble();
        wall_amp = js["wall_amp"].asDouble();
        wall_sigma = js["wall_sigma"].asDouble();
    }



    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"    Ion-temperature: = "<<tau[1]<<"\n";
        os <<"    a_e   = "<<a[0]<<"\n"
           <<"    mu_e  = "<<mu[0]<<"\n"
           <<"    tau_e = "<<tau[0]<<"\n";
        os <<"    a_i   = "<<a[1]<<"\n"
           <<"    mu_i  = "<<mu[1]<<"\n"
           <<"    tau_i = "<<tau[1]<<"\n";
        os <<"    a_z   = "<<a[2]<<"\n"
           <<"    mu_z  = "<<mu[2]<<"\n"
           <<"    tau_z = "<<tau[2]<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n";
        os << "Boundary conditions in x are: \n"
            <<"    "<<bc2str(bc_x)<<"\n";
        os << "Boundary conditions in y are: \n"
            <<"    "<<bc2str(bc_y)<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Blob parameters are: \n"
            << "    width:        "<<sigma<<"\n"
            << "    amplitude:    "<<amp<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Mode is            "<<mode<<"\n";
        os << "Vorticity is       "<<vorticity<<"\n";
        os << "Wall parameters are: \n"
            << "    pos:          "<<wall_pos<<"\n"
            << "    sigma:        "<<wall_sigma<<"\n"
            << "    amplitude:    "<<wall_amp<<"\n";
        os << "Stopping for CG:         "<<eps_pol<<"\n"
            <<"Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl; //the endl is for the implicit flush 
    }
};
}//namespace imp
