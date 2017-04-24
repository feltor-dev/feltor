#pragma once
#include "dg/enums.h"
#include "json/json.h"

namespace eule
{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny; 
    double dt; 

    double eps_pol, eps_maxwell, eps_time;
    double lxhalf, lyhalf; 

    double nu;

    double dhat[2];
    double rhohat[2];
    double tau[2];

    double amp;

    unsigned itstp; 
    unsigned maxout;
    dg::bc bc_x, bc_y;

    Parameters( const Json::Value& js) {
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        dt = js["dt"].asDouble();
        eps_pol = js["eps_pol"].asDouble();
        eps_maxwell = js["eps_maxwell"].asDouble();
        eps_time = js["eps_time"].asDouble();
        lxhalf = js["lxhalf"].asDouble()*M_PI;
        lyhalf = js["lyhalf"].asDouble()*M_PI;
        bc_x = dg::str2bc(js["bc_x"].asString());
        bc_y = dg::str2bc(js["bc_y"].asString());
        dhat[0] = js["dehat"].asDouble();
        dhat[1] = js["dihat"].asDouble();
        rhohat[0] = js["rhoshat"].asDouble();
        rhohat[1] = js["rhoihat"].asDouble();
        nu = js["nu_perp"].asDouble();
        amp = js["amplitude"].asDouble();
        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();
    }
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    rhos  = "<<rhohat[0]<<"\n"
            <<"    rhoi  = "<<rhohat[1]<<"\n"
            <<"    de: = "<<dhat[0]<<"\n"
            <<"    di: = "<<dhat[1]<<"\n"
            <<"    Viscosity:       = "<<nu<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<2.*lxhalf<<"\n"
            <<"    ly = "<<2.*lyhalf<<"\n";
        os << "Boundary conditions in x are: \n"
            <<"    "<<bc2str(bc_x)<<"\n";
        os << "Boundary conditions in y are: \n"
            <<"    "<<bc2str(bc_y)<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Perturbation parameters are: \n"
            << "    amplitude:    "<<amp<<"\n";
        os << "Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"Stopping for Aparl CG:   "<<eps_maxwell<<"\n"
            <<"Stopping for Time  CG:   "<<eps_time<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl; //the endl is for the implicit flush 
    }
};
} //namespace eule


    

