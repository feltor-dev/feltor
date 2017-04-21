#pragma once
#include <string>
#include "dg/enums.h"
#include "json/json.h"

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny; 
    double dt; 

    double eps, eps_time;
    double lx, ly; 
    enum dg::bc bc_x, bc_y;

    double D;
    int global;

    double U, R, posX, posY;

    unsigned itstp; 
    unsigned maxout;
    std::string initial;

    Parameters( const Json::Value& js) {
        //names are chosen to be as consisten with other projects as possible
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        dt = js["dt"].asDouble();
        eps = js["eps_pol"].asDouble();
        eps_time = js["eps_time"].asDouble();
        lx = js["lx"].asDouble();
        ly = js["ly"].asDouble();
        bc_x = dg::str2bc(js["bc_x"].asString());
        bc_y = dg::str2bc(js["bc_y"].asString());
        D = js["nu_perp"].asDouble();
        R = js["sigma"].asDouble();
        posX = js["posX"].asDouble();
        posY = js["posY"].asDouble();
        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();
        U = js.get("velocity", 1).asDouble();
        initial = js.get("initial", "lamb").asString();
    }

    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<D<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<bc2str(bc_x)<<"\n"
            <<"Boundary conditions in y are: \n"
            <<"    "<<bc2str(bc_y)<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Dipole parameters are: \n"
            << "    radius:       "<<R<<"\n"
            << "    velocity:     "<<U<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Stopping for CG:         "<<eps<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Initial condition:       "<<initial<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl;

    }
};
