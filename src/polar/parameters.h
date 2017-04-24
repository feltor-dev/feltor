#pragma once
#include <string>
#include "dg/enums.h"
#include "json/json.h"

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, k, Nx, Ny; 
    double dt; 

    double eps, eps_time;
    double r_min, r_max;

    double nu;
    int global;

    double U, R, posX, posY;

    unsigned itstp; 
    unsigned maxout;

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const Json::Value& js) {
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        dt = js["dt"].asDouble();
        eps = js["eps_pol"].asDouble();
        eps_time = js["eps_time"].asDouble();
        r_min = js["r_min"].asDouble();
        r_max = js["r_max"].asDouble();
        nu = js["nu"].asDouble();
        
        U = js.get("velocity", 1).asDouble();
        R = js["sigma"].asDouble();
        posX = js["posX"].asDouble();
        posY = js["posY"].asDouble();
        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();

    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<nu<<"\n";
        os << "Boundary parameters are: \n"
            <<"    r_min = "<<r_min<<"\n"
            <<"    r_max = "<<r_max<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Initial value parameters are: \n"
            << "    radius:       "<<R<<"\n"
            << "    velocity:     "<<U<<"\n"
            << "    pos_r:         "<<posX<<"\n"
            << "    pos_varphi:    "<<posY<<"\n";
        os << "Tolerance for CG:         "<<eps<<"\n"
           << "Tolerance for time step:  "<<eps_time<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl;
    }
};

