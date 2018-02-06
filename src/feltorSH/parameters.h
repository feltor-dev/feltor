#ifndef _DG_PARAMETERS_ 
#define _DG_PARAMETERS_
#include <string>
#include "dg/enums.h"
#include "json/json.h"

namespace eule{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny; 
    double dt; 
    unsigned n_out, Nx_out, Ny_out; 
    unsigned itstp, maxout;

    double eps_pol,  eps_gamma, eps_time;

    double mu[2];
    double tau[2];
    double mcv;
    double lx,ly;
    double nu_perp;
    
    double amp, sigma, posX, posY;

    double  nprofileamp, bgprofamp;
    unsigned init, iso, flrmode;
    enum dg::bc bc_x,bc_y; 

    /**
     * @brief constructor to make a const object
     *
     * @param js json object
     */
    Parameters(const Json::Value& js) {
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
        mu[0]   = -0.000272121;
        mu[1]   = 1.;
        tau[0]  = -1.;
        tau[1]  = js["tau"].asDouble();
        mcv     = js["curvature"].asDouble();
        nu_perp = js["nu_perp"].asDouble();
        amp     = js["amplitude"].asDouble();
        sigma   = js["sigma"].asDouble();
        posX    = js["posX"].asDouble();
        posY    = js["posY"].asDouble();
        nprofileamp = 0.;
        bgprofamp   = 1.;
        lx = js["lx"].asDouble();
        ly =  js["ly"].asDouble();
        bc_x = dg::str2bc(js["bc_x"].asString());
        bc_y = dg::str2bc(js["bc_y"].asString());
        init = js["initmode"].asUInt();
        iso =  js["tempmode"].asUInt();
        flrmode =  js["flrmode"].asUInt();            
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
            <<"     mcv               = "<<mcv<<"\n"
            <<"     El.-temperature:  = "<<tau[0]<<"\n"
            <<"     Ion-temperature:  = "<<tau[1]<<"\n"
            <<"     perp. Viscosity:  = "<<nu_perp<<"\n"
            <<"     eff grav./diss f. = "<<(1.+tau[1])*sigma*sigma*sigma*mcv*amp*(2.+amp)/(nu_perp*nu_perp)<<"\n"
            <<"     cst/dyn FLR (0/1) = "<<flrmode<<"\n"
            <<"     isothermal (0/1)  = "<<iso<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Profile parameters are: \n"
            <<"     density profile amplitude:    "<<nprofileamp<<"\n"
            <<"     background profile amplitude: "<<bgprofamp<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     dt = "<<dt<<"\n";
        os << "     Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"     Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"     Stopping for Time  CG:   "<<eps_time<<"\n";
        os << "Output parameters are: \n"
            <<"     n_out  =              "<<n_out<<"\n"
            <<"     Nx_out =              "<<Nx_out<<"\n"
            <<"     Ny_out =              "<<Ny_out<<"\n"
            <<"     Steps between output: "<<itstp<<"\n"
            <<"     Number of outputs:    "<<maxout<<"\n";
        os << "Box params: \n"
            <<"     lx  =              "<<lx<<"\n"
            <<"     ly  =              "<<ly<<"\n"; 
        os << "Boundary conditions in x are: \n"
            <<"    "<<bc2str(bc_x)<<"\n"; 
        os << "Boundary conditions in y are: \n"
            <<"    "<<bc2str(bc_y)<<"\n";
        os << std::flush;//the endl is for the implicit flush 
    }
};

}//namespace eule

#endif//_DG_PARAMETERS_

    

