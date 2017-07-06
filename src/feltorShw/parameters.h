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
    double eps_hat;

    double mu[2];
    double tau[2];
    double lx,ly;
    double invkappa;
    double Chat,g;
    double nu_perp,alpha;
    
    double jfactor;
    
    double amp, sigma, posX, posY;
    
    double  nprofileamp, bgprofamp;
    unsigned hwmode,modelmode,cmode,initmode;
    double omega_source,sourceb,sourcew;
    enum dg::bc bc_x,bc_y,bc_x_phi;

    Parameters(const Json::Value& js)        {
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
        jfactor = js["jumpfactor"].asDouble();
        eps_gamma = js["eps_gamma"].asDouble();
        eps_time = js["eps_time"].asDouble();
        eps_hat = 1.;
        mu[0] = js["mu_e"].asDouble();
        mu[1] = 1.;
        tau[0] = -1.;
        tau[1] = js["tau"].asDouble();
        modelmode = js["modelmode"].asUInt();
        cmode = js["cmode"].asUInt();
        nu_perp = js["nu_perp"].asDouble();
        alpha   = js["alpha"].asDouble();
        amp     = js["amplitude"].asDouble();
        sigma   = js["sigma"].asDouble();
        posX    = js["posX"].asDouble();
        posY    = js["posY"].asDouble();
        nprofileamp = js["prof_amp"].asDouble();
        bgprofamp =   js["bg_prof_amp"].asDouble();
        lx =  js["lx"].asDouble();
        ly =  js["ly"].asDouble();
        bc_x = dg::str2bc(js["bc_x"].asString());
        bc_x_phi = dg::str2bc(js["bc_x_phi"].asString());
        bc_y = dg::str2bc(js["bc_y"].asString());
        hwmode =  js["hwmode"].asUInt();
	initmode =  js["initmode"].asUInt();
        invkappa =   js["invkappa"].asDouble();
        Chat = (double)(lx*alpha);
        g = (double) (lx/invkappa);
        omega_source = js["prof_source_rate"].asDouble();
        sourceb = js["source_b"].asDouble();
        sourcew = js["source_damping_width"].asDouble();                    
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"     mu_e                             = "<<mu[0]<<"\n"
            <<"     mu_i                             = "<<mu[1]<<"\n"
            <<"     Full-F/Full-F-boussinesq/delta-f = "<<modelmode<<"\n"
            <<"     El.-temperature:                 = "<<tau[0]<<"\n"
            <<"     Ion-temperature:                 = "<<tau[1]<<"\n"
            <<"     perp. Viscosity:                 = "<<nu_perp<<"\n"
            <<"     alpha:                           = "<<alpha<<"\n"
            <<"     Chat  :                          = "<<Chat<<"\n"
            <<"     g     :                          = "<<g<<"\n"
            <<"     modelmode:                       = "<<modelmode<<"\n"
            <<"     hwmode:                          = "<<hwmode<<"\n"
            <<"     cmode:                           = "<<cmode<<"\n"
	    <<"     initmode:                        = "<<initmode<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Profile parameters are: \n"
            <<"     invkappa:                     = "<<invkappa<<"\n"
            <<"     density profile amplitude:    = "<<nprofileamp<<"\n"
            <<"     background profile amplitude: = "<<bgprofamp<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     dt = "<<dt<<"\n";
        os << "     Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"     Jump scale factor:   "<<jfactor<<"\n"
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
        os << "Boundary conditions in phi in x are: \n"
            <<"    "<<bc2str(bc_x_phi)<<"\n";
        os << "SOL/EDGE/Source params \n"
            <<"     source rate  =    "<<omega_source<<"\n"
            <<"     source boundary = "<<sourceb<<"\n"
            <<"     source width =    "<<sourcew<<"\n";
        os << std::flush;//the endl is for the implicit flush 
    }
};

}//namespace eule

#endif//_DG_PARAMETERS_

    

