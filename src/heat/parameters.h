#pragma once
#include "dg/enums.h"
#include "json.h"

namespace eule{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny, Nz; 
    double dt; 
    unsigned n_out, Nx_out, Ny_out, Nz_out; 
    unsigned itstp, maxout;
    unsigned p_adv,p_diff,p_diffperp,p_torlim;
    
    double nu_perp, nu_parallel;
    
    double amp, sigma, posX, posY, sigma_z;
    double k_psi; 
    
    double nprofileamp, bgprofamp;
    enum dg::bc bc;
    double boxscaleRp,boxscaleRm,boxscaleZp,boxscaleZm;
    double eps_time;
    Parameters( const Json::Value& js) {
        n  = js["n"].asUInt();
        Nx = js["Nx"].asUInt();
        Ny = js["Ny"].asUInt();
        Nz = js["Nz"].asUInt();
        dt = js["dt"].asDouble();
        n_out  = js["n_out"].asUInt();
        Nx_out = js["Nx_out"].asUInt();
        Ny_out = js["Ny_out"].asUInt();
        Nz_out = js["Nz_out"].asUInt();
        itstp = js["itstp"].asUInt();
        maxout = js["maxout"].asUInt();

        nu_perp     = js["nu_perp"].asDouble();
        nu_parallel = js["nu_parallel"].asDouble();
        amp = js["amplitude"].asDouble();
        sigma = js["sigma"].asDouble();
        posX = js["posX"].asDouble();
        posY = js["posY"].asDouble();
        sigma_z = js["sigma_z"].asDouble();
        k_psi = js["k_psi"].asDouble();

        eps_time = js["eps_time"].asDouble();
        bc = dg::str2bc(js["bc"].asString());
        nprofileamp = js["nprofileamp"].asDouble();
        bgprofamp = js["bgprofamp"].asDouble();
        boxscaleRp = js["boxscaleRp"].asDouble();
        boxscaleRm = js["boxscaleRm"].asDouble();
        boxscaleZp = js["boxscaleZp"].asDouble();
        boxscaleZm = js["boxscaleZm"].asDouble();
        p_adv       =js["adv"].asUInt();
        p_diff      =js["diff"].asUInt();
        p_diffperp  =js["diffperp"].asUInt();
        p_torlim    =js["torlim"].asUInt();
    }

    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"     perp. Viscosity:  = "<<nu_perp<<"\n"
            <<"     par. Viscosity:   = "<<nu_parallel<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n"
            << "    sigma_z:      "<<sigma_z<<"\n";
        os << "Profile parameters are: \n"
            <<"     density profile amplitude:    "<<nprofileamp<<"\n"
            <<"     background profile amplitude: "<<bgprofamp<<"\n"
            <<"     zonal modes                   "<<k_psi<<"\n"
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
        os << "Output parameters are: \n"
            <<"     n_out  =              "<<n_out<<"\n"
            <<"     Nx_out =              "<<Nx_out<<"\n"
            <<"     Ny_out =              "<<Ny_out<<"\n"
            <<"     Nz_out =              "<<Nz_out<<"\n"
            <<"     Steps between output: "<<itstp<<"\n"
            <<"     Number of outputs:    "<<maxout<<"\n";
        os << "Operator parameters are: \n"
            <<"     p_adv  =              "<<p_adv<<"\n"
            <<"     p_diff =              "<<p_diff<<"\n"            
            <<"     p_diffperp =          "<<p_diffperp<<"\n"
            <<"     p_torlim =            "<<p_torlim<<"\n";           
        os << "Boundary condition is: \n"
            <<"     global BC  =              "<<bc2str(bc)<<"\n";
        os << "PCG epsilon for time stepper: \n"
            <<"     eps_time  =              "<<eps_time<<"\n";
        os << std::flush;//the endl is for the implicit flush 
    }
};

}//namespace eule


    

