#pragma once
#include "dg/enums.h"

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
    unsigned init, iso;
    enum dg::bc bc_x,bc_y; 

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const std::vector< double>& v) {
        n  = (unsigned)v[1]; 
        Nx = (unsigned)v[2];
        Ny = (unsigned)v[3];
        dt = v[4];
        n_out = v[5];
        Nx_out = v[6];
        Ny_out = v[7];
        itstp = v[8];
        maxout = v[9];
        eps_pol = v[10];
        eps_gamma = v[11];
        eps_time = v[12];
        mu[0] = -0.000272121;
        mu[1] = 1.;
        tau[0] = -1.;
        tau[1]  = v[13];
        mcv     = v[14];
        nu_perp = v[15];
        amp     = v[16];
        sigma   = v[17];
        posX    = v[18];
        posY    = v[19];
        nprofileamp = 0.;
        bgprofamp   = 1.;
        lx = v[20];
        ly = v[21];
        bc_x = map((int)v[22]);
        bc_y = map((int)v[23]);
        init = v[24];
        iso =  v[25];
            
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
            <<"     eff grav./diss f. = "<<(1.+tau[1])*sigma*sigma*sigma*mcv*amp/(nu_perp*nu_perp)<<"\n"
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
            <<"     ly  =              "<<ly<<"\n"
            <<"     bcx =              "<<bc_x<<"\n"
            <<"     bcy =              "<<bc_y<<"\n";
        os << std::flush;//the endl is for the implicit flush 
    }
    private:
    dg::bc map( int i)
    {
        switch( i)
        {
            case(0): return dg::PER;
            case(1): return dg::DIR;
            case(2): return dg::DIR_NEU;
            case(3): return dg::NEU_DIR;
            case(4): return dg::NEU;
            default: return dg::PER;
        }
    }

};

}//namespace eule


    

