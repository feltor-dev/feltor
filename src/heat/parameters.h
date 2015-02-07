#pragma once
#include "dg/enums.h"

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

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const std::vector< double>& v):layout_(0) {
        if( layout_ == 0)
        {
            n  = (unsigned)v[1]; 
            Nx = (unsigned)v[2];
            Ny = (unsigned)v[3];
            Nz = (unsigned)v[4];
            dt = v[5];
            n_out = v[6];
            Nx_out = v[7];
            Ny_out = v[8];
            Nz_out = v[9];
            itstp = v[10];
            maxout = v[11];
            nu_perp = v[12];
            nu_parallel = v[13];
            amp = v[14];
            sigma = v[15];
            posX = v[16];
            posY = v[17];
            sigma_z = v[18];
            k_psi = v[19];
            nprofileamp = v[20];
            bgprofamp = v[21];
            bc = map((int)v[22]);
            boxscaleRp = v[23];
            boxscaleRm = v[24];
            boxscaleZp = v[25];
            boxscaleZm = v[26];
            p_adv       =(unsigned) v[27];
            p_diff      =(unsigned) v[28];
            p_diffperp  =(unsigned) v[29];
            p_torlim    =(unsigned) v[30];

        }
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
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
            <<"     p_torlim =          "<<p_torlim<<"\n";           
        os << "Boundary condition is: \n"
            <<"     global BC  =              "<<bc<<"\n";
        os << std::flush;//the endl is for the implicit flush 
    }
    private:
    int layout_;
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
    void displayBC( std::ostream& os, dg::bc bc) const
    {
        os << "Boundary conditions  are: \n";
        switch( bc)
        {
            case(0): os << "    PERIODIC";
                     break;
            case(1): os << "    DIRICHLET";
                     break;
            case(2): os << "    DIR_NEU";
                     break;
            case(3): os << "    NEU_DIR";
                     break;
            case(4): os << "    NEUMANN";
                     break;
        }

        os <<"\n";
    }
};

}//namespace eule


    

