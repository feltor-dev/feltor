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

    double eps_pol, eps_maxwell, eps_gamma, eps_time;
    double eps_hat;

    double mu[2];
    double tau[2];
    double beta;

    double nu_perp, nu_parallel, c;
    
    double amp, sigma, posX, posY, sigma_z;
    double k_psi; 

    double amp_source, boxscale,boxlimscale, nprofileamp, bgprofamp;

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
            eps_pol = v[12];
            eps_maxwell = v[13];
            eps_gamma = v[14];
            eps_time = v[15];
            eps_hat = 1.;
            mu[0] = v[16];
            mu[1] = 1.;
            tau[0] = -1.;
            tau[1] = v[17];
            beta = v[18];
            nu_perp = v[19];
            nu_parallel = v[20];
            c = v[21];            
            amp = v[22];
            sigma = v[23];
            posX = v[24];
            posY = v[25];
            sigma_z = v[26];
            k_psi = v[27];
            nprofileamp = v[28];
            bgprofamp = v[29];
            amp_source = v[30];
            boxscale = v[31];
            boxlimscale = v[32];
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
            <<"     Source strength:              "<<amp_source<<"\n"
            <<"     density profile amplitude:    "<<nprofileamp<<"\n"
            <<"     background profile amplitude: "<<bgprofamp<<"\n"
            <<"     boxscale:                     "<<boxscale<<"\n"
            <<"     boxlimscale:                  "<<boxlimscale<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"     n  = "<<n<<"\n"
            <<"     Nx = "<<Nx<<"\n"
            <<"     Ny = "<<Ny<<"\n"
            <<"     Nz = "<<Nz<<"\n"
            <<"     dt = "<<dt<<"\n";
        os << "     Stopping for Polar CG:   "<<eps_pol<<"\n"
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
    void displayBC( std::ostream& os, dg::bc bcx, dg::bc bcy) const
    {
        os << "Boundary conditions in x are: \n";
        switch( bcx)
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
        os << "\nBoundary conditions in y are: \n";
        switch( bcy)
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


    

