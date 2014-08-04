#pragma once
#include "dg/enums.h"

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny, Nz; 
    double dt; 

    double eps_pol, eps_maxwell, eps_gamma, eps_time;

    double a, b, R_0; 
    double damping_width, damping_strength;
    double eps_hat;

    double lnn_inner;
    double nu_perp, nu_parallel, c;

    double mu[2];
    double tau[2];
    double beta;
    
    double amp, sigma, posX, posY;
    double amp_source;
    double m_par;

    unsigned itstp; 
    unsigned maxout;

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
            eps_pol = v[6];
            eps_maxwell = v[7];
            eps_gamma = v[8];
            eps_time = v[9];
            b = v[10];
            a = v[11];
            assert( a>b && "Source radius must be smaller than minor radius!" );
            R_0 = v[12];
            eps_hat = 1.;//4.*M_PI*M_PI*R_0*R_0;
            lnn_inner = v[13];
            mu[0] = v[14];
            mu[1] = 1.;
            tau[0] = -1.;
            tau[1] = v[15];
            beta = v[16];
            nu_perp = v[17];
            nu_parallel = v[18];
            c = v[19];
            
            amp = v[20];
            sigma = v[21];
            posX = v[22];
            posY = v[23];
            m_par = v[24];
            itstp = v[25];
            maxout = v[26];
            damping_width    = v[27];
            damping_strength = v[28];
            amp_source = v[29];
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
            <<"    mu_e             = "<<mu[0]<<"\n"
            <<"    mu_i             = "<<mu[1]<<"\n"
            <<"    beta             = "<<beta<<"\n"
            <<"Electron-temperature: = "<<tau[0]<<"\n"
            <<"    Ion-temperature:  = "<<tau[1]<<"\n"
            <<"    perp Viscosity:   = "<<nu_perp<<"\n"
            <<"    perp Resistivity: = "<<c<<"\n"
            <<"    par Viscosity:    = "<<nu_parallel<<"\n";
        os << "Boundary parameters are: \n"
            <<"    Ring thickness = "<<a-b<<"\n"
            <<"    minor Radius a = "<<a<<"\n"
            <<"    major Radius R = "<<R_0<<"\n"
            <<"    inner density ln n = "<<lnn_inner<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    Nz = "<<Nz<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Blob parameters are: \n"
            << "    amplitude:    "<<amp<<"\n"
            << "    width:        "<<sigma<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"Stopping for Time  CG:   "<<eps_time<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl; //the endl is for the implicit flush 
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


    

