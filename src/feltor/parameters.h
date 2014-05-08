#pragma once
#include "dg/grid.cuh"

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny, Nz; 
    double dt; 

    double eps_pol, eps_gamma, eps_time;
    double thickness, a, R_0; 

    double lnn_inner;
    double nu_perp, nu_parallel, c, mcv, tau_i;

    double mu_e;

    double amp, sigma, posX, posY;
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
            eps_gamma = v[7];
            eps_time = v[8];
            thickness = v[9];
            a = v[10];
            R_0 = v[11];
            lnn_inner = v[12];
            mu_e = v[13];
            tau_i = v[14];
            mcv = v[15];
            nu_perp = v[16];
            nu_parallel = v[17];
            c = v[18];
            
            amp = v[19];
            sigma = v[20];
            posX = v[21];
            posY = v[22];
            m_par = v[23];
            itstp = v[24];
            maxout = v[25];
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
            <<"    mu_e             = "<<mu_e<<"\n"
            <<"    Ion-temperature: = "<<tau_i<<"\n"
            <<"    perp Viscosity:  = "<<nu_perp<<"\n"
            <<"    perp Resistivity:= "<<c<<"\n"
            <<"    par Viscosity:   = "<<nu_parallel<<"\n"
            <<"    magnetic curvature:  = "<<mcv<<"\n";
        os << "Boundary parameters are: \n"
            <<"    Ring thickness = "<<thickness<<"\n"
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


    

