
#pragma once
#include "dg/enums.h"

namespace imp
{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny; 
    double dt; 

    double eps_pol, eps_gamma;

    double lx, ly; 
    enum dg::bc bc_x, bc_y;

    double nu, kappa;

    double a[3], mu[3], tau[3];

    double amp, sigma, posX, posY;

    unsigned itstp; 
    unsigned maxout;

    unsigned mode;
    unsigned wall_pos, wall_amp, wall_sigma;

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
        eps_pol = v[5];
        eps_gamma = v[6];
        lx = v[7]; 
        ly = v[8];
        bc_x = map((int)v[9]), bc_y = map((int)v[10]);
        nu = v[11];
        kappa = v[12];
        tau[1] = v[13]; 
        amp = v[14];
        sigma = v[15];
        posX = v[16];
        posY = v[17];
        itstp = v[18];
        maxout = v[19];

        a[2] = v[20];
        mu[2] = v[21];
        tau[2] = v[22];

        a[0] = -1, a[1] = 1-a[2];
        mu[0] = 0, mu[1] = 1;
        tau[0] = -1;
        mode = v[23];
        wall_pos = v[24];
        wall_amp = v[25];
        wall_sigma = v[26];

    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"    Ion-temperature: = "<<tau[1]<<"\n";
        os <<"    a_z   = "<<a[2]<<"\n"
           <<"    mu_z  = "<<mu[2]<<"\n"
           <<"    tau_z = "<<tau[2]<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n";
        displayBC( os, bc_x, bc_y);
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Blob parameters are: \n"
            << "    width:        "<<sigma<<"\n"
            << "    amplitude:    "<<n0<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Stopping for CG:         "<<eps_pol<<"\n"
            <<"Stopping for Gamma CG:   "<<eps_gamma<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl; //the endl is for the implicit flush 
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
}//namespace imp


    

