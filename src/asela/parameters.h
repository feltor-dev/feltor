#pragma once
#include "dg/backend/grid.h"
<<<<<<< HEAD

=======
#include "dg/enums.h"
>>>>>>> 69b95fe1d05493e5e4cb97e206edbf72f11aab09

namespace eule
{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, Nx, Ny, Nz; 
    double dt; 

    double eps_pol, eps_maxwell, eps_time;
    double lxhalf, lyhalf; 

    double nu;

    double dhat[2];
    double rhohat[2];
    double tau[2];

    double amp;

    unsigned itstp; 
    unsigned maxout;
    dg::bc bcx, bcy;

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
            eps_time = v[8];
            lxhalf = v[9]*M_PI;
            lyhalf = v[10]*M_PI;
            bcx = map((int)v[11]), bcy = map((int)v[12]);
            dhat[0] = v[13];
            dhat[1] = v[14];
            rhohat[0] = v[15];
            rhohat[1] = v[16];
            nu  = v[17];
            
            amp = v[19];
            itstp = v[20];
            maxout = v[21];
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
            <<"    rhos  = "<<rhohat[0]<<"\n"
            <<"    rhoi  = "<<rhohat[1]<<"\n"
            <<"    de: = "<<dhat[0]<<"\n"
            <<"    di: = "<<dhat[1]<<"\n"
            <<"    Viscosity:       = "<<nu<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<2.*lxhalf<<"\n"
            <<"    ly = "<<2.*lyhalf<<"\n";
        displayBC(os, bcx, bcy);

        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    Nz = "<<Nz<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Perturbation parameters are: \n"
            << "    amplitude:    "<<amp<<"\n";
        os << "Stopping for Polar CG:   "<<eps_pol<<"\n"
            <<"Stopping for Aparl CG:   "<<eps_maxwell<<"\n"
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
} //namespace eule


    

