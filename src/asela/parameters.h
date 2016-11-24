#pragma once
#include "dg/enums.h"

namespace eule{
/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n; //!< \# of polynomial coefficients in R and Z
    unsigned Nx; //!< \# of cells in x -direction
    unsigned Ny; //!< \# of cells in y -direction
    unsigned Nz; //!< \# of cells in z -direction

    double dt;  //!< timestep
    unsigned n_out;  //!< \# of polynomial coefficients in output file
    unsigned Nx_out;  //!< \# of cells in x-direction in output file
    unsigned Ny_out; //!< \# of cells in y-direction in output file
    unsigned Nz_out; //!< \# of cells in z-direction in output file
    unsigned itstp; //!< \# of steps between outputs
    unsigned maxout; //!< \# of outputs excluding first

    double eps_pol;  //!< accuracy of polarization 
    double eps_maxwell; //!< accuracy of induction equation
    double eps_gamma; //!< accuracy of gamma operator
    double eps_time;//!< accuracy of implicit timestep
    double eps_hat;//!< 1

    double mu[2]; //!< mu[0] = mu_e, m[1] = mu_i
    double tau[2]; //!< tau[0] = -1, tau[1] = tau_i
    double beta; //!< plasma beta

    double nu_perp;  //!< perpendicular diffusion
    double nu_parallel;  //!< parallel diffusion
    double c; //!< parallel resistivity
    
    double amp;  //!< blob amplitude
    double sigma; //!< perpendicular blob width
    double posX;  //!< perpendicular position relative to box width
    double posY; //!< perpendicular position relative to box height
    double sigma_z; //!< parallel blob width in units of pi
    double k_psi; //!< mode number

    double omega_source; //!< source amplitude 
    double nprofileamp; //!< amplitude of profile
    double bgprofamp; //!< background profile amplitude
    double boxscaleRp; //!< box can be larger
    double boxscaleRm;//!< box can be larger
    double boxscaleZp;//!< box can be larger
    double boxscaleZm;//!< box can be larger

    enum dg::bc bc; //!< global perpendicular boundary condition
    unsigned pollim; //!< 0= no poloidal limiter, 1 = poloidal limiter
    unsigned pardiss; //!< 0 = adjoint parallel dissipation, 1 = nonadjoint parallel dissipation
    unsigned mode; //!< 0 = blob simulations (several rounds fieldaligned), 1 = straight blob simulation( 1 round fieldaligned), 2 = turbulence simulations ( 1 round fieldaligned), 
    unsigned initcond; //!< 0 = zero electric potential, 1 = ExB vorticity equals ion diamagnetic vorticity
    unsigned curvmode; //!< 0 = low beta, 1 = toroidal field line 
    unsigned flrmode;
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
            omega_source = v[30];
            boxscaleRp = v[31];
            if(v.size() > 32)
            {
                boxscaleRm = v[32];
                boxscaleZp = v[33];
                boxscaleZm = v[34];
                bc = map((int)v[35]);
                pollim = (unsigned)v[36];
                pardiss = (unsigned)v[37];
                if( v.size() > 38)
                    mode = (unsigned)v[38];
                else mode = 0;
                initcond = (unsigned)v[39];
                curvmode = (unsigned)v[40];
                flrmode = (unsigned)v[41];
            }
            else //to be compatible with older versions
            {
                boxscaleRm = v[31];
                boxscaleZp = v[31];
                boxscaleZm = v[31];
                bc = map( 1);
                pollim = 0;
                pardiss = 0;
                mode = 0;
            }
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
            <<"     omega_source:                 "<<omega_source<<"\n"
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
        os << "Boundary condition is: \n"
            <<"     global BC             =              "<<bc<<"\n"
            <<"     Poloidal limiter      =              "<<pollim<<"\n"
            <<"     Parallel dissipation  =              "<<pardiss<<"\n"
            <<"     Computation mode      =              "<<mode<<"\n"
            <<"     init cond             =              "<<initcond<<"\n"
            <<"     curvature mode        =              "<<curvmode<<"\n"
            <<"     em. FLR mode          =              "<<flrmode<<"\n";
        os << std::flush;
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


    

