#ifndef _POLAR_PARAMETERS_
#define _POLAR_PARAMETERS_
#include "dg/backend/grid.h"

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, k, Nx, Ny; 
    double dt; 

    double eps;
    double r_min, r_max;

    double D;
    int global;

    unsigned iv;
    double U, R, posX, posY;

    unsigned itstp; 
    unsigned maxout;

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const std::vector< double>& v, int layout=0) {
        n  = (unsigned)v[1];
        Nx = (unsigned)v[2];
        Ny = (unsigned)v[3];
        k  = (unsigned)v[4];
        dt = v[5];
        eps = v[6];
        r_min = v[7];
        r_max = v[8];
        D = v[9];
        iv = (unsigned)v[10];
        U = v[11];
        R = v[12];
        posX = v[13];
        posY = v[14];
        itstp = v[15];
        maxout = v[16];
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Physical parameters are: \n"
            <<"    Viscosity:       = "<<D<<"\n";
        os << "Boundary parameters are: \n"
            <<"    r_min = "<<r_min<<"\n"
            <<"    r_max = "<<r_max<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    k  = "<<k<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Initial value parameters are: \n"
            << "    iv:           "<<iv<<"\n"
            << "    radius:       "<<R<<"\n"
            << "    velocity:     "<<U<<"\n"
            << "    pos_r:         "<<posX<<"\n"
            << "    pos_varphi:    "<<posY<<"\n";
        os << "Stopping for CG:         "<<eps<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl;
    }
};


    

#endif//_DG_PARAMETERS_
