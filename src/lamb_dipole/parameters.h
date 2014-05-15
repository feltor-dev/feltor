#ifndef _SHU_PARAMETERS_
#define _SHU_PARAMETERS_
#include "dg/grid.cuh"

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, k, Nx, Ny; 
    double dt; 

    double eps;
    double lx, ly; 
    enum dg::bc bc_x, bc_y;

    double D;
    int global;

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
        lx = v[7]; 
        ly = v[8];
        bc_x = dg::PER, bc_y = dg::PER;
        if( v[9]) bc_x = dg::DIR;
        if( v[10]) bc_y = dg::DIR;
        D = v[11];
        U = v[12];
        R = v[13];
        posX = v[14];
        posY = v[15];
        itstp = v[16];
        maxout = v[17];
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
        char per[] = "PERIODIC", dir[] = "DIRICHLET";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n"
            <<"Boundary conditions in x are: \n"
            <<"    "<<(bc_x == dg::DIR ? dir:per)<<"\n"
            <<"Boundary conditions in y are: \n"
            <<"    "<<(bc_y == dg::DIR ? dir:per)<<"\n";
        os << "Algorithmic parameters are: \n"
            <<"    n  = "<<n<<"\n"
            <<"    Nx = "<<Nx<<"\n"
            <<"    Ny = "<<Ny<<"\n"
            <<"    k  = "<<k<<"\n"
            <<"    dt = "<<dt<<"\n";
        os  <<"Dipole parameters are: \n"
            << "    radius:       "<<R<<"\n"
            << "    velocity:     "<<U<<"\n"
            << "    posX:         "<<posX<<"\n"
            << "    posY:         "<<posY<<"\n";
        os << "Stopping for CG:         "<<eps<<"\n"
            <<"Steps between output:    "<<itstp<<"\n"
            <<"Number of outputs:       "<<maxout<<std::endl;
    }
};


    

#endif//_DG_PARAMETERS_
