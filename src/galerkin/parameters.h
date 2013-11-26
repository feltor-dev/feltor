#ifndef _DG_PARAMETERS_
#define _DG_PARAMETERS_
#include "dg/grid.cuh"

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned n, k, Nx, Ny; 
    double dt; 

    double eps_pol, eps_gamma;
    double lx, ly; 
    enum dg::bc bc_x, bc_y;

    bool global;
    double nu, kappa, tau;

    double n0, sigma, posX, posY;

    unsigned itstp; 
    unsigned maxout;

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const std::vector< double>& v, int layout = 0) {
        if( layout == 0)
        {
            n  = (unsigned)v[1]; 
            Nx = (unsigned)v[2];
            Ny = (unsigned)v[3];
            k  = (unsigned)v[4];
            dt = v[5];
            eps_pol = v[6];
            eps_gamma = v[7];
            lx = v[8]; 
            ly = v[9];
            bc_x = dg::PER, bc_y = dg::PER;
            if( v[10]) bc_x = dg::DIR;
            if( v[11]) bc_y = dg::DIR;
            global = v[12];
            nu = v[13];
            kappa = v[14];
            tau = v[15]; 
            n0 = v[16];
            sigma = v[17];
            posX = v[18];
            posY = v[19];
            itstp = v[20];
            maxout = v[21];
        }
        else if( layout == 1)
        {
            n = 1;
            Nx = (unsigned)v[1];
            Ny = (unsigned)v[2];
            k = 3;
            dt = v[3];
            eps_pol = 1e-6;
            eps_gamma = 1e-10;
            ly = v[4];
            lx = ly/(double)Ny*(double)Nx;
            bc_x = bc_y = dg::PER;
            if( v[5]) bc_x = dg::DIR;
            global = 0;
            nu = v[8];
            kappa = v[9];
            tau = v[12];
            n0 = v[10];
            sigma = v[21];
            posX = v[23];
            posY = v[24];
            itstp = v[19];
            maxout = v[22];
        }
        else ;
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
            <<"    Ion-temperature: = "<<tau<<"\n";
        char local[] = "LOCAL" , glo[] = "GLOBAL";
        os  <<"Mode is:   \n"
            <<"    "<<(global?glo:local)<<"\n";
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
};


    

#endif//_DG_PARAMETERS_
