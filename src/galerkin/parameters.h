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

    int global;
    double nu, kappa, tau;

    double a_z, mu_z, tau_z;

    double n0, sigma, posX, posY;

    unsigned itstp; 
    unsigned maxout;

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const std::vector< double>& v, int layout = 0) {
        layout_ = layout;
        if( layout == 2) 
        {
            a_z = v[22];
            mu_z = v[23];
            tau_z = v[24];
            layout = 0;
        }
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
            bc_x = map((int)v[10]), bc_y = map((int)v[11]);
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
            Nx = (unsigned)v[1]/v[25];
            Ny = (unsigned)v[2]/v[25]; //reduction parameter v[25]!
            k = 3;
            dt = v[3];
            eps_pol = 1e-6;
            eps_gamma = 1e-10;
            ly = v[4];
            lx = ly/(double)Ny*(double)Nx;
            bc_x = bc_y = dg::PER;
            bc_x = map((int)v[5]);
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
        if( layout_ == 2)
        {
            os <<"    a_z   = "<<a_z<<"\n"
               <<"    mu_z  = "<<mu_z<<"\n"
               <<"    tau_z = "<<tau_z<<"\n";
        }
        char local[] = "LOCAL" , glo[] = "GLOBAL";
        os  <<"Mode is:   \n"
            <<"    "<<(global?glo:local)<<global<<"\n";
        //char per[] = "PERIODIC", dir[] = "DIRICHLET", neu[] = "NEUMANN";
        //char dir_neu[] = "DIR_NEU", neu_dir[] = "NEU_DIR";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n";
        displayBC( os, bc_x, bc_y);
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


    

#endif//_DG_PARAMETERS_
