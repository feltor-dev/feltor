#ifndef _BLUEPRINT_
#define _BLUEPRINT_

#include <iostream>
#include <cmath>
#include "toefl/ghostmatrix.h" // holds boundary conditions
#include "toefl/message.h"

namespace toefl{
/*! @addtogroup parameters
 * @{
 */
/*! @brief Possible capacities of a toefl solver
 */
enum cap{   IMPURITY, //!< Include impurities
            GLOBAL, //!< Solve global equations
            MHW //!< Modify parallel term in electron density equation
};

/*! @brief Possible targets for memory buffer
 */
enum target{ 
    ELECTRONS, //!< Electron density
    IONS, //!< Ion density
    IMPURITIES, //!< Impurity density
    POTENTIAL, //!< Potential
    ALL //!< all buffers
};

/**
 * @brief Provide a mapping between input file and named parameters
 */
struct Parameters
{
    unsigned nx, ny; 
    double h;
    double dt; 

    unsigned itstp, maxout;

    double nu_perp, nu_parallel, c;
    
    double amp, imp_amp; //
    double blob_width, posX, posY;

    double d, nu, kappa, g_e, g[2], a[2], mu[2], tau[2];
    double lx, ly; 
    enum bc bc_x;
    bool imp, global, mhw;

    /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */
    Parameters( const std::vector< double>& para) {
        imp = global = mhw = false;
        nx = para[1];
        ny = para[2];
        dt = para[3];

        ly = para[4];
        h = ly/(double)ny;
        lx = h*(double)nx;
        switch( (unsigned)para[5])
        {
            case( 0): bc_x = TL_PERIODIC; break;
            case( 1): bc_x = TL_DST10; break;
            case( 2): bc_x = TL_DST01; break;
        }
        if( para[6])
            mhw = true;

        d = para[7];
        nu = para[8];
        kappa = para[9];
        amp = para[10];

        g_e = g[0] = para[11];
        tau[0] = para[12];
        if( para[13])
        {
            imp = true;
            imp_amp = para[14];
            g[1] = para[15];
            a[1] = para[16];
            mu[1] = para[17];
            tau[1] = para[18];
        }
        else 
            g[1] = a[1] = mu[1] = tau[1] = 0;

        a[0] = 1. -a[1];
        g[0] = (g_e - a[1] * g[1])/(1.-a[1]);
        mu[0] = 1.0;//single charged ions
        itstp = para[19];
        blob_width = para[21];
        maxout = para[22];
        posX = para[23];
        posY = para[24];
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout) const
    {
        os << "Physical parameters are: \n"
            <<"    Coupling:        = "<<d<<"\n"
            <<"    Viscosity:       = "<<nu<<"\n"
            <<"    Curvature_y:     = "<<kappa<<"\n"
            <<"   Species/Parameter   g\ta\tmu\ttau\n"
            <<"    Electrons:         "<<g_e  <<"\t"<<"-1"<<"\t"<<"0"<<"\t"<<"1\n"
            <<"    Ions:              "<<g[0] <<"\t"<<a[0]<<"\t"<<mu[0]<<"\t"<<tau[0]<<"\n"
            <<"    Impurities:        "<<g[1] <<"\t"<<a[1]<<"\t"<<mu[1]<<"\t"<<tau[1]<<"\n";
        os << "Boundary parameters are: \n"
            <<"    lx = "<<lx<<"\n"
            <<"    ly = "<<ly<<"\n"
            <<"Boundary condition in x is: \n";
        switch(bc_x)
        {
            case(TL_PERIODIC): os << "    PERIODIC \n"; break;
            case(   TL_DST00): os << "    DST-1-like \n"; break;
            case(   TL_DST10): os << "    DST-2-like \n"; break;
            case(   TL_DST01): os << "    DST-3-like \n"; break;
            case(   TL_DST11): os << "    DST-4-like \n"; break;
            default: os << "    Not specified!!\n"; 
        }
        os << "Algorithmic parameters are: \n"
            <<"    nx = "<<nx<<"\n"
            <<"    ny = "<<ny<<"\n"
            <<"    h  = "<<h<<"\n"
            <<"    dt = "<<dt<<"\n";
        char enabled[] = "ENABLED", disabled[] = "DISABLED";

        os << "Impurities are: \n"
            <<"    "<<(imp?enabled:disabled)<<"\n"
            //<<"Global solvers are: \n"
            //<<"    "<<(global?enabled:disabled)<<"\n"
            <<"Modified Hasegawa Wakatani: \n"
            <<"    "<<(mhw?enabled:disabled)<<std::endl;
        os << std::flush;//the endl is for the implicit flush 
    }
    void consistencyCheck() const{
        //Check algorithm and boundaries
        if( dt <= 0) 
            throw Message( "dt <= 0!\n", _ping_);
        if( fabs( h - lx/(double)nx) > 1e-15) 
            throw Message( "h != lx/nx\n", _ping_); 
        if( fabs( h - ly/(double)ny) > 1e-15) 
            throw Message( "h != ly/ny\n", _ping_);
        if( nx == 0||ny == 0) 
            throw Message( "Set nx and ny!\n", _ping_);
        //Check physical parameters
        if( nu < 0) 
            throw Message( "nu < 0!\n", _ping_);
        if( a[0] <= 0 || mu[0] <= 0 || tau[0] < 0) 
            throw Message( "Ion species badly set\n", _ping_);
        if( imp && (a[1] < 0 || mu[1] <= 0 || tau[1] < 0)) 
            throw Message( "Impuritiy species badly set\n", _ping_);
        if( fabs(a[0] + a[1] - 1) > 1e-15)
            throw Message( "a_i + a_z != 1 (background not neutral)\n", _ping_);
        if( fabs( g_e - a[0]*g[0]- a[1]*g[1]) > 1e-15)
            throw Message( "Background is not neutral! \n", _ping_);
        //inconsistency when impurities are not set??
        if( !imp && (a[1] != 0 || mu[1] != 0 || tau[1] != 0)) 
            throw Message( "Impurity disabled but z species not 0!\n", _ping_);
        //Some Warnings
        if( global && (g_e != 0||g[0] != 0||g[1] != 0))
            std::cerr << "TL_WARNING: Global solver ignores gradients\n";

    }
    private:
};






} //namespace toefl

#endif //_BLUEPRINT_
