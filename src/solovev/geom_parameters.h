#pragma once
#include <vector>
/*!@file
 *
 * Geometry parameters
 */
namespace solovev
{
///@addtogroup geom
///@{
/**
 * @brief Constructs and display geometric parameters
 */    
struct GeomParameters
{
    double A,
           R_0,
           a, 
           elongation,
           triangularity,
           alpha, //for damping width
           rk4eps, 
           psipmin, //for source ??
           psipmax, //for profile
           psipmaxcut, //for cutting
           psipmaxlim; //for limiter
    std::vector<double> c; 
     /**
     * @brief constructor to make a const object
     *
     * @param v Vector from read_input function
     */   
    GeomParameters( const std::vector< double>& v) {
        A=v[1];
        c.resize(13);
        for (unsigned i=0;i<12;i++) c[i]=v[i+2];
        R_0 = v[14];
        a=R_0*v[15];
        elongation=v[16];
        triangularity=v[17];
        alpha=v[18];
        rk4eps=v[19];
        psipmin= v[20];
        psipmax= v[21];
        psipmaxcut = v[22];
        psipmaxlim = v[23];
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Geometrical parameters are: \n"
            <<" A             = "<<A<<"\n";
        for( unsigned i=0; i<12; i++)
            os<<" c"<<i+1<<"\t\t = "<<c[i]<<"\n";

        os  <<" R0            = "<<R_0<<"\n"
            <<" epsilon_a     = "<<a/R_0<<"\n"
            <<" elongation    = "<<elongation<<"\n"
            <<" triangularity = "<<triangularity<<"\n"
            <<" alpha         = "<<alpha<<"\n"
            <<" rk4 epsilon   = "<<rk4eps<<"\n"
            <<" psipmin       = "<<psipmin<<"\n"
            <<" psipmax       = "<<psipmax<<"\n"
            <<" psipmaxcut    = "<<psipmaxcut<<"\n"
            <<" psipmaxlim    = "<<psipmaxlim<<"\n";
        os << std::flush;

    }
};
} //namespace solovev
