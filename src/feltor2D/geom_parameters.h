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
           psipmin,
           psipmax,
           a, 
           elongation,
           triangularity,
           alpha,
           rk4eps, 
           psipmaxcut,
           
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
    }
    /**
     * @brief Display parameters
     *
     * @param os Output stream
     */
    void display( std::ostream& os = std::cout ) const
    {
        os << "Geometrical parameters are: \n"
            <<"A             = "<<A<<"\n"
            <<"c1            = "<<c[0]<<"\n"
            <<"c2            = "<<c[1]<<"\n"
            <<"c3            = "<<c[2]<<"\n"
            <<"c4            = "<<c[3]<<"\n"
            <<"c5            = "<<c[4]<<"\n"
            <<"c6            = "<<c[5]<<"\n"
            <<"c7            = "<<c[6]<<"\n"
            <<"c8            = "<<c[7]<<"\n"
            <<"c9            = "<<c[8]<<"\n"
            <<"c10           = "<<c[9]<<"\n"
            <<"c11           = "<<c[10]<<"\n"
            <<"c12           = "<<c[11]<<"\n"
            <<"R0            = "<<R_0<<"\n"
            <<"epsilon_a     = "<<a/R_0<<"\n"
            <<"elongation    = "<<elongation<<"\n"
            <<"triangularity = "<<triangularity<<"\n"
            <<"alpha         = "<<alpha<<"\n"
            <<"rk4 epsilon   = "<<rk4eps<<"\n"
            <<"psipmin       = "<<psipmin<<"\n"
            <<"psipmax       = "<<psipmax<<"\n"            
            <<"psipmaxcut    = "<<psipmaxcut<<"\n"
            
    }
};
} //namespace solovev
