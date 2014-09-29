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
           lnN_inner,
           k_psi,
           rk4eps, 
           boxscale,
           nprofileamp,
           bgprofamp,
           psipmaxcut,
           psipmaxlap; //only for inverse tanh damping profile
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
        psipmin= v[15];
        psipmax= v[16];
        a=R_0*v[17];
        elongation=v[18];
        triangularity=v[19];
        alpha=v[20];
        lnN_inner=v[21];
        k_psi=v[22];
        rk4eps=v[23];
        boxscale=v[24];
        nprofileamp=v[25];
        bgprofamp=v[26];
        psipmaxcut = v[27];
        psipmaxlap = v[28];
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
            <<"psipmin       = "<<psipmin<<"\n"
            <<"psipmax       = "<<psipmax<<"\n"
            <<"epsilon_a     = "<<a/R_0<<"\n"
            <<"elongation    = "<<elongation<<"\n"
            <<"triangularity = "<<triangularity<<"\n"
            <<"alpha         = "<<alpha<<"\n"
            <<"lnN_inner     = "<<lnN_inner<<"\n"
            <<"zonal modes   = "<<k_psi<<"\n" 
            <<"rk4 epsilon   = "<<rk4eps<<"\n"
            <<"boxscale      = "<<boxscale<<"\n"
            <<"nprofileamp   = "<<nprofileamp<<"\n"
            <<"bgprofamp     = "<<bgprofamp<<"\n"
            <<"psipmaxcut    = "<<psipmaxcut<<"\n"
            <<"psipmaxlap    = "<<psipmaxlap<<"\n"; 
    }
};
} //namespace solovev
