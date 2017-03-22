#pragma once
#include <string>
#include <vector>
#include "json/json.h"
/*!@file
 *
 * Geometry parameters
 */
namespace dg
{
namespace geo
{
namespace solovev
{
/**
 * @brief Constructs and display geometric parameters for the solovev and taylor fields
 * @ingroup geom
 */    
struct GeomParameters
{
    double A, //!< A
           R_0, //!< central tokamak radius
           a,  //!<  little tokamak radius
           elongation, //!< elongation of the magnetic surfaces
           triangularity, //!< triangularity of the magnetic surfaces
           alpha, //!< damping width
           rk4eps,  //!< accuracy for the field line integration
           psipmin, //!< for source 
           psipmax, //!< for profile
           psipmaxcut, //!< for cutting
           psipmaxlim,  //!< for limiter
           qampl; //scales grad-shafranov q factor
    std::vector<double> c;  //!< coefficients for the solovev equilibrium
    std::string equilibrium;
     /**
     * @brief constructor to make an object
     *
     * maps parameters from input file to parameters 
     * @param v Vector from read_input function
     */   
    GeomParameters( const std::vector< double>& v) {
        A=v[1];
        c.resize(13);//there are only 12 originially c[12] is to make fieldlines straight
        for (unsigned i=0;i<12;i++) c[i]=v[i+2];
        c[12] = 0;
        if( A!=0) c[12] = 1;
        for( unsigned i=0; i<12; i++)
            if(c[i]!=0) c[12] = 1.;
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
        qampl = v[24];
    }
    GeomParameters( const Json::Value& js) {
        A  = js.get("A", 0).asDouble();
        c.resize(13);//there are only 12 originially c[12] is to make fieldlines straight
        for (unsigned i=0;i<12;i++) c[i] = js["c"][i].asDouble();
        c[12] = 0;
        if( A!=0) c[12] = 1;
        for( unsigned i=0; i<12; i++)
            if(c[i]!=0) c[12] = 1.;
        R_0  = js["R_0"].asDouble();
        a  = R_0*js["inverseaspectratio"].asDouble();
        elongation=js["elongation"].asDouble();
        triangularity=js["triangularity"].asDouble();
        alpha=js["alpha"].asDouble();
        rk4eps=js["rk4eps"].asDouble();
        psipmin= js["psip_min"].asDouble();
        psipmax= js["psip_max"].asDouble();
        psipmaxcut= js["psip_max_cut"].asDouble();
        psipmaxlim= js["psip_max_lim"].asDouble();
        qampl = js.get("qampl", 1.).asDouble();
        equilibrium = js.get( "equilibrium", "solovev").asString();
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
        for( unsigned i=0; i<13; i++)
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
            <<" psipmaxlim    = "<<psipmaxlim<<"\n"
            <<" qampl    = "<<qampl<<"\n";
        os << std::flush;

    }
};
} //namespace solovev
} //namespace geo
} //namespace dg
