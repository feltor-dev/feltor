#pragma once
#include <string>
#include <vector>
#ifdef JSONCPP_VERSION_STRING
#include <dg/file/json_utilities.h>
#endif
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
 * @note include \c json/json.h before \c geometries.h in order to activate json functionality
 */
struct Parameters
{
    double A, //!< A coefficient
           R_0, //!< major tokamak radius
           pp, //!< prefactor for Psi_p
           pi, //!< prefactor for current I
           a,  //!<  little tokamak radius
           elongation, //!< elongation of the magnetic surfaces
           triangularity; //!< triangularity of the magnetic surfaces
    std::vector<double> c;  //!< 12 coefficients for the solovev equilibrium;
    std::string equilibrium;
#ifdef JSONCPP_VERSION_STRING
    /**
     * @brief Construct from Json dataset
     * @param js Can contain the variables "A" (0), "c" (0), "PP" (1.), "PI"
     * (1.), "R_0" , "inverseaspectratio" , "elongation" (1), "triangularity"
     * (0), "equilibrium" ("solovev")
     * @param mode determine what happens when a key is missing
     * @note the default values in brackets are taken if the variables are not found in the input file
     * @attention This Constructor is only defined if \c json/json.h is included before \c dg/geometries/geometries.h
     */
    Parameters( const Json::Value& js, file::error mode = file::error::is_silent) {
        A  = file::get( mode, js, "A", 0).asDouble();
        pp  = file::get( mode, js, "PP", 1).asDouble();
        pi  = file::get( mode, js, "PI", 1).asDouble();
        c.resize(12);
        for (unsigned i=0;i<12;i++)
            c[i] = file::get_idx( mode, js, "c", i, 0.).asDouble();

        R_0  = file::get( mode, js, "R_0", 0.).asDouble();
        a  = R_0*file::get( mode, js, "inverseaspectratio", 0.).asDouble();
        elongation=file::get( mode, js, "elongation", 1.).asDouble();
        triangularity=file::get( mode, js, "triangularity", 0.).asDouble();
        equilibrium = file::get( mode, js, "equilibrium", "solovev").asString();
    }
    /**
     * @brief Put values into a json string
     *
     * @return Json value
     * @attention This member is only defined if \c json/json.h is included before \c dg/geometries/geometries.h
     */
    Json::Value dump( ) const
    {
        Json::Value js;
        js["A"] = A;
        js["PP"] = pp;
        js["PI"] = pi;
        for (unsigned i=0;i<12;i++) js["c"][i] = c[i];
        js["R_0"] = R_0;
        js["inverseaspectratio"] = a/R_0;
        js["elongation"] = elongation;
        js["triangularity"] = triangularity;
        js[ "equilibrium"] = equilibrium;
        return js;
    }
#endif // JSONCPP_VERSION_STRING
    /**
    * @brief True if any coefficient \c c_i!=0 with \c 7<=i<12
    *
    * The Xpoint is situated close to
     <tt> R_X = R_0-1.1*triangularity*a</tt>
     <tt> Z_X = -1.1*elongation*a</tt>
    *
    * @return \c true if Psip has an Xpoint, \c false else
    */
    bool hasXpoint( ) const{
        bool Xpoint = false;
        for( int i=7; i<12; i++)
            if( fabs(c[i]) >= 1e-10)
                Xpoint = true;
        return Xpoint;
    }
    /**
    * @brief True if \c pp==0
    *
    * @return \c true if the flux function is a constant
    */
    bool isToroidal() const{
        if( pp == 0)
            return true;
        return false;
    }
    ///Write variables as a formatted string
    void display( std::ostream& os = std::cout ) const
    {
        os << "Geometrical parameters are: \n"
            <<" A               = "<<A<<"\n"
            <<" Prefactor Psi   = "<<pp<<"\n"
            <<" Prefactor I     = "<<pi<<"\n";
        for( unsigned i=0; i<12; i++)
            os<<" c"<<i+1<<"\t\t = "<<c[i]<<"\n";

        os  <<" R0            = "<<R_0<<"\n"
            <<" a             = "<<a<<"\n"
            <<" epsilon_a     = "<<a/R_0<<"\n"
            <<" elongation    = "<<elongation<<"\n"
            <<" triangularity = "<<triangularity<<"\n";
        os << std::flush;

    }
};
} //namespace solovev
} //namespace geo
} //namespace dg
