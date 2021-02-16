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
namespace polynomial
{
/*! @class hide_polynomial_json
 * @code
// Polynomial geometry parameters
{
    "equilibrium" : "polynomial",
    "M" : 8,
    "N" : 8,
    "PI" : -1.0,
    "PP" : -1.0,
    "R_0" : 906.38,
    "c" :
    [
        -0.96689843290517163,
        3.0863312163153722,
        // ... M*N coefficients in total
    ],
    "description" : "standardX",
    "elongation" : 1.5,
    "inverseaspectratio" : 0.27593818984547458,
    "triangularity" : 0.40000000000000002
}
@endcode
*/
/**
 * @brief Constructs and display geometric parameters for the polynomial fields
 * @ingroup polynomial
 * @note include \c json/json.h before \c geometries.h in order to activate json functionality
 */
struct Parameters
{
    double R_0, //!< major tokamak radius
           pp, //!< prefactor for Psi_p
           pi, //!< prefactor for current I
           a,  //!<  little tokamak radius
           elongation, //!< elongation of the magnetic surfaces
           triangularity; //!< triangularity of the magnetic surfaces
    unsigned M, //!< number of coefficients in R
             N; //!< number of coefficients in Z
    std::vector<double> c;  //!< M*N coefficients for the polynomial equilibrium, \c c[i*N+j] corresponds to R^i Z^j;
    std::string description;
#ifdef JSONCPP_VERSION_STRING
    /**
     * @brief Construct from Json dataset
     * @copydoc hide_polynomial_json
     * @sa dg::geo::description to see valid values for the %description field
     * @param js valid Json object (see code above to see the valid key : value pairs)
     * @param mode determine what happens when a key is missing
     * @note the default values in brackets are taken if the variables are not found in the input file
     * @attention This Constructor is only defined if \c json/json.h is included before \c dg/geometries/geometries.h
     */
    Parameters( const Json::Value& js, dg::file::error mode = dg::file::error::is_silent) {
        pp  = dg::file::get( mode, js, "PP", 1).asDouble();
        pi  = dg::file::get( mode, js, "PI", 1).asDouble();
        M = dg::file::get( mode, js, "M", 1).asUInt();
        N = dg::file::get( mode, js, "N", 1).asUInt();
        c.resize(M*N);
        for (unsigned i=0;i<M*N;i++)
            c[i] = dg::file::get_idx( mode, js, "c", i, 0.).asDouble();

        R_0  = dg::file::get( mode, js, "R_0", 0.).asDouble();
        a  = R_0*dg::file::get( mode, js, "inverseaspectratio", 0.).asDouble();
        elongation=dg::file::get( mode, js, "elongation", 1.).asDouble();
        triangularity=dg::file::get( mode, js, "triangularity", 0.).asDouble();
        description = dg::file::get( mode, js, "description", "standardX").asString();
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
        js["M"] = M;
        js["N"] = N;
        js["PP"] = pp;
        js["PI"] = pi;
        for (unsigned i=0;i<N*N;i++) js["c"][i] = c[i];
        js["R_0"] = R_0;
        js["inverseaspectratio"] = a/R_0;
        js["elongation"] = elongation;
        js["triangularity"] = triangularity;
        js[ "equilibrium"] = "polynomial";
        js[ "description"] = description;
        return js;
    }
#endif // JSONCPP_VERSION_STRING
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
        os << "Polynomial Geometrical parameters are: \n"
            <<" Prefactor Psi   = "<<pp<<"\n"
            <<" Prefactor I     = "<<pi<<"\n"
            <<" number in R     = "<<M<<"\n"
            <<" number in Z     = "<<N<<"\n";
        for( unsigned i=0; i<M*N; i++)
            os<<" c"<<i+1<<"\t\t = "<<c[i]<<"\n";

        os  <<" R0            = "<<R_0<<"\n"
            <<" a             = "<<a<<"\n"
            <<" epsilon_a     = "<<a/R_0<<"\n"
            <<" description   = "<<description<<"\n"
            <<" elongation    = "<<elongation<<"\n"
            <<" triangularity = "<<triangularity<<"\n";
        os << std::flush;

    }
};
} //namespace polynomial
} //namespace geo
} //namespace dg
