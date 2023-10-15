#pragma once
#include <string>
#include <vector>
#include <dg/file/json_utilities.h>
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
    Parameters() = default;
    /**
     * @brief Construct from Json dataset
     * @copydoc hide_polynomial_json
     * @sa dg::geo::description to see valid values for the %description field
     * @param js valid Json object (see code above to see the valid key : value pairs)
     * @note the default values in brackets are taken if the variables are not found in the input file
     */
    Parameters( const dg::file::WrappedJsonValue& js) {
        pp  = js.get( "PP", 1).asDouble();
        pi  = js.get( "PI", 1).asDouble();
        M = js.get( "M", 1).asUInt();
        N = js.get( "N", 1).asUInt();
        c.resize(M*N);
        for (unsigned i=0;i<M*N;i++)
            c[i] = js["c"].get(i,0.).asDouble();

        R_0  = js.get( "R_0", 0.).asDouble();
        a  = R_0*js.get( "inverseaspectratio", 0.).asDouble();
        elongation=js.get( "elongation", 1.).asDouble();
        triangularity=js.get( "triangularity", 0.).asDouble();
        description = js.get( "description", "standardX").asString();
    }
    /**
     * @brief Put values into a json string
     *
     * @return Json value
     */
    dg::file::WrappedJsonValue dump( ) const
    {
        nlohmann::json js;
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
        auto js = dump();
        os << "Polynomial Geometrical parameters are: \n"
            <<js.asJson().dump(4);
        os << std::flush;

    }
};
} //namespace polynomial
} //namespace geo
} //namespace dg
