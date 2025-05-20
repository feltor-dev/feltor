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
namespace solovev
{
/*! @class hide_solovev_json
 * @code
// Solovev (and Taylor) geometry parameters
{
    "equilibrium": "solovev",
    // "equilibrium" : "taylor",
    "A": 0,
    "R_0": 213.36,
    "PP": 1,
    "PI": 1,
    "c":[
        0.072597888572520090,
        -0.14926096478076946,
        // ... 12 coefficients in total
    ],
    "description" : "standardX",
    "inverseaspectratio": 0.3211009174311926,
    "triangularity": 0.3,
    "elongation": 1.44
}
@endcode
*/
/**
 * @brief Constructs and display geometric parameters for the solovev and taylor fields
 * @ingroup solovev
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
    std::string description;
    Parameters() = default;
    /**
     * @brief Construct from Json dataset
     * @copydoc hide_solovev_json
     * @sa \c dg::geo::description to see valid values for the %description field
     * @note the \c dg::geo::taylor field is chosen by setting "taylor" in the equilibrium field
     * @param js valid Json object (see code above to see the valid key : value pairs)
     * @note the default values in brackets are taken if the variables are not found in the input file
     */
    Parameters( const dg::file::WrappedJsonValue& js) {
        A   = js.get("A", 0).asDouble();
        pp  = js.get("PP", 1).asDouble();
        pi  = js.get("PI", 1).asDouble();
        c.resize(12);
        for (unsigned i=0;i<12;i++)
            c[i] = js["c"].get(i,0.).asDouble();

        R_0  =          js.get( "R_0", 0.).asDouble();
        a        = R_0* js.get( "inverseaspectratio", 0.).asDouble();
        elongation=     js.get( "elongation", 1.).asDouble();
        triangularity=  js.get( "triangularity", 0.).asDouble();
        try{
            description = js.get( "description", "standardX").asString();
        } catch ( std::exception& err)
        {
            if( isToroidal())
                description = "none";
            else if( !hasXpoint())
                description = "standardO";
            else
                description = "standardX";
        }
    }
    /**
     * @brief Put values into a json string
     *
     * @return Json value
     */
    dg::file::JsonType dump( ) const
    {
        dg::file::JsonType js;
        js["A"] = A;
        js["PP"] = pp;
        js["PI"] = pi;
        for (unsigned i=0;i<12;i++) js["c"][i] = c[i];
        js["R_0"] = R_0;
        js["inverseaspectratio"] = a/R_0;
        js["elongation"] = elongation;
        js["triangularity"] = triangularity;
        js[ "equilibrium"] = "solovev";
        js[ "description"] = description;
        return js;
    }
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
        dg::file::WrappedJsonValue js = dump();
        os << "Solovev Geometrical parameters are: \n"
            <<js.toStyledString();
        os << std::flush;
    }
};
} //namespace solovev
} //namespace geo
} //namespace dg
