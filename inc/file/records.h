#pragma once
#include <map>
#include <functional>

namespace dg
{
namespace file
{

///@cond
namespace detail
{

template<class Signature>
struct get_first_argument_type;

template<class R, class Arg1, class ...A>
struct get_first_argument_type<R(Arg1, A...)>
{
    using type = Arg1;
};
}//namespace detail
///@endcond
/// If <tt> Signature = R(Arg1, A...)</tt> return \c Arg1
/// @ingroup netcdf
template<class Signature>
using get_first_argument_type_t = std::decay_t<typename detail::get_first_argument_type<Signature>::type>;

/// If <tt> Signature = R(Arg1, A...)</tt> return \c R
/// @ingroup netcdf
template<class Signature>
using get_result_type_t = typename std::function<Signature>::result_type;

/*!@brief Facilitate construction of CF attribute "long_name"  in records lists
 *
 * @sa Can be used as the \c Attributes class in \c dg::file::Record
 * @ingroup netcdf
 */
struct LongNameAttribute
{
    /// Make Iterable so file.put_atts will work
    auto begin() const {return m_atts.begin();}
    /// Make Iterable so file.put_atts will work
    auto end() const {return m_atts.end();}

    /*!@brief Specifically convert string literal to attribute
     */
    LongNameAttribute( const char* long_name)
    : m_atts( {{"long_name", std::string(long_name)}})
    {
    }
    private:
    std::map<std::string, nc_att_t> m_atts;

};

/**
 * @brief A realisation of the %Record concept. Helper to generate NetCDF variables.
 *
 * Supposed to be used to generate variable lists to write to NetCDF
 * @snippet nc_utilities_t.cpp record
   @tparam SignatureType Signature of the callable function
   @tparam Attributes Type of the attributes list, needs to be Iterable i.e.
   \c begin() and \c end() (e.g. \c dg::file::LongNameAttribute)
   @sa For the list of recommended variable attributes see <a href="https://docs.unidata.ucar.edu/netcdf-c/current/attribute_conventions.html">Attribute Convenctions</a>
   @ingroup netcdf
 */
template<class SignatureType, class Attributes = std::map<std::string, nc_att_t>>
struct Record
{
    using Signature = SignatureType; //!< Signature of the \c function
    std::string name; //!< Name of the variable to create
    Attributes atts; //!< Attributes of the variable: "long_name" is strongly recommended
    std::function<Signature> function; //!< The function to call that generates data for the variable
};

} //namespace file
}//namespace dg
