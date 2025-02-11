#pragma once

#include <variant>
#include <vector>
#include <map>
#include <string>
#include <cassert>
#include <sstream>
#include <iomanip> // std::put_time
#include <ctime>  // std::localtime
#include <netcdf.h>
#include "nc_error.h"

namespace dg
{
namespace file
{

/*! @brief Utility type to simplify dealing with heterogeneous attribute types
 *
 *  @note Unfortunately, user defined types exist so not every attribute can be
 *  an nc_att_t
 *  @ingroup netcdf
*/
using nc_att_t = std::variant<int, unsigned, float, double, bool, std::string,
      std::vector<int>, std::vector<unsigned>, std::vector<float>,
      std::vector<double>, std::vector<bool>>;

/*! @brief Generate one line entry for the history global attribute
 *
 * This will generate a string containing a whitespace seperated list of
 *  -# the current day in  <tt>"%Y-%m-%d" (the ISO 8601 date format)</tt>
 *  -# the current time in <tt>"%H:%M:%S" (the ISO 8601 time format)</tt>
 *  -# locale-dependent time zone name or abbreviation <tt>"%Z%"</tt>
 *  -# all given argv (whitespace separated)
 *  -# A newline
 *  .
 * @snippet easy_atts_t.cpp timestamp
 * @param argc from main function
 * @param argv from main function
 * @return string containing current time followed by the parameters with which the program was invoked
 * @sa See history in <a href="https://docs.unidata.ucar.edu/netcdf-c/current/attribute_conventions.html">Attribute Convenctions</a>
 * @sa <a href="https://en.cppreference.com/w/cpp/io/manip/put_time">std::put_time</a>
 * @ingroup netcdf
 */
inline std::string timestamp( int argc, char* argv[])
{
    // Get local time
    auto ttt = std::time(nullptr);
    std::ostringstream oss;
    // time string  + program-name + args
    oss << std::put_time(std::localtime(&ttt), "%F %T %Z");
    for( int i=0; i<argc; i++) oss << " "<<argv[i];
    oss << std::endl;
    return oss.str();
}

/*! @brief Version compile time constants available as a map
 *
 * Is intended to be used as NetCDF file attributes
 * @snippet easy_atts_t.cpp timestamp
 * @note We use underscore instead of "git-hash"  so that python netcdf can
 *   more easily read the attribute
 *
 * The entries in the map are filled only if the corresponding MACROs are defined at compile time.
 * Use <tt>-DVERSION_FLAGS</tt> during compilation to define all otherwise it remains empty.
 * This is the corresponding entry in \c feltor/config/version.mk
 * @code{.sh}
 * GIT_HASH=$(git rev-parse HEAD)
 * COMPILE_TIME=$(date -u +'%Y-%m-%d %H:%M:%S UTC')
 * GIT_BRANCH=$(git branch --show-current)
 * @endcode
 * @sa This approach follows
 <a href="https://stackoverflow.com/questions/44038428/include-git-commit-hash-and-or-branch-name-in-c-c-source/44038455#44038455">stackoverflow</a>.
 *  @ingroup netcdf
 */
inline const std::map<std::string, std::string> version_flags =
{
#ifdef GIT_HASH
    {"git_hash", GIT_HASH},
#endif // GIT_HASH
#ifdef GIT_BRANCH
    {"git_branch", GIT_BRANCH},
#endif // GIT_BRANCH
#ifdef COMPILE_TIME
    {"compile_time", COMPILE_TIME},
#endif // COMPILE_TIME
};

///@cond
namespace detail
{
template<class value_type>
inline nc_type getNCDataType(){ assert( false && "Type not supported!\n" ); return NC_DOUBLE; }
template<>
inline nc_type getNCDataType<double>(){ return NC_DOUBLE;}
template<>
inline nc_type getNCDataType<float>(){ return NC_FLOAT;}
template<>
inline nc_type getNCDataType<int>(){ return NC_INT;}
template<>
inline nc_type getNCDataType<unsigned>(){ return NC_UINT;}
template<>
inline nc_type getNCDataType<bool>(){ return NC_BYTE;}
template<>
inline nc_type getNCDataType<std::string>(){ return NC_STRING;}
template<>
inline nc_type getNCDataType<const char*>(){ return NC_STRING;}

// Variant for user defined data types (compound types)
// S allows both std::string and const char* to be used
template<class S, class T>
void put_att( int ncid, int varid, const std::tuple<S, nc_type,
        std::vector<T>>& att)
{
    // This will convert const char* to std::string
    std::string name_string = std::get<0>(att);
    auto name = name_string.c_str();
    nc_type xtype = std::get<1>(att);
    const std::vector<T>& data = std::get<2>(att);
    unsigned size = data.size();
    NC_Error_Handle err;
    // Test xtype ? netcdf allows xtype to be anything ...
    if constexpr( std::is_same_v<T, int>)
    {
        err = nc_put_att_int( ncid, varid, name, xtype, size, &data[0]);
    }
    else if constexpr( std::is_same_v<T, unsigned>)
    {
        err = nc_put_att_uint( ncid, varid, name, xtype, size, &data[0]);
    }
    else if constexpr( std::is_same_v<T, float>)
    {
        err = nc_put_att_float( ncid, varid, name, xtype, size, &data[0]);
    }
    else if constexpr( std::is_same_v<T, double>)
    {
        err = nc_put_att_double( ncid, varid, name, xtype, size, &data[0]);
    }
    else if constexpr( std::is_same_v<T, std::string> or
                       std::is_same_v<T, const char*>)
    {
        if( size != 1)
            throw std::runtime_error( "Cannot write a string array attribute to NetCDF");
        std::string tmp = data[0];
        err = nc_put_att_text( ncid, varid, name, tmp.size(), tmp.c_str());
    }
    else if constexpr( std::is_same_v<T, bool>)
    {
        // std::vector<bool> is not necessarily contiguous
        std::vector<signed char> dataB(size);
        for( unsigned i=0; i<size; i++)
            dataB[i] = data[i];
        err = nc_put_att_schar( ncid, varid, name, NC_BYTE, size, &dataB[0]);
    }
    else // default
    {
        err = nc_put_att( ncid, varid, name, xtype, size, &data[0]);
    }
}

template<class S, class T> // T cannot be nc_att_t
void put_att( int ncid, int varid, std::tuple<S, nc_type, T> att)
{
    put_att( ncid, varid, std::make_tuple( std::get<0>(att), std::get<1>(att),
                std::vector<T>( 1, std::get<2>(att)) ));
}

// Variants for normal types
template<class S, class T>
void put_att( int ncid, int varid, const std::pair<S, T>& att)
{
    put_att( ncid, varid, std::make_tuple( att.first,
                detail::getNCDataType<T>(), std::vector<T>(1,att.second)));
}

template<class S, class T>
void put_att( int ncid, int varid, const std::pair<S, std::vector<T>>& att)
{
    put_att( ncid, varid, std::make_tuple( att.first,
                detail::getNCDataType<T>(), att.second));
}
// Amazing
template<class S>
void put_att( int ncid, int varid, const std::pair<S, nc_att_t>& att)
{
    S name = att.first;
    const nc_att_t& v = att.second;
    std::visit( [ncid, varid, name]( auto&& arg) { put_att( ncid, varid,
                std::make_pair( name, arg)); }, v);
}


template<class Iterable> // *it must be usable in put_att
void put_atts( int ncid, int varid, const Iterable& atts)
{
    for( const auto& it : atts)
        put_att( ncid, varid, it);
}

/////////////////////////////GETTERS////////////////////////////

template<class T>
std::vector<T> get_att_v( int ncid, int varid, std::string att)
{
    auto name = att.c_str();
    size_t size;
    NC_Error_Handle err;
    err = nc_inq_attlen( ncid, varid, name, &size);
    nc_type xtype;
    err = nc_inq_atttype( ncid, varid, name, &xtype);
    std::vector<T> data(size);
    if ( xtype == NC_STRING or xtype == NC_CHAR)
    {
        std::string str( size, 'x');
        err = nc_get_att_text( ncid, varid, name, &str[0]);
        if constexpr ( std::is_convertible_v<std::string,T>)
            data[0] = str;
        else
            throw std::runtime_error("Cannot convert NC_STRING to given type");
    }
    else if ( xtype == NC_INT)
    {
        std::vector<int> tmp( size);
        err = nc_get_att_int( ncid, varid, name, &tmp[0]);
        if constexpr ( std::is_convertible_v<int,T>)
            std::copy( tmp.begin(), tmp.end(), data.begin());
        else
            throw std::runtime_error("Cannot convert NC_INT to given type");
    }
    else if ( xtype == NC_UINT)
    {
        std::vector<unsigned> tmp( size);
        err = nc_get_att_uint( ncid, varid, name, &tmp[0]);
        if constexpr ( std::is_convertible_v<unsigned,T>)
            std::copy( tmp.begin(), tmp.end(), data.begin());
        else
            throw std::runtime_error("Cannot convert NC_UINT to given type");
    }
    else if ( xtype == NC_FLOAT)
    {
        std::vector<float> tmp( size);
        err = nc_get_att_float( ncid, varid, name, &tmp[0]);
        if constexpr ( std::is_convertible_v<float,T>)
            std::copy( tmp.begin(), tmp.end(), data.begin());
        else
            throw std::runtime_error("Cannot convert NC_FLOAT to given type");
    }
    else if ( xtype == NC_DOUBLE)
    {
        std::vector<double> tmp( size);
        err = nc_get_att_double( ncid, varid, name, &tmp[0]);
        if constexpr ( std::is_convertible_v<double,T>)
            std::copy( tmp.begin(), tmp.end(), data.begin());
        else
            throw std::runtime_error("Cannot convert NC_DOUBLE to given type");
    }
    else if( xtype == NC_BYTE) // counts as bool
    {
        std::vector<signed char> tmp(size);
        err = nc_get_att_schar( ncid, varid, name, &tmp[0]);
        if constexpr ( std::is_convertible_v<bool,T>)
            std::copy( tmp.begin(), tmp.end(), data.begin());
        else
            throw std::runtime_error("Cannot convert NC_BYTE to given type");
    }
    else // default
    {
        // std::vector<bool> is not necessarily contiguous
        if constexpr (std::is_same_v<T, bool>)
        {
            std::vector<signed char> tmp(size);
            err = nc_get_att( ncid, varid, name, &tmp[0]);
            std::copy( tmp.begin(), tmp.end(), data.begin());
        }
        else
            err = nc_get_att( ncid, varid, name, &data[0]);
    }
    return data;
}

inline dg::file::nc_att_t get_att_t( int ncid, int varid,
        std::string att_name)
{
    auto name = att_name.c_str();
    size_t size;
    NC_Error_Handle err;
    err = nc_inq_attlen( ncid, varid, name, &size);
    nc_type xtype;
    err = nc_inq_atttype( ncid, varid, name, &xtype);
    if ( xtype == NC_INT and size == 1)
        return get_att_v<int>( ncid, varid, att_name)[0];
    else if ( xtype == NC_INT and size != 1)
        return get_att_v<int>( ncid, varid, att_name);
    else if ( xtype == NC_UINT and size == 1)
        return get_att_v<unsigned>( ncid, varid, att_name)[0];
    else if ( xtype == NC_UINT and size != 1)
        return get_att_v<unsigned>( ncid, varid, att_name);
    else if ( xtype == NC_FLOAT and size == 1)
        return get_att_v<float>( ncid, varid, att_name)[0];
    else if ( xtype == NC_FLOAT and size != 1)
        return get_att_v<float>( ncid, varid, att_name);
    else if ( xtype == NC_DOUBLE and size == 1)
        return get_att_v<double>( ncid, varid, att_name)[0];
    else if ( xtype == NC_DOUBLE and size != 1)
        return get_att_v<double>( ncid, varid, att_name);
    else if ( xtype == NC_BYTE and size == 1)
    {
        // BugFix: explicitly convert to bool
        bool value = get_att_v<bool>( ncid, varid, att_name)[0];
        return value;
    }
    else if ( xtype == NC_BYTE and size != 1)
        return get_att_v<bool>( ncid, varid, att_name);
    else if ( xtype == NC_STRING || xtype == NC_CHAR)
        return get_att_v<std::string>( ncid, varid, att_name)[0];
    else
        throw std::runtime_error( "Cannot convert attribute type to nc_att_t");
}

//namespace detail
//{
// utility overloads to be able to implement get_att and get_atts
template<class T>
void get_att_h( int ncid, int varid, std::string att_name, T& att)
{
    att = get_att_v<T>( ncid, varid, att_name)[0];
}
template<class T>
void get_att_h( int ncid, int varid, std::string att_name, std::vector<T>& att)
{
    att = get_att_v<T>( ncid, varid, att_name);
}
inline void get_att_h( int ncid, int varid, std::string att_name, dg::file::nc_att_t& att)
{
    att = get_att_t( ncid, varid, att_name);
}
//}
template<class T> // T can be nc_att_t
T get_att_as( int ncid, int varid, std::string att_name)
{
    T att;
    detail::get_att_h( ncid, varid, att_name, att);
    return att;
}
template<class T> // T can be nc_att_t
std::map<std::string, T> get_atts_as( int ncid, int varid)
{
    NC_Error_Handle err;
    int number;
    if( varid == NC_GLOBAL)
        err = nc_inq_natts( ncid, &number);
    else
        err = nc_inq_varnatts( ncid, varid, &number);
    std::map < std::string, T> map;
    for( int i=0; i<number; i++)
    {
        char att_name[NC_MAX_NAME]; // 256
        err = nc_inq_attname( ncid, varid, i, att_name);
        detail::get_att_h( ncid, varid, att_name, map[att_name]);
    }
    return map;
}
} // namespace detail
///@endcond
}// namespace file
}// namespace dg
