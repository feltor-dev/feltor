#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept> //std::runtime_error

#include "json/json.h"
/*!@file
 *
 * Json utility functions
 */

//Note that the json utilities are separate from netcdf utilities because
//of the different library dependencies that they incur
namespace dg
{
namespace file
{
/**
 * @defgroup json JsonCPP utilities
 * \#include "dg/file/json_utilities.h" (link -ljsoncpp)
 *
 * @addtogroup json
 * @{
 */

///@brief Switch between how to handle errors in a Json utitlity functions
enum class error{
    is_throw, //!< throw an error
    is_warning, //!< Handle the error by writing a warning to \c std::cerr
    is_silent //!< Ignore the error and silently continue execution
};

///@brief Switch how comments are treated in a json string or file
enum class comments{
    are_kept, //!< Keep comments in the Json value
    are_discarded, //!< Allow comments but discard them in the Json value
    are_forbidden //!< Treat comments as invalid Json
};

/**
 * @brief Wrapped Access to Json values with error handling
 *
 * The purpose of this class is to wrap the
 * access to a Json::Value with guards that raise exceptions or display
 * warnings in case an error occurs, for example when a key is misspelled,
 * missing or has the wrong type.
 * The goal is the composition of a good error message that helps a user
 * quickly debug the input (file).
 *
 * The Wrapper is necessary because Jsoncpp by default silently
 * generates a new key in case it is not present which in our scenario is an
 * invitation for stupid mistakes.
 *
 * You can use the \c WrappedJsonValue like a \c Json::Value with read-only access:
 * @code
Json::Value js;
dg::file::file2Json( "test.json", js);
dg::file::WrappedJsonValue ws( js, dg::file::error::is_throw);
try{
    std::string hello = ws.get( "hello", "").asString();
    // the following access will throw
    int idx0 = ws[ "array" ][out_of_bounds_index].asInt();
} catch ( std::exception& e){
    std::cerr << "Error in file test.json\n";
    //the what string knows that the out of bounds error occured in the array
    //called "array"
    std::cerr << e.what()<<std::endl;
}
 * @endcode
 * A feature of the class is that it keeps track of how a value is called.
 * For example
 * @code
void some_function( dg::file::WrappedJsonValue ws)
{
    int value = ws[ "some_non_existent_key"].asUInt();
}

dg::file::WrappedJsonValue js;
try{
    some_function( js["nested"]);
} catch ( std::exception& e){ std::cerr << e.what()<<std::endl; }
//The what string knows that "some_non_existent_key" is expected to be
//contained in the "nested" key.
 * @endcode
 */
struct WrappedJsonValue
{
    ///Default constructor
    WrappedJsonValue() : m_js(0), m_mode( error::is_silent){}
    ///@brief Construct with error mode
    ///@param mode The error mode
    WrappedJsonValue( error mode): m_js(0), m_mode( mode) {}
    ///@brief By default the error mode is \c error::is_throw
    ///@param js The Json value that will be guarded
    WrappedJsonValue(Json::Value js): m_js(js), m_mode( error::is_throw) {}
    ///@brief Construct with Json value and error mode
    ///@param js The Json value that will be guarded
    ///@param mode The error mode
    WrappedJsonValue(Json::Value js, error mode): m_js(js), m_mode( mode) {}
    ///@brief Change the error mode
    ///@param mode The new error mode
    void set_mode( error new_mode){
        m_mode = new_mode;
    }
    ///Read access to the raw Json value
    const Json::Value& asJson( ) const{ return m_js;}
    ///Write access to the raw Json value (if you know what you are doing)
    Json::Value& asJson( ) { return m_js;}

    ////////////Members imitating the original Json::Value///////////////
    /// Wrap the corresponding Json::Value function with error handling
    WrappedJsonValue operator[](std::string key) const{
        return get( key, Json::ValueType::objectValue, "empty object ");
    }
    /// Wrap the corresponding Json::Value function with error handling
    WrappedJsonValue get( std::string key, const Json::Value& value) const{
        std::stringstream default_str;
        default_str << "value "<<value;
        return get( key, value, default_str.str());
    }
    /// Wrap the corresponding Json::Value function with error handling
    WrappedJsonValue operator[]( unsigned idx) const{
        return get( idx, Json::ValueType::objectValue, "empty array");
    }
    /// Wrap the corresponding Json::Value function with error handling
    WrappedJsonValue get( unsigned idx, const Json::Value& value) const{
        std::stringstream default_str;
        default_str << "value "<<value;
        return get( idx, value, default_str.str());
    }
    /// Wrap the corresponding Json::Value function with error handling
    unsigned size() const{
        return m_js.size();
    }
    /// Wrap the corresponding Json::Value function with error handling
    double asDouble( double value = 0) const{
        if( m_js.isDouble())
            return m_js.asDouble();
        return type_error<double>( value, "a Double");
    }
    /// Wrap the corresponding Json::Value function with error handling
    unsigned asUInt( unsigned value = 0) const{
        if( m_js.isUInt())
            return m_js.asUInt();
        return type_error<unsigned>( value, "an Unsigned");
    }
    /// Wrap the corresponding Json::Value function with error handling
    int asInt( int value = 0) const{
        if( m_js.isInt())
            return m_js.asInt();
        return type_error<int>( value, "an Int");
    }
    /// Wrap the corresponding Json::Value function with error handling
    bool asBool( bool value = false) const{
        if( m_js.isBool())
            return m_js.asBool();
        return type_error<bool>( value, "a Bool");
    }
    /// Wrap the corresponding Json::Value function with error handling
    std::string asString( std::string value = "") const{
        //return m_js["hhaha"].asString(); //does not throw
        if( m_js.isString())
            return m_js.asString();
        return type_error<std::string>( value, "a String");
    }
    private:
    WrappedJsonValue(Json::Value js, error mode, std::string access):m_js(js), m_mode( mode), m_access_str(access) {}
    WrappedJsonValue get( std::string key, const Json::Value& value, std::string default_str) const
    {
        std::string access = m_access_str + "\""+key+"\": ";
        std::stringstream message;
        if( !m_js.isObject( ) || !m_js.isMember(key))
        {
            message <<"*** Key error: "<<access<<" not found.";
            raise_error( message.str(), default_str);
            return WrappedJsonValue( value, m_mode, access);
        }
        return WrappedJsonValue(m_js[key], m_mode, access);
    }
    WrappedJsonValue get( unsigned idx, const Json::Value& value, std::string default_str) const
    {
        std::string access = m_access_str + "["+std::to_string(idx)+"] ";
        if( !m_js.isArray() || !m_js.isValidIndex(idx))
        {
            std::stringstream message;
            //if( !m_js.isArray())
            //    message <<"*** Key error: "<<m_access_str<<" is not an Array.";
            //else
            if( m_access_str.empty())
                message <<"*** Index error: Index "<<idx<<" not present.";
            else
                message <<"*** Index error: Index "<<idx<<" not present in "<<m_access_str<<".";
            raise_error( message.str(), default_str);
            return WrappedJsonValue( value, m_mode, access);
        }
        return WrappedJsonValue(m_js[idx], m_mode, access);
    }
    template<class T>
    T type_error( T value, std::string type) const
    {
        std::stringstream message, default_str;
        default_str << "value "<<value;
        message <<"*** Type error: "<<m_access_str<<" "<<m_js<<" is not "<<type<<".";
        raise_error( message.str(), default_str.str());
        return value;
    }
    void raise_error( std::string message, std::string default_str) const
    {
        if( error::is_throw == m_mode)
            throw std::runtime_error( message);
        else if ( error::is_warning == m_mode)
            std::cerr <<"WARNING "<< message<<" Using default "<<default_str<<"\n";
        else
            ;
    }
    Json::Value m_js;
    error m_mode;
    std::string m_access_str = "";
};

/**
 * @brief DEPRECATED Wrapper around Json::Value::get function that handles missing keys
 *
 * @tparam T value type
 * @param err determines what to do when a key is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param value the value to take if key is missing
 *
 * @return js[key] if key is present, else value
 */
template<class T>
Json::Value get( enum error err, const Json::Value& js, std::string key, T value)
{
    WrappedJsonValue ws( js, err);
    return ws.get( key, value);
}

/**
 * @brief DEPRECATED Wrapper around Json::Value::get function that handles missing keys
 *
 * @tparam T value type
 * @param err determines what to do when a key or index is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param idx the idx within key to look for in js
 * @param value the value to take if key is missing
 *
 * @return js[key][idx] if key is present, else value
 */
template<class T>
Json::Value get_idx( enum error err, const Json::Value& js, std::string key, unsigned idx, T value)
{
    WrappedJsonValue ws( js, err);
    return ws[key].get( idx, value);
}
/**
 * @brief DEPRECATED Wrapper around Json::Value::get function that handles missing keys
 *
 * @tparam T value type
 * @param err determines what to do when a key is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param key2 the key to look for in \c key
 * @param value the value to take if key is missing
 *
 * @return js[key][key2] if key is present, else value
 */
template<class T>
Json::Value get( enum error err, const Json::Value& js, std::string key, std::string key2, T value)
{
    WrappedJsonValue ws( js, err);
    return ws[key].get( key2, value);
}
/**
 * @brief DEPRECATED Wrapper around Json::Value::get function that handles missing keys
 *
 * @tparam T value type
 * @param err determines what to do when a key or index is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param key2 the key to look for in \c key
 * @param idx the index to look for in \c key2
 * @param value the value to take if key is missing
 *
 * @return js[key][key2][idx] if key is present, else value
 */
template<class T>
Json::Value get_idx( enum error err, const Json::Value& js, std::string key, std::string key2, unsigned idx, T value)
{
    WrappedJsonValue ws( js, err);
    return ws[key][key2].get( idx, value);
}

/**
 * @brief Convenience wrapper to open a file and parse it into a Json::Value
 *
 * @note included in \c json_utilities.h
 * @param filename Name of the JSON file to parse
 * @param js Contains all the found Json variables on output
 * @param comm determines the handling of comments in the Json file
 * @param err determines how parser errors are handled by the function
 * \c error::is_throw:  throw a \c std::runtime_error containing an error message on any error that occurs on parsing;
 * \c error::is_warning: write the error message to std::cerr and return;
 * \c error::is_silent: silently return
 */
static inline void file2Json(std::string filename, Json::Value& js, enum comments comm = file::comments::are_discarded, enum error err = file::error::is_throw)
{
    Json::CharReaderBuilder parser;
    if( comments::are_forbidden == comm )
        Json::CharReaderBuilder::strictMode( &parser.settings_);
    else if( comments::are_discarded == comm )
    {
        Json::CharReaderBuilder::strictMode( &parser.settings_);
        // workaround for a linker bug in jsoncpp from package manager
        Json::Value js_true (true);
        Json::Value js_false (false);
        parser.settings_["allowComments"].swap( js_true);
        parser.settings_["collectComments"].swap(js_false);
    }
    else
        Json::CharReaderBuilder::setDefaults( &parser.settings_);

    std::ifstream isI( filename);
    if( !isI.good())
    {
        std::string message = "\nAn error occured while parsing "+filename+"\n";
        message +=  "*** File does not exist! *** \n\n";
        if( err == error::is_throw)
            throw std::runtime_error( message);
        else if (err == error::is_warning)
            std::cerr << "WARNING: "<<message<<std::endl;
        else
            ;
        return;
    }
    std::string errs;
    if( !parseFromStream( parser, isI, &js, &errs) )
    {
        std::string message = "An error occured while parsing "+filename+"\n"+errs;
        if( err == error::is_throw)
            throw std::runtime_error( message);
        else if (err == error::is_warning)
            std::cerr << "WARNING: "<<message<<std::endl;
        else
            ;
        return;
    }
}
/**
 * @brief Convenience wrapper to parse a string into a Json::Value
 *
 * Parse a string into a Json Value
 * @attention This function will throw a \c std::runtime_error with the Json error string on any error that occurs on parsing.
 * @note included in \c json_utilities.h
 * @param input The string to interpret as a Json string
 * @param js Contains all the found Json variables on output
 * @param comm determines the handling of comments in the Json string
 * @param err determines how parser errors are handled by the function
 * \c error::is_throw:  throw a \c std::runtime_error containing an error message on any error that occurs on parsing;
 * \c error::is_warning: write the error message to std::cerr and return;
 * \c error::is_silent: silently return
 */
static inline void string2Json(std::string input, Json::Value& js, enum comments comm = file::comments::are_discarded, enum error err = file::error::is_throw)
{
    Json::CharReaderBuilder parser;
    if( comments::are_forbidden == comm )
        Json::CharReaderBuilder::strictMode( &parser.settings_);
    else if( comments::are_discarded == comm )
    {
        Json::CharReaderBuilder::strictMode( &parser.settings_);
        parser.settings_["allowComments"] = true;
        parser.settings_["collectComments"] = false;
    }
    else
        Json::CharReaderBuilder::setDefaults( &parser.settings_);

    std::string errs;
    std::stringstream ss(input);
    if( !parseFromStream( parser, ss, &js, &errs) )
    {
        if( err == error::is_throw)
            throw std::runtime_error( errs);
        else if (err == error::is_warning)
            std::cerr << "WARNING: "<<errs<<std::endl;
        else
            ;
        return;
    }
}

///@}
}//namespace file
}//namespace dg
