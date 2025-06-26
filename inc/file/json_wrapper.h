#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <stdexcept> //std::runtime_error

#ifdef DG_USE_JSONHPP
#include <nlohmann/json.hpp>
#else
#include "json/json.h"
#endif

//Note that the json utilities are separate from netcdf utilities because
//of the different library dependencies that they incur
namespace dg
{
namespace file
{
/**
 * @addtogroup wrapper
 * @{
 */

#ifdef DG_USE_JSONHPP
using JsonType = nlohmann::json;
#else
/**
 * @brief Json Type to use in \c dg::file functions and classes
 *
 * By default this typedef expands to jsoncpp's \c Json::Value
 * @note
 * If the Macro \c DG_USE_JSONHPP is defined before the inclusion of <tt> "dg/file/file.h"</tt>
 * then the typedef expands to \c nlohmann::json
 */
using JsonType = Json::Value;
#endif


///@brief Switch between how to handle errors in a Json utitlity functions
enum class error{
    is_throw, //!< throw an error (\c std::runtime_error)
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
 * The purpose of this class is to serve as an extremely pedantic access
 * guard to a Json file, in the sense that it will raise exceptions at
 * the slightest misstep, for example when a key is misspelled,
 * missing or has the wrong type.
 * It will then compose an error message that shows where exactly
 * the access in the file went wrong and thus help a user
 * quickly debug the input (file).
 *
 * This is necessary if the cost of a faulty input file with silly mistakes
 * like misspelling could lead to potentially large (computational) costs if
 * uncaught.
 *
 * The interface of \c WrappedJsonValue is modelled after jsoncpp's \c Json::Value:
 * @code
auto js = dg::file::file2Json( "test.json", js);
dg::file::WrappedJsonValue ws( js, dg::file::error::is_throw);
try{
    std::string hello = ws.get( "hello", "").asString();
    // the following access will throw a std::runtime_error
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
 *
 * The caveat of this class is that once a json value is wrapped it is somewhat
 * awkward to change its value (because it is what the class wants to prevent)
 * @attention This class is only for read access. If you must change a value do
 * so on the raw \c JsonType accesible with the \c asJson() method
 * @attention Do not assign to a key like this:
 * @code
 * dg::file::WrappedJsonValue ws;
 * ws["hello"]["world"] = dg::file::file2Json("test.json");
 * // NOT WHAT YOU EXPECT!
 * // it just assigns to a copy that goes out of scope
 * // instead you need to work on the unwrapped json directly
 * auto js = ws.asJson();
 * js["hello"]["world"] = dg::file::file2Json("test.json")
 * @endcode
 *
 * @note If the Marco \c DG_USE_JSONHPP is defined, the <tt>\#include <nlohmann/json.hpp></tt>
 * parser is used instead of <tt>\#include <json/json.h></tt> Since the former is header-only
 * no additional linker options must be present at compilation.
 */
struct WrappedJsonValue
{
    ///@brief Default constructor
    ///By default the error mode is \c error::is_throw
    WrappedJsonValue() : m_js(0), m_mode( error::is_throw){}
    ///@brief Construct with error mode
    ///@param mode The error mode
    WrappedJsonValue( error mode): m_js(0), m_mode( mode) {}
    ///@brief By default the error mode is \c error::is_throw
    ///@param js The Json value that will be guarded
    WrappedJsonValue(JsonType js): m_js(js), m_mode( error::is_throw) {}
    ///@brief Construct with Json value and error mode
    ///@param js The Json value that will be guarded
    ///@param mode The error mode
    WrappedJsonValue(JsonType js, error mode): m_js(js), m_mode( mode) {}
    ///@brief Change the error mode
    ///@param new_mode The new error mode
    void set_mode( error new_mode){
        m_mode = new_mode;
    }
    ///Read access to the raw Json value
    const JsonType& asJson( ) const{ return m_js;}
    ///Write access to the raw Json value (if you know what you are doing)
    JsonType& asJson( ) { return m_js;}

    /*! @brief The creation history of the object
     *
     * Useful to print when debugging parameter files
     * @return A string containing object history
     */
    std::string access_string() const {return m_access_str;}

    /*! @brief The stored json object as a formatted string
     *
     * Useful when writing json to file
     * @return A string displaying json object
     */
    std::string toStyledString() const{
#ifdef DG_USE_JSONHPP
        return m_js.dump(4);
#else
        return m_js.toStyledString();
#endif
    }

    /// Return true if key is a Member of the json value
    bool isMember(std::string key) const{
#ifdef DG_USE_JSONHPP
        return m_js.contains(key);
#else
        return m_js.isMember(key);
#endif
    }

    // //////////Members imitating the original JsonType///////////////
    /// Wrap the corresponding JsonType function with error handling
    /// @attention Do not assign to this! You will assign to a copy
    WrappedJsonValue operator[](std::string key) const{
#ifdef DG_USE_JSONHPP
        return get( key, nlohmann::json::object(), "empty object ");
#else
        return get( key, Json::ValueType::objectValue, "empty object ");
#endif
    }
//    The problem with this is that if key is misspelled then
//    it will silently generate it
//    JsonType& operator[]( std::string key){
//        return m_js[key];
//    }
    /// Wrap the corresponding JsonType function with error handling
    WrappedJsonValue get( std::string key, const JsonType& value) const{
        std::stringstream default_str;
        default_str << "value "<<value;
        return get( key, value, default_str.str());
    }
    /// Wrap the corresponding JsonType function with error handling
    WrappedJsonValue operator[]( unsigned idx) const{
#ifdef DG_USE_JSONHPP
        return get( idx, nlohmann::json::object(), "empty array");
#else
        return get( idx, Json::ValueType::objectValue, "empty array");
#endif
    }
    /// Wrap the corresponding JsonType function with error handling
    WrappedJsonValue get( unsigned idx, const JsonType& value) const{
        std::stringstream default_str;
        default_str << "value "<<value;
        return get( idx, value, default_str.str());
    }
    /// Wrap the corresponding JsonType function with error handling
    unsigned size() const{
        return (unsigned)m_js.size();
    }
    /// Wrap the corresponding JsonType function with error handling
    double asDouble( double value = 0) const{
#ifdef DG_USE_JSONHPP
        if( m_js.is_number()) // we just want anything that can be cast to double
            return m_js.template get<double>();
#else
        if( m_js.isDouble())
            return m_js.asDouble();
#endif
        return type_error<double>( value, "a Double");
    }
    /// Wrap the corresponding JsonType function with error handling
    unsigned asUInt( unsigned value = 0) const{
#ifdef DG_USE_JSONHPP
        if( m_js.is_number()) // check for sign?
            return m_js.template get<unsigned>();
#else
        if( m_js.isUInt())
            return m_js.asUInt();
#endif
        return type_error<unsigned>( value, "an Unsigned");
    }
    /// Wrap the corresponding JsonType function with error handling
    int asInt( int value = 0) const{
#ifdef DG_USE_JSONHPP
        if( m_js.is_number())
            return m_js.template get<int>();
#else
        if( m_js.isInt())
            return m_js.asInt();
#endif
        return type_error<int>( value, "an Int");
    }
    /// Wrap the corresponding JsonType function with error handling
    bool asBool( bool value = false) const{
#ifdef DG_USE_JSONHPP
        if( m_js.is_boolean())
            return m_js.template get<bool>();
#else
        if( m_js.isBool())
            return m_js.asBool();
#endif
        return type_error<bool>( value, "a Bool");
    }
    /// Wrap the corresponding JsonType function with error handling
    std::string asString( std::string value = "") const{
#ifdef DG_USE_JSONHPP
        if( m_js.is_string())
            return m_js.template get<std::string>();
#else
        //return m_js["hhaha"].asString(); //does not throw
        if( m_js.isString())
            return m_js.asString();
#endif
        return type_error<std::string>( value, "a String");
    }
    private:
    WrappedJsonValue(JsonType js, error mode, std::string access):m_js(js), m_mode( mode), m_access_str(access) {}
    WrappedJsonValue get( std::string key, const JsonType& value, std::string default_str) const
    {
        std::string access = m_access_str + "\""+key+"\": ";
        std::stringstream message;
#ifdef DG_USE_JSONHPP
        if( !m_js.is_object( ) || !m_js.contains(key))
#else
        if( !m_js.isObject( ) || !m_js.isMember(key))
#endif
        {
            message <<"*** Key error: "<<access<<" not found.";
            raise_error( message.str(), default_str);
            return WrappedJsonValue( value, m_mode, access);
        }
        return WrappedJsonValue(m_js[key], m_mode, access);
    }
    WrappedJsonValue get( unsigned idx, const JsonType& value, std::string default_str) const
    {
        std::string access = m_access_str + "["+std::to_string(idx)+"] ";
#ifdef DG_USE_JSONHPP
        if( !m_js.is_array() || !(idx < m_js.size()))
#else
        if( !m_js.isArray() || !m_js.isValidIndex(idx))
#endif
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
        {
        }
    }
    JsonType m_js;
    error m_mode;
    std::string m_access_str = "";
};
///@}

///@addtogroup json
///@{
/**
 * @brief Convenience wrapper to open a file and parse it into a JsonType
 *
 * @note included in \c json_utilities.h
 * @param filename Name of the JSON file to parse (the file path is relative to where the calling program is executed)
 * @param comm determines the handling of comments in the Json file
 * @param err determines how parser errors are handled by the function
 * \c error::is_throw:  throw a \c std::runtime_error containing an error message on any error that occurs on parsing;
 * \c error::is_warning: write the error message to std::cerr and return;
 * \c error::is_silent: silently return
 * @return js object with all the found variables in \c filename
 */
inline JsonType file2Json(std::string filename, enum comments comm =
        file::comments::are_discarded, enum error err = file::error::is_throw)
{
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
        {
        }
        return JsonType();
    }
    JsonType js;
#ifdef DG_USE_JSONHPP
    bool ignore_comments = false, allow_exceptions = false;
    if ( comm == file::comments::are_discarded)
        ignore_comments =  true;
    if ( err == error::is_throw)
        allow_exceptions = true; //throws nlohmann::json::parse_error

    js = nlohmann::json::parse( isI, nullptr, allow_exceptions, ignore_comments);
    if( !allow_exceptions && err == error::is_warning && js.is_discarded())
    {
        std::string message = "An error occured while parsing "+filename+"\n";
        std::cerr << "WARNING: "<<message<<std::endl;
    }
#else
    Json::CharReaderBuilder parser;
    if( comments::are_forbidden == comm )
        Json::CharReaderBuilder::strictMode( &parser.settings_);
    else if( comments::are_discarded == comm )
    {
        Json::CharReaderBuilder::strictMode( &parser.settings_);
        // workaround for a linker bug in jsoncpp from package manager
        JsonType js_true (true);
        JsonType js_false (false);
        parser.settings_["allowComments"].swap( js_true);
        parser.settings_["collectComments"].swap(js_false);
    }
    else
        Json::CharReaderBuilder::setDefaults( &parser.settings_);

    std::string errs;
    if( !parseFromStream( parser, isI, &js, &errs) )
    {
        std::string message = "An error occured while parsing "+filename+"\n"+errs;
        if( err == error::is_throw)
            throw std::runtime_error( message);
        else if (err == error::is_warning)
            std::cerr << "WARNING: "<<message<<std::endl;
        else
        {
        }
    }
#endif
    return js;
}

/// @brief Same as <tt>js = dg::file::file2Json( filename, comm, err)</tt>
inline void file2Json(std::string filename, JsonType& js, enum comments comm = file::comments::are_discarded, enum error err = file::error::is_throw)
{
    js = file2Json( filename, comm, err);
}


/**
 * @brief Convenience wrapper to parse a string into a JsonType
 *
 * Parse a string into a Json Value
 * @attention This function will throw a \c std::runtime_error with the Json error string on any error that occurs on parsing.
 * @note included in \c json_utilities.h
 * @param input The string to interpret as a Json string
 * @param comm determines the handling of comments in the Json string
 * @param err determines how parser errors are handled by the function
 * \c error::is_throw:  throw a \c std::runtime_error containing an error message on any error that occurs on parsing;
 * \c error::is_warning: write the error message to std::cerr and return;
 * \c error::is_silent: silently return
 * @return json object with all the found Json variables in \c input
 */
inline JsonType string2Json(std::string input, enum comments comm = file::comments::are_discarded, enum error err = file::error::is_throw)
{
    JsonType js;
#ifdef DG_USE_JSONHPP
    bool ignore_comments = false, allow_exceptions = false;
    if ( comm == file::comments::are_discarded)
        ignore_comments =  true;
    if ( err == error::is_throw)
        allow_exceptions = true; //throws nlohmann::json::parse_error


    js = nlohmann::json::parse( input, nullptr, allow_exceptions, ignore_comments);
    if( !allow_exceptions && err == error::is_warning && js.is_discarded())
    {
        std::string message = "An error occured while parsing \n";
        std::cerr << "WARNING: "<<message<<std::endl;
    }
    return js;

#else

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
        {
        }
    }
#endif
    return js;
}

/// @brief Same as <tt>js = string2Json( input, comm, err)</tt>
inline void string2Json(std::string input, JsonType& js, enum comments comm = file::comments::are_discarded, enum error err = file::error::is_throw)
{
    js = string2Json( input, comm, err);
}

/**
 * @brief convert a vector to a json arrray
 *
 * @param shared Any shared memory container that allows range based for loops
 */
template<class ContainerType>
dg::file::JsonType vec2json( const ContainerType& shared)
{
#ifdef DG_USE_JSONHPP
    return nlohmann::json(shared);
#else
    Json::Value val;
    for( const auto& value : shared)
        val.append(value);
    return val;
#endif
}

/// Specialization for intitializer list
template<class T>
dg::file::JsonType vec2json( std::initializer_list<T> shared)
{
    std::vector<T> cc( shared);
    return vec2json(cc);
}
///@}


}//namespace file
}//namespace dg
