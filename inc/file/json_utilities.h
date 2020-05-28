#pragma once

#include <iostream>
#include <fstream>
#include <string>
#include <stdexcept> //std::runtime_error

#include "json/json.h"
/*!@file
 *
 * Json utility functions
 */

//Note that the json utilities are separate from netcdf utilities because
//of the different library dependencies that they incur
namespace file
{

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
 * @brief Wrapper around Json::Value::get function that handles missing keys
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
    if( js.isMember(key))
        return js[key];
    else
    {
        std::stringstream message;
        message <<"*** "<<key<<" not found.";
        if( error::is_throw == err)
            throw std::runtime_error( message.str());
        else if ( error::is_warning == err)
            std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
        else
            ;
        return value;
    }
}

/**
 * @brief Wrapper around Json::Value::get function that handles missing keys
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
    if( js.isMember(key))
    {
        if( js[key].isArray() && js[key].isValidIndex(idx))
            return js[key][idx];
        else
        {
            std::stringstream message;
            message << "*** Index "<<idx<<" not present in "<<key;
            if( error::is_throw == err)
                throw std::runtime_error( message.str());
            else if (error::is_warning == err)
                std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
            else
                ;
            return value;
        }
    }
    else
    {
        std::stringstream message;
        message << "*** "<<key<<"["<<idx<<"] not found.";
        if( error::is_throw == err)
            throw std::runtime_error( message.str());
        else if (error::is_warning == err)
            std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
        else
            ;
        return value;
    }
}
/**
 * @brief Wrapper around Json::Value::get function that handles missing keys
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
    if( js.isMember(key))
    {
        if( js[key].isObject() && js[key].isMember(key2))
            return js[key][key2];
        else
        {
            std::stringstream message;
            message << "*** "<<key2<<" not found in "<<key;
            if( error::is_throw == err)
                throw std::runtime_error( message.str());
            else if (error::is_warning == err)
                std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
            else
                ;
            return value;
        }
    }
    else
    {
        std::stringstream message;
        message << "*** "<<key<<" : "<<key2<<" not found.";
        if( error::is_throw == err)
            throw std::runtime_error( message.str());
        else if (error::is_warning == err)
            std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
        else
            ;
        return value;
    }
}
/**
 * @brief Wrapper around Json::Value::get function that handles missing keys
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
    if( js.isMember(key))
    {
        if( js[key].isObject() && js[key].isMember(key2))
        {
            if( js[key][key2].isArray() && js[key][key2].isValidIndex(idx))
                return js[key][key2][idx];
            else
            {
                std::stringstream message;
                message << "*** Index "<<idx<<" not present in "<<key<<" : "<<key2;
                if( error::is_throw == err)
                    throw std::runtime_error( message.str());
                else if (error::is_warning == err)
                    std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
                else
                    ;
                return value;
            }
        }
        else
        {
            std::stringstream message;
            message << "*** "<<key2<<"["<<idx<<"] not found in "<<key;
            if( error::is_throw == err)
                throw std::runtime_error( message.str());
            else if (error::is_warning == err)
                std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
            else
                ;
            return value;
        }
    }
    else
    {
        std::stringstream message;
        message << "*** "<<key<<" : "<<key2<<"["<<idx<<"] not found.";
        if( error::is_throw == err)
            throw std::runtime_error( message.str());
        else if (error::is_warning == err)
            std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
        else
            ;
        return value;
    }
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
        parser.settings_["allowComments"] = true;
        parser.settings_["collectComments"] = false;
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

}//namespace file
