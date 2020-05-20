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
//of the different dependencies that they incur
namespace file
{

/**
 * @brief Switch between modes how to handle missing keys in a Json file
 */
enum ErrorMode{
    throwOnError, //!< If a key is missing throw
    warning, //!< If a key is missing, write a warning to std::cerr and continue
    silent //!< If a key is missing, continue
};


/**
 * @brief Wrapper around Json::Value::get function that handles missing keys
 *
 * @tparam T value type
 * @param mode determines what to do when a key is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param value the value to take if key is missing
 *
 * @return js[key] if key is present, else value
 */
template<class T>
Json::Value get( enum ErrorMode mode, const Json::Value& js, std::string key, T value)
{
    if( js.isMember(key))
        return js[key];
    else
    {
        std::stringstream message;
        message <<"*** "<<key<<" not found.";
        if( throwOnError == mode)
            throw std::runtime_error( message.str());
        else if (warning == mode)
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
 * @param mode determines what to do when a key or index is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param idx the idx within key to look for in js
 * @param value the value to take if key is missing
 *
 * @return js[key][idx] if key is present, else value
 */
template<class T>
Json::Value get_idx( enum ErrorMode mode, const Json::Value& js, std::string key, unsigned idx, T value)
{
    if( js.isMember(key))
    {
        if( js[key].isValidIndex(idx))
            return js[key][idx];
        else
        {
            std::stringstream message;
            message << "*** Index "<<idx<<" not present in "<<key;
            if( throwOnError == mode)
                throw std::runtime_error( message.str());
            else if (warning == mode)
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
        if( throwOnError == mode)
            throw std::runtime_error( message.str());
        else if (warning == mode)
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
 * @param mode determines what to do when a key is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param key2 the key to look for in \c key
 * @param value the value to take if key is missing
 *
 * @return js[key][key2] if key is present, else value
 */
template<class T>
Json::Value get( enum ErrorMode mode, const Json::Value& js, std::string key, std::string key2, T value)
{
    if( js.isMember(key))
    {
        if( js[key].isMember(key2))
            return js[key][key2];
        else
        {
            std::stringstream message;
            message << "*** "<<key2<<" not found in "<<key;
            if( throwOnError == mode)
                throw std::runtime_error( message.str());
            else if (warning == mode)
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
        if( throwOnError == mode)
            throw std::runtime_error( message.str());
        else if (warning == mode)
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
 * @param mode determines what to do when a key or index is missing
 * @param js the input Json value
 * @param key the key to look for in js
 * @param key2 the key to look for in \c key
 * @param idx the index to look for in \c key2
 * @param value the value to take if key is missing
 *
 * @return js[key][key2][idx] if key is present, else value
 */
template<class T>
Json::Value get_idx( enum ErrorMode mode, const Json::Value& js, std::string key, std::string key2, unsigned idx, T value)
{
    if( js.isMember(key))
    {
        if( js[key].isMember(key2))
        {
            if( js[key][key2].isValidIndex(idx))
                return js[key][key2][idx];
            else
            {
                std::stringstream message;
                message << "*** Index "<<idx<<" not present in "<<key<<" : "<<key2;
                if( throwOnError == mode)
                    throw std::runtime_error( message.str());
                else if (warning == mode)
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
            if( throwOnError == mode)
                throw std::runtime_error( message.str());
            else if (warning == mode)
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
        if( throwOnError == mode)
            throw std::runtime_error( message.str());
        else if (warning == mode)
            std::cerr <<"WARNING "<< message.str()<<" Using default value "<<value<<"\n";
        else
            ;
        return value;
    }
}

/**
 * @brief Convenience wrapper to open a file and parse it into a Json::Value
 *
 * @attention This function will throw a \c std::runtime_error containing an error message on any error that occurs on parsing.
 * @note included in \c json_utilities.h
 * @param filename Name of the JSON file to parse
 * @param js Contains all the found Json variables on output
 * @param mode Either "default" in which case comments are allowed or "strict" in which case they are not
 */
static inline void file2Json( std::string filename, Json::Value& js, std::string mode = "default")
{
    Json::CharReaderBuilder parser;
    if( "strict" == mode )
        Json::CharReaderBuilder::strictMode( &parser.settings_);
    else
        Json::CharReaderBuilder::setDefaults( &parser.settings_);

    std::ifstream isI( filename);
    if( !isI.good())
    {
        std::string message = "\nAn error occured while parsing "+filename+"\n";
        message +=  "*** File does not exist! *** \n\n";
        throw std::runtime_error( message);
    }
    std::string errs;
    if( !parseFromStream( parser, isI, &js, &errs))
    {
        std::string message = "An error occured while parsing "+filename+"\n"+errs;
        throw std::runtime_error( message);
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
 * @param mode Either "default" in which case comments are allowed or "strict" in which case they are not
 */
static inline void string2Json( std::string input, Json::Value& js, std::string mode = "default")
{
    Json::CharReaderBuilder parser;
    if( "strict" == mode )
        Json::CharReaderBuilder::strictMode( &parser.settings_);
    else
        Json::CharReaderBuilder::setDefaults( &parser.settings_);
    std::string errs;
    std::stringstream ss(input);
    if( !parseFromStream( parser, ss, &js, &errs) )
        throw std::runtime_error( errs);
}

}//namespace file
