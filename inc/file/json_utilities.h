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
