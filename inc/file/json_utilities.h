#pragma once

#include <iostream>
#include <fstream>
#include <string>

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
 * @attention This function will print an error message to \c std::cerr and brutally call \c exit(EXIT_FAILURE) on any error that occurs.
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
        std::cerr << "\nAn error occured while parsing "<<filename<<"\n";
        std::cerr << "*** File does not exist! *** \n\n";
        exit( EXIT_FAILURE);
    }
    std::string errs;
    if( !parseFromStream( parser, isI, &js, &errs))
    {
        std::cerr << "An error occured while parsing "<<filename<<"\n"<<errs;
        exit( EXIT_FAILURE);
    }
}
/**
 * @brief Convenience wrapper to parse a string into a Json::Value
 *
 * Parse a string into a Json Value
 * @attention This function will print an error message to \c std::cerr and brutally call \c exit(EXIT_FAILURE) on any error that occurs.
 * @note included in \c json_utilities.h
 * @param filename (a string to print when an error occurs, has no further use)
 * @param input The string to interpret as a Json string
 * @param js Contains all the found Json variables on output
 * @param mode Either "default" in which case comments are allowed or "strict" in which case they are not
 */
static inline void string2Json( std::string filename, std::string input, Json::Value& js, std::string mode = "default")
{
    Json::CharReaderBuilder parser;
    if( "strict" == mode )
        Json::CharReaderBuilder::strictMode( &parser.settings_);
    else
        Json::CharReaderBuilder::setDefaults( &parser.settings_);
    std::string errs;
    std::stringstream ss(input);
    if( !parseFromStream( parser, ss, &js, &errs) )
    {
        std::cerr << "An error occured while parsing "<<filename<<"\n"<<errs;
        exit( EXIT_FAILURE);
    }
}

}//namespace file
