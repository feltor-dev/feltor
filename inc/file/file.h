#ifndef _FILE_
#define _FILE_

#include <iostream>
#include <iomanip>
#include <sstream>

#include "read_input.h"
#include "hdf5.h"
#include "hdf5_hl.h"


/**
 * @file 
 * Utility functions for proper scientific h5 file format
 */

/**
 * @brief Namespace containing all functions
 */
namespace file{

/**
 * @brief Create a string like F5 time name
 *
 * @param time Time
 *
 * @return string formatted like F5 
 */
std::string setTime( double time)
{
    std::stringstream title; 
    title << "t=";
    title << std::setfill('0');
    title   <<std::setw(6) <<std::right
            <<(unsigned)(floor(time))
            <<"."
            <<std::setw(6) <<std::left
            <<(unsigned)((time-floor(time))*1e6);
    return title.str();
}

/**
 * @brief Get the time from string
 *
 * @param s string created by setTime function
 *
 * @return The time in the string
 */
double getTime( std::string& s)
{
    return file::read_input( s)[1]; 
}

/**
 * @brief Get the title string containing time 
 *
 * @param file The opened file
 * @param idx The index of the dataset
 *
 * @return string containing group name
 */
std::string getName( hid_t file, unsigned idx)
{
    hid_t group;
    hsize_t length = H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, idx, NULL, 10, H5P_DEFAULT);
    std::string name( length+1, 's');
    H5Lget_name_by_idx( file, ".", H5_INDEX_NAME, H5_ITER_INC, idx, &name[0], length+1, H5P_DEFAULT); 
        //std::cout << "Index "<<index<<" "<<name<<"\n";
    return name;
}

/**
 * @brief Get number of objects in file
 *
 * @param file opened file
 *
 * @return number of objects in file
 */
hsize_t getNumObjs( hid_t file)
{
    H5G_info_t group_info;
    H5Gget_info( file, &group_info);
    return group_info.nlinks;
}

    //hid_t input_id      = H5Dopen( file, "inputfile" , H5P_DEFAULT);
    //hid_t input_space   = H5Dget_space( input_id);
    //hssize_t points; 
    //points = H5Sget_simple_extent_npoints( input_space );
    //H5Sclose( input_space);
    //H5Dclose( input_id);
    //std::cout << "Size of dataset "<<points<<"\n";

} //namespace file

#endif//_FILE_
