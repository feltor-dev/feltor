#pragma once

#include <netcdf.h>
#include "json_utilities.h"
#include "nc_utilities.h"

namespace dg
{
namespace file
{

/*! @brief Write a json dictionary as attributes of a netcdf variable or file
 *
 * Example code
 * @code
    dg::file::JsonType atts;
    atts["text"] = "Hello World!";
    atts["number"] = 3e-4;
    atts["int"] = -1;
    atts["uint"] = 10;
    atts["bool"] = true;
    atts["realarray"] = {-1.1, 42.3}; // works for nlohmann::json
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( "atts.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    dg::file::json2nc_attributes( ncid, NC_GLOBAL, atts);
    nc_close(ncid);
 * @endcode
 * @attention The json values cannot be nested, only primitive variables or
 * arrays thereof can be written i.e. something like
 * <tt> value["test"]["nested"] = 42 </tt> in the above example will throw an
 * error.  This is because netcdf attributes cannot be nested.
 * Furthermore, all elements of an array must have the same type.
 * @note In an MPI program only one thread can call this function!
 * @note boolean values are mapped to int netcdf attributes
 * @param atts A Json Dictionary containing all the attributes for the variable or file. \c atts can be empty in which case no attribute is written.
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID, or NC_GLOBAL for a global attribute
 */
static void json2nc_attributes( const dg::file::JsonType& atts, int ncid, int varid )
{
    NC_Error_Handle err;
#ifdef DG_USE_JSONHPP
    for (auto it = atts.begin(); it != atts.end();  ++it )
    {
        auto & value = *it;
        if ( value.is_number_integer())
        {
            int data = value.template get<int>();
            err = nc_put_att_int( ncid, varid, it.key().data(), NC_INT, 1, &data);
        }
        else if ( value.is_number_unsigned() )
        {
            unsigned data = value.template get<unsigned>();
            err = nc_put_att_uint( ncid, varid, it.key().data(), NC_UINT, 1, &data);
        }
        else if ( value.is_number_float() )
        {
            double data = value.template get<double>();
            err = nc_put_att_double( ncid, varid, it.key().data(), NC_DOUBLE, 1, &data);
        }
        else if ( value.is_boolean() )
        {
            int convert = value.template get<bool>(); // converts false to 0; true to 1
            err = nc_put_att_int( ncid, varid, it.key().data(), NC_INT, 1, &convert);
        }
        else if( value.is_string())
        {
            std::string data = value.template get<std::string>();
            err = nc_put_att_text( ncid, varid, it.key().data(), data.size(), data.data());
        }
        else if ( value.is_array())
        {
            int size = value.size();
            if ( size == 0 )
                throw std::runtime_error( "Can't write a zero sized array attribute to netcdf");
            dg::file::JsonType valz = value[0];
            if ( valz.is_number_integer())
            {
                std::vector<int> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].template get<int>();
                err = nc_put_att_int( ncid, varid, it.key().data(), NC_INT, size, data.data());
            }
            else if ( valz.is_number_unsigned() )
            {
                std::vector<unsigned> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].template get<unsigned>();
                err = nc_put_att_uint( ncid, varid, it.key().data(), NC_UINT, size, data.data());
            }
            else if ( valz.is_number_float() )
            {
                std::vector<double> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].template get<double>();
                err = nc_put_att_double( ncid, varid, it.key().data(), NC_DOUBLE, size, data.data());
            }
            else if ( valz.is_boolean() )
            {
                std::vector<int> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].template get<bool>();
                err = nc_put_att_int( ncid, varid, it.key().data(), NC_INT, size, data.data());
            }
            else if( valz.is_string())
            {
                throw std::runtime_error( "Can't write a string array attribute to netcdf");
            }

        }
        else if( value.is_object())
            throw std::runtime_error( "Can't write Json object as netcdf attribute\n");
        else
            throw std::runtime_error( "Data type not supported by netcdf\n");
    }
#else
    for (auto it = atts.begin(); it != atts.end();  ++it )
    {
        auto & value = *it;
        if ( value.isInt())
        {
            int data = value.asInt();
            err = nc_put_att_int( ncid, varid, it.name().data(), NC_INT, 1, &data);
        }
        else if ( value.isUInt() )
        {
            unsigned data = value.asUInt();
            err = nc_put_att_uint( ncid, varid, it.name().data(), NC_UINT, 1, &data);
        }
        else if ( value.isDouble() )
        {
            double data = value.asDouble();
            err = nc_put_att_double( ncid, varid, it.name().data(), NC_DOUBLE, 1, &data);
        }
        else if ( value.isBool() )
        {
            int convert = value.asBool(); // converts false to 0; true to 1
            err = nc_put_att_int( ncid, varid, it.name().data(), NC_INT, 1, &convert);
        }
        else if( value.isString())
        {
            std::string data = value.asString();
            err = nc_put_att_text( ncid, varid, it.name().data(), data.size(), data.data());
        }
        else if ( value.isArray())
        {
            int size = value.size();
            if ( size == 0 )
                throw std::runtime_error( "Can't write a zero sized array attribute to netcdf");
            Json::Value valz = value[0];
            if ( valz.isInt())
            {
                std::vector<int> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].asInt();
                err = nc_put_att_int( ncid, varid, it.name().data(), NC_INT, size, data.data());
            }
            else if ( valz.isUInt() )
            {
                std::vector<unsigned> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].asUInt();
                err = nc_put_att_uint( ncid, varid, it.name().data(), NC_UINT, size, data.data());
            }
            else if ( valz.isDouble() )
            {
                std::vector<double> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].asDouble();
                err = nc_put_att_double( ncid, varid, it.name().data(), NC_DOUBLE, size, data.data());
            }
            else if ( valz.isBool() )
            {
                std::vector<int> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].asBool();
                err = nc_put_att_int( ncid, varid, it.name().data(), NC_INT, size, data.data());
            }
            else if( valz.isString())
            {
                throw std::runtime_error( "Can't write a string array attribute to netcdf");
            }

        }
        else if( value.isObject())
            throw std::runtime_error( "Can't write Json object as netcdf attribute\n");
        else
            throw std::runtime_error( "Data type not supported by netcdf\n");
    }
#endif

}

}//namespace file
}//namespace dg
