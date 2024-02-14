#pragma once

#include <netcdf.h>
#include "json_utilities.h"
#include "nc_utilities.h"

namespace dg
{
namespace file
{
/**
 * @class hide_json_netcdf_example
 * @code
    dg::file::JsonType atts;
    atts["text"] = "Hello World!";
    atts["number"] = 3e-4;
    atts["int"] = -1;
    atts["uint"] = 10;
    atts["bool"] = true;
    atts["realarray"] = dg::file::vec2json({-1.1, 42.3});
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( "atts.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    dg::file::json2nc_attrs( atts, ncid, NC_GLOBAL);
    err = nc_close(ncid);
    // read attributes back to json
    err = nc_open( "atts.nc", 0, &ncid);
    dg::file::JsonType read = dg::file::nc_attrs2json( ncid, NC_GLOBAL);
    // read and atts are the same now
    err = nc_close(ncid);
 * @endcode
 */

/*! @brief Write a json dictionary as attributes of a netcdf variable or file
 *
 * Example code
 * @copydoc hide_json_netcdf_example
 * @attention The json values cannot be nested, only primitive variables or
 * arrays thereof can be written i.e. something like
 * <tt> value["test"]["nested"] = 42 </tt> in the above example will throw an
 * error.  This is because netcdf attributes cannot be nested.
 * Furthermore, all elements of an array must have the same type.
 * @note In an MPI program only one thread can call this function!
 * @note boolean values are mapped to byte netcdf attributes (0b for true, 1b for false)
 * @param atts A Json Dictionary containing all the attributes for the variable
 * or file. \c atts can be empty in which case no attribute is written.
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID, or NC_GLOBAL for a global attribute
 */
static void json2nc_attrs( const dg::file::JsonType& atts, int ncid, int varid )
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
            signed char convert = value.template get<bool>(); // converts false to 0; true to 1
            err = nc_put_att_schar( ncid, varid, it.key().data(), NC_BYTE, 1, &convert);
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
                std::vector<signed char> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].template get<bool>();
                err = nc_put_att_schar( ncid, varid, it.key().data(), NC_BYTE, size, data.data());
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
            signed char convert = value.asBool(); // converts false to 0; true to 1
            err = nc_put_att_schar( ncid, varid, it.name().data(), NC_BYTE, 1, &convert);
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
                std::vector<signed char> data(size);
                for( int i=0; i<size; i++)
                    data[i] = value[i].asBool();
                err = nc_put_att_schar( ncid, varid, it.name().data(), NC_BYTE, size, data.data());
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

/*! @brief Read netcdf attributes into a json dictionary
 *
 * Example code
 * @copydoc hide_json_netcdf_example
 * @note In an MPI program only one thread can call this function!
 * @note byte attributes are mapped to boolean values (0b for true, 1b for false)
 * @return A Json Dictionary containing all the attributes for the variable
 * or file. Can be empty if no attribute is present.
 * @param ncid NetCDF file or group ID
 * @param varid Variable ID, or NC_GLOBAL for a global attribute
 */
static dg::file::JsonType nc_attrs2json(int ncid, int varid)
{
    NC_Error_Handle err;
    int number;

    if( varid == NC_GLOBAL)
        err = nc_inq_natts( ncid, &number);
    else
        err = nc_inq_varnatts( ncid, varid, &number);
    dg::file::JsonType json;
    for( int i=0; i<number; i++)
    {
        char name[NC_MAX_NAME]; // 256
        err = nc_inq_attname( ncid, varid, i, name);
        nc_type att_type;
        size_t att_length;
        err = nc_inq_att( ncid, varid, name, &att_type, &att_length);
        if( att_type == NC_INT)
        {
            std::vector<int> att( att_length);
            err = nc_get_att_int( ncid, varid, name, &att[0]);
            json[name] = att_length == 1 ? dg::file::JsonType(att[0]) : vec2json(att);
        }
        else if( att_type == NC_UINT)
        {
            std::vector<unsigned> att( att_length);
            err = nc_get_att_uint( ncid, varid, name, &att[0]);
            json[name] = att_length == 1 ? dg::file::JsonType(att[0]) : vec2json(att);
        }
        else if( att_type == NC_DOUBLE)
        {
            std::vector<double> att( att_length);
            err = nc_get_att_double( ncid, varid, name, &att[0]);
            json[name] = att_length == 1 ? dg::file::JsonType(att[0]) : vec2json(att);
        }
        else if( att_type == NC_BYTE)
        {
            std::vector<signed char> att( att_length);
            err = nc_get_att_schar( ncid, varid, name, &att[0]);
            std::vector<bool> att_as_bool( att.begin(), att.end());
            json[name] = att_length == 1 ? dg::file::JsonType((bool)att[0]) : vec2json(att_as_bool);
        }
        else if( att_type == NC_STRING || att_type == NC_CHAR)
        {
            std::string att( att_length, 'x');
            err = nc_get_att_text( ncid, varid, name, &att[0]);
            json[name] = att;
        }
        else
            throw std::runtime_error( "Data type "+std::to_string(att_type)+" not supported by our converter\n");
    }
    return json;
}


}//namespace file
}//namespace dg
