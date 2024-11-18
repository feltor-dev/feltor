#pragma once

#include <variant>
#include <netcdf.h>
#include "nc_error.h"
#include "easy_dims.h"
#include "easy_output.h"
#include "easy_input.h"

/*!@file
 *
 * Our take on a modern C++ implementation of the NetCDF-4 data model
 */


namespace dg
{
namespace file
{

/// All Files are opened/ created in Netcdf-4 data format
enum NcFileMode
{
    nc_nowrite,  //!< NC_NOWRITE Open an existing file for read-only access, fail if it does not exist
    nc_write,    //!< NC_WRITE Open an existing file for read and write access, fail if it does not exist
    nc_clobberr, //!< NC_CLOBBER Create a new file for read and write access, overwrite if file exists
    nc_noclobber //!< NC_NOCLOBBER Create new file for read and write access, fail if already exists
}


// Utility type to simplify dealing with heterogeneous types
// We can write utility to convert json -> WrappedJsonValue -> nc_att_t
// and back
using nc_att_t = std::variant<int, unsigned, float, double, bool, std::string,
      std::vector<int>, std::vector<unsigned>, std::vector<float>,
      std::vector<double>, std::vector<bool>>;

// This is a singleton that cannot be copied/assigned but only moved/move-assign
// Only Netcdf-4 files are supported
// We orient ourselves at std::fstream behaviour for opening / creating files

struct SerialNcFile
{
    /////////////////////////////// CONSTRUCTORS/DESTRUCTOR /////////
    /// Construct a File Handle not associated to any file
    SerialNcFile () = default;
    /*!
      @brief Open/Create a netCDF file.
      @param path  Name of the netCDF file to open or create
      @param mode determine how to open/create the file
      @sa NcFileMode
    */
    SerialNcFile(const std::string& path, enum NcFileMode mode = nc_nowrite)
    {
        open( path, mode);
    }
    /*! @brief  There can only be exactly one file handle per physical file.
     *
     * Because the destructor releases all resources, a second handle
     * could be left in an invalid state
     */
    SerialNcFile(const SerialNcFile& rhs) = delete;
    ///@copydoc SerialNcFile(const SerialNcFile&)
    SerialNcFile& operator =(const SerialNcFile & rhs) = delete;

    /*! @brief Swap resources between two file handles
     */
    SerialNcFile(SerialNcFile&& rhs) = default;
    ///@copydoc SerialNcFile(SerialNcFile&&)
    SerialNcFile& operator =(SerialNcFile && rhs) = default;

    /// @brief Close open nc file and release all resources
    ~SerialNcFile()
    {
        // A destructor may not throw any errors
        try
        {
            close();
        }
        catch (NC_Error &e)
        {
            cerr << e.what() << endl;
        }
    }
    ///////////////////// open/close /////////

    /*! Explicitly open or create a netCDF file
     *
     * @note If a file is already open this call will throw.
     * Close it before opening a new file.
      @param path    File name
      @param ncFileFlags File flags from netcdf.h
      @sa nc_open
    */
    void open(const std::string& path, enum NcFileMode mode = nc_nowrite);
    {
        // Like a std::fstream opening fails if file already associated
        if( m_open)
            throw std::exception( "Close file before opening a new one!");

        NC_Error_Handle err;
        switch (mode)
        {
            case nc_nowrite:
                err = nc_open( path.c_str(), NC_NETCDF4 | NC_NOWRITE, &m_ncid);
                break;
            case nc_write:
                err = nc_open( path.c_str(), NC_NETCDF4 | NC_WRITE, &m_ncid);
                break;
            case nc_noclobber:
                err = nc_create( path.c_str(), NC_NETCDF4 | NC_NOCLOBBER, &m_ncid);
                break;
            case nc_clobber:
                err = nc_create( path.c_str(), NC_NETCDF4 | NC_CLOBBER, &m_ncid);
                break;
        }
        m_open = true;
        m_grp = m_ncid;
    }
    /// Check if a file is open
    bool is_open() const
    {
        return m_open;
    }

    /*! @brief Explicitly close a file
     *
     * Closing a file triggers all buffers to be written and
     * memory to be released.
     *
     * @note This function may throw
     * After closing a new file can be associated to this handle again
     */
    void close()
    {
        // when the file is closed the ncid may be assigned to a new file
        if (m_open)
        {
            NC_Error_Handle err;
            err = nc_close(m_ncid);
        }

        m_open = false;
        m_grp = m_ncid = 0;
    }

    /// Call nc_sync
    void sync()
    {
        NC_Error_Handle err;
        err = nc_sync( m_ncid);
    }

    /////////////// Groups /////////////////
    void def_group( std::string name)
    {
        int new_grp_id = 0;
        NC_Error_Handle err;
        err = nc_def_grp( m_grp, name.c_str(), &new_grp_id);
        // we just forget the id => always need to ask netcdf for id
        // should be no performance hit as (hopefully) cached
    }
    // using inq_group_parent we can use ".." to go up in the hierarchy
    // All subsequent calls to Atts, Dims and Vars are made to group
    void set_group( std::string name)
    {
        NC_Error_Handle err;
        if( !m_open)
            throw std::exception( "Can't set group in a closed file!");
        if ( name == ".")
            return;
        if ( name == "..")
        {
            err = nc_inq_grp_parent( m_grp, &m_grp);
        }
        else
        {
            // TODO What happens if 2 nested groups with same name exist?
            err = nc_inq_grp_ncid( m_ncid, name.c_str(), &m_grp);
        }
    }
    void rename_group( std::string old_name, std::string new_name)
    {
        if( !m_open)
            throw std::exception( "Can't rename group in a closed file!");
        NC_Error_Handle err;
        int old_grp;
        err = nc_inq_grp_ncid( m_ncid, old_name.c_str(), &old_grp);
        err = nc_rename_grp( old_grp, new_name.c_str());
    }

    std::vector<std::string> get_groups( ) const
    {
        if( !m_open)
            throw std::exception( "Can't get groups in a closed file!");
        NC_Error_Handle err;
        int num_grps;
        err = nc_inq_grps( m_ncid, &num_grps, NULL);
        if( num_grps == 0)
            return {};
        std::vector<int> group_ids( num_grps);
        err = nc_inq_grps( m_ncid, &num_grps, &num_grps[0]);
        std::vector<std::string> groups( num_grps);

        for( int i=0; i<num_grps; i++)
        {
            int len;
            err = nc_inq_grpname_len( group_ids[i], &len);
            char name [len];
            err = nc_inq_grpname( m_grp, name);
            groups[i] = name;
        }
        return groups;
    }

    ////////////// Dimensions ////////////////////////
    void def_dim( std::string name, size_t size = NC_UNLIMITED)
    {
        if( !m_open)
            throw std::exception( "Can't define dimension in a closed file!");
        NC_Error_Handle err;
        int dim_id;
        err = nc_def_dim( m_grp, name.c_str(), size,&dim_id);
    }
    void rename_dim( std::string old_name, std::string new_name)
    {
        if( !m_open)
            throw std::exception( "Can't renam dimension in a closed file!");
        int dimid;
        NC_Error_handle err;
        err = nc_inq_dimid( m_grp, old_name.c_str(), &dimid);
        err = nc_rename_dim( m_grp, dimid, new_name.c_str());
    }
    size_t get_size( std::string name) const
    {
        if( !m_open)
            throw std::exception( "Can't get dimension in a closed file!");
        NC_Error_handle err;
        int dimid;
        err = nc_inq_dimid( m_grp, name.c_str(), &dimid);
        size_t len;
        err = nc_inq_dimlen( m_grp, name.c_str(), &len);
        return len;

    }
    std::vector<std::string> get_dims() const
    {
        if( !m_open)
            throw std::exception( "Can't get dimension in a closed file!");
        NC_Error_handle err;
        int ndims;
        err = nc_inq_ndims( m_grp, &ndims);
        // Dimension ids are 0, 1, 2, ... in the order in which the dimensions were defined
        std::vector<std::string> dims;
        for ( int dimid = 0; dimid < ndims; dimid++)
        {
            char dimname [ NC_MAX_NAME+1];
            err = nc_inq_dimname( m_grp, dimid, dimname);
            dims[i] = dimname;
        }
        return dims;
    }

    template<class ContainerTypeType>
    void defput_dim( std::string name,
            std::map<std::string, nc_att_t> atts,
            const ContainerTypeType& abscissas);
    /////////////// Attributes setters
    // Empty var string makes a global (to the group) attribute
    // Overwrites existing!?
    // Strong attribute set
    template<class T>
    void set_att( std::string id, std::pair<std::string, T> att)
    {
        set_att( id, std::make_tuple( att.first, getNCDataType<T>(),
                    std::vector<T>(1,att.second)));
    }

    template<class T>
    void set_att( std::string id, std::pair<std::string, std::vector<T>> att)
    {
        set_att( id, std::make_tuple( att.first, getNCDataType<T>(),
                    att.second));
    }
    // Amazing
    void set_att( std::string id, std::pair<std::string, nc_att_t> att)
    {
        std::string name = att.first;
        const nc_att_t v = att.second;
        std::visit( [this,id,name]( auto&& arg) {
                this->set_att( id, std::make_pair( name, arg));
                        }, v);
    }

    template<class T> // T cannot be nc_att_t
    void set_att( std::string id, std::tuple<std::string, nc_type, T> att)
    {
        set_att( id, std::make_tuple( std::get<0>(att), std::get<1>(att),
                    std::vector<T>( 1, std::get<2>(att)) );
    }

    // This function works for compound types
    template<class T> // T cannot be nc_att_t
    void set_att( std::string id,
        std::tuple<std::string, nc_type, std::vector<T>> att);


    // Iterable can be e.g. std::vector<std::pair...>, std::map , etc.
    template<class Iterable> // *it must be usable in set_att
    void set_atts( std::string id, const Iterable& atts)
    {
        for( const auto& it : atts)
            set_att( id, it);
    }

    /////////////////// Attribute getters

    dg::file::nc_att_t get_att_t( std::string id,
            std::string att_name) const
    {
        int varid = name2varid( id, "Can't get attribute in a closed file!");
        auto name = att.c_str();
        size_t size;
        err = nc_inq_attlen( m_grp, varid, name, &size);
        nc_type xtype;
        err = nc_inq_atttype( m_grp, varid, name, &xtype);
        if ( xtype == NC_INT and size == 1)
            return get_att<int>( id, att_name);
        else if ( xtype == NC_INT and size != 1)
            return get_att_v<int>( id, att_name);
        else if ( xtype == NC_UINT and size == 1)
            return get_att<unsigned>( id, att_name);
        else if ( xtype == NC_UINT and size != 1)
            return get_att_v<unsigned>( id, att_name);
        else if ( xtype == NC_FLOAT and size == 1)
            return get_att<float>( id, att_name);
        else if ( xtype == NC_FLOAT and size != 1)
            return get_att_v<float>( id, att_name);
        else if ( xtype == NC_DOUBLE and size == 1)
            return get_att<double>( id, att_name);
        else if ( xtype == NC_DOUBLE and size != 1)
            return get_att_v<double>( id, att_name);
        else if ( xtype == NC_BYTE and size == 1)
            return get_att<bool>( id, att_name);
        else if ( xtype == NC_BYTE and size != 1)
            return get_att_v<bool>( id, att_name);
        else if ( xtype == NC_STRING )
            return get_att<std::string>( id, att_name);
        else
            throw std::runtime_error( "Cannot convert attribute type to nc_att_t");
    }

    template<class T>
    T get_att_i( std::string id, std::string att_name, unsigned idx = 0) const
    {
        auto vec = get_att_v( id, att_name);
        return vec[idx];
    }

    // This works for compound types
    template<class T>
    std::vector<T> get_att_v( std::string id, std::string att_name) const;

    // utility overloads to be able to implement get_atts
    template<class T>
    void get_att( std::string id, std::string att_name, T& att) const
    {
        att = get_att_i<T>( id, att_name);
    }
    template<class T>
    void get_att( std::string id, std::string att_name, std::vector<T>& att) const
    {
        att = get_att_v<T>( id, att_name);
    }
    void get_att( std::string id, std::string att_name, dg::file::nc_att_t& att) const
    {
        att = get_att_t( id, att_name);
    }

    // Get all attributes of a given type
    template<class T> // T can be nc_att_t
    std::map<std::string, T> get_atts( std::string id) const
    {
        int varid = name2varid( id, "Can't get attributes in a closed file!");
        NC_Error_Handle err;
        int number;
        if( varid == NC_GLOBAL)
            err = nc_inq_natts( m_grp, &number);
        else
            err = nc_inq_varnatts( m_grp, varid, &number);
        std::map < std::string, T> map;
        for( int i=0; i<number; i++)
        {
            char name[NC_MAX_NAME]; // 256
            err = nc_inq_attname( ncid, varid, i, name);
            map[name] = T{};
            get_att( id, att_name, map[name]);
        }
        return map;
    }


    void rm_att( std::string id, std::string att)
    {
        int varid = name2varid( id, "Can't delete attribute in a closed file!");
        auto name = att.c_str();
        NC_Error_Handle err;
        err = nc_del_att( m_grp, varid, name);
    }
    void rename_att( std::string id, std::string old_att_name, std::string new_att_name)
    {
        int varid = name2varid( id, "Can't delete attribute in a closed file!");
        auto old_name = old_att_name.c_str();
        auto new_name = new_att_name.c_str();
        NC_Error_Handle err;
        err = nc_rename_att( m_grp, varid, old_name, new_name);
    }

    std::vector<std::tuple<std::string, nc_type, std::any>> get_atts( std::string id = ".") const;

    ////////////// Variables ////////////////////////
    // Overwrite existing?
    void def_var_x( std::string name, nc_type xtype, std::vector<std::string> dim_names);
    //template<class Attributes>
    //void def_var_x( std::string name, nc_type xtype, std::vector<std::string> dim_names,
    //        const Attributes& atts);

    template<class T>
    void def_var( std::string name, std::vector<std::string> dim_names);

    //template<class T, class Attributes>
    //void def_var( std::string name, std::vector<std::string> dim_names,
    //        const Attributes& atts);

    template<class ContainerType>
    void put_var( std::string name, const ContainerType& data, unsigned slice=0)
    template<class ContainerType>
    void stack_var( std::string name, const ContainerType& data)

    //template<class Attributes, class ContainerType>
    //void defput_var( std::string name,
    //        std::vector<std::string> dim_names,
    //        const dg::file::JsonType& atts,
    //        const ContainerType& data)

    template<class ContainerType>
    void get_var( std::string name, ContainerType& data, unsigned slice=0) const

    std::vector<std::string> get_vars() const;
    std::vector<unsigned> get_shape( std::string var) const;

    private:
    int name2varid( std::string id, std::string error_message) const
    {
        if( !m_open)
            throw std::exception( error_message );
        NC_Error_Handle err;
        int varid;
        if ( id == ".")
            varid = NC_GLOBAL;
        else
        {
            err = nc_inq_varid( m_grp, id.c_str(), &varid);
        }
        return varid;

    }

    bool m_open = false;
    int m_ncid = 0;
    // For group activities
    int m_grp = 0; // the currently active group
    // For dimensions
    // std::map<std::string, ids>
    // For variables
    // std::any for Buffer for device to host transfer, and dg::assign


};


template<class T>
void SerialNcFile::set_att( std::string id,
    std::tuple<std::string, nc_type, std::vector<T>> att)
{
    int varid = name2varid( id, "Can't set attribute in a closed file!");
    auto name = std::get<0>(att).c_str();
    nc_type xtype = std::get<1>(att);
    const std::vector<T>& data = std::get<2>(att);
    unsigned size = data.size();
    int ncid = m_grp;
    NC_Error_Handle err;
    if ( xtype == NC_INT)
    {
        err = nc_put_att_int( ncid, varid, name, size, &data[0]);
    }
    else if ( xtype == NC_UINT)
    {
        err = nc_put_att_uint( ncid, varid, name, size, &data[0]);
    }
    else if ( xtype == NC_FLOAT)
    {
        err = nc_put_att_float( ncid, varid, name, size, &data[0]);
    }
    else if ( xtype == NC_DOUBLE)
    {
        err = nc_put_att_double( ncid, varid, name, size, &data[0]);
    }
    else if ( xtype == NC_STRING)
    {
        if( size != 1)
            throw std::runtime_error( "Cannot write a string array attribute to NetCDF");
        err = nc_put_att_text( ncid, varid, name, data[0].size(), data[0].c_str());
    }
    else if( xtype == NC_BYTE) // counts as bool
    {
        std::vector<signed char> dataB(size);
        for( int i=0; i<size; i++)
            dataB[i] = data[i];
        err = nc_put_att_schar( ncid, varid, name, NC_BYTE, size, &dataB[0]);
    }
    else // default
    {
        err = nc_put_att( ncid, varid, name, xtype, size, &data[0]);
    }
}

template<class T>
std::vector<T> SerialNcFile::get_att_v( std::string id, std::string att) const
{
    int varid = name2varid( id, "Can't get attribute in a closed file!");
    auto name = att.c_str();
    int ncid = m_grp;
    size_t size;
    err = nc_inq_attlen( m_grp, varid, name, &size);
    nc_type xtype;
    err = nc_inq_atttype( m_grp, varid, name, &xtype);
    std::vector<T> data(size);
    NC_Error_Handle err;
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
        err = nc_get_att_schar( ncid, varid, name, NC_BYTE, size, &tmp[0]);
        if constexpr ( std::is_convertible_v<bool,T>)
            std::copy( tmp.begin(), tmp.end(), data.begin());
        else
            throw std::runtime_error("Cannot convert NC_BYTE to given type");
    }
    else // default
    {
        err = nc_get_att( ncid, varid, name, &data[0]);
    }
    return data;
}

}// namespace file
}// namespace dg
