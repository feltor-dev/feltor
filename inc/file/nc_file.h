#pragma once

#include <filesystem>
#include "dg/blas.h"
#include "dg/backend/memory.h"
#include "nc_error.h"
#include "easy_atts.h"
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
    nc_clobber, //!< NC_CLOBBER Create a new file for read and write access, overwrite if file exists
    nc_noclobber //!< NC_NOCLOBBER Create new file for read and write access, fail if already exists
};

struct NcVariable
{
    std::string name;
    nc_type xtype;
    std::vector<std::string> dims;
};



// This is a singleton that cannot be copied/assigned but only moved/move-assign
// Only Netcdf-4 files are supported
// We orient ourselves at std::fstream behaviour for opening / creating files
// The class hides all integer ids that the netcdf
// C-library uses ("Ids do not exist in the Netcdf-4 data model")

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
            std::cerr << e.what() << std::endl;
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
    void open(const std::string& path, enum NcFileMode mode = nc_nowrite)
    {
        // Like a std::fstream opening fails if file already associated
        if( m_open)
            throw std::runtime_error( "Close file before opening a new one!");

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
        if( !m_open)
            throw std::runtime_error( "Can't sync a closed file!");
        NC_Error_Handle err;
        err = nc_sync( m_ncid);
    }
    int get_ncid() const { return m_ncid;}

    /////////////// Groups /////////////////
    // create all required intermediary groups in path as well
    void def_grp( std::string name)
    {
        if( !m_open)
            throw std::runtime_error( "Can't create group in a closed file!");
        int new_grp_id = 0;
        NC_Error_Handle err;
        err = nc_def_grp( m_grp, name.c_str(), &new_grp_id);
        // we just forget the id => always need to ask netcdf for id
        // should be no performance hit as (hopefully) cached
    }
    void def_grp_p( std::filesystem::path path)
    {
        if( !m_open)
            throw std::runtime_error( "Can't create group in a closed file!");
        if( not path.has_root_path()) // it is a relative path
        {
            auto current = get_grp_path();
            path = (current / path);
        }
        int groupid = m_ncid;
        auto rel_path = path.relative_path();
        NC_Error_Handle err;
        for( auto it = rel_path.begin(); it != rel_path.end(); it++)
        {
            std::string grp = *it;
            int new_grpid;
            int retval = nc_inq_ncid( groupid, grp.c_str(), &new_grpid);
            if( retval != NC_NOERR)
                err = nc_def_grp( groupid, grp.c_str(), &groupid);
            else
                groupid = new_grpid;
        }
    }
    bool grp_exists( std::filesystem::path path) const
    {
        if( !m_open)
            throw std::runtime_error( "Can't check group in a closed file!");
        std::string name = path.generic_string();
        if( not path.has_root_path()) // it is a relative path
        {
            auto current = get_grp_path();
            name = (current / path ).generic_string();
        }
        int grpid=0;
        int retval = nc_inq_grp_full_ncid( m_ncid, name.c_str(), &grpid);
        return retval == NC_NOERR;
    }
    // using inq_group_parent we can use ".." to go up in the hierarchy
    // All subsequent calls to Atts, Dims and Vars are made to group
    // Empty string goes back to root group
    // What happens if 2 nested groups with same name exist?
    // A: That is allowed to happen
    // path can be absolute or relative
    void set_grp( std::filesystem::path path = "")
    {
        if( !m_open)
            throw std::runtime_error( "Can't set group in a closed file!");
        NC_Error_Handle err;
        std::string name = path.generic_string();
        if ( name == ".")
            return;
        if ( name == "" or name == "/")
        {
            m_grp = m_ncid;
        }
        else if ( name == "..")
        {
            if( m_grp == m_ncid)
                return;
            err = nc_inq_grp_parent( m_grp, &m_grp);
        }
        else
        {
            if( not path.has_root_path()) // it is a relative path
            {
                auto current = get_grp_path();
                name = (current / path ).generic_string();
            }
            err = nc_inq_grp_full_ncid( m_ncid, name.c_str(), &m_grp);
        }
    }
    // rename a subgroup in the current group
    void rename_grp( std::string old_name, std::string new_name)
    {
        if( !m_open)
            throw std::runtime_error( "Can't rename group in a closed file!");
        NC_Error_Handle err;
        int old_grp;
        err = nc_inq_grp_ncid( m_grp, old_name.c_str(), &old_grp);
        err = nc_rename_grp( old_grp, new_name.c_str());
    }

    int get_grpid() const { return m_grp;}

    std::filesystem::path get_grp_path( ) const
    {
        if( !m_open)
            throw std::runtime_error( "Can't get group in a closed file!");
        return get_grp_path( m_grp);
    }

    /// Get all subgroups in the current group as absolute paths
    std::vector<std::filesystem::path> get_grps( ) const
    {
        if( !m_open)
            throw std::runtime_error( "Can't get groups in a closed file!");
        auto grps = get_grps(m_grp);
        std::vector<std::filesystem::path> grps_v;
        for( auto grp : grps)
            grps_v.push_back( grp.second);
        return grps_v;

    }
    /// Get all subgroups recursively in the current group as absolute paths
    std::vector<std::filesystem::path> get_grps_r( ) const
    {
        auto grps = get_grps_r(m_grp);
        std::vector<std::filesystem::path> grps_v;
        for( auto grp : grps)
            grps_v.push_back( grp.second);
        return grps_v;
    }

    ////////////// Dimensions ////////////////////////
    //Remember that dimensions do not have attributes, only variables (or groups)
    void def_dim( std::string name, size_t size)
    {
        if( !m_open)
            throw std::runtime_error( "Can't define dimension in a closed file!");
        NC_Error_Handle err;
        int dim_id;
        err = nc_def_dim( m_grp, name.c_str(), size, &dim_id);
    }
    void rename_dim( std::string old_name, std::string new_name)
    {
        if( !m_open)
            throw std::runtime_error( "Can't rename dimension in a closed file!");
        int dimid;
        NC_Error_Handle err;
        err = nc_inq_dimid( m_grp, old_name.c_str(), &dimid);
        err = nc_rename_dim( m_grp, dimid, new_name.c_str());
    }
    size_t dim_size( std::string name) const
    {
        if( !m_open)
            throw std::runtime_error( "Can't get dimension in a closed file!");
        NC_Error_Handle err;
        int dimid;
        err = nc_inq_dimid( m_grp, name.c_str(), &dimid);
        size_t len;
        err = nc_inq_dimlen( m_grp, dimid, &len);
        return len;

    }

    std::vector<size_t> dims_shape( const std::vector<std::string>& dims) const
    {
        std::vector<size_t> shape( dims.size());
        for( unsigned u=0; u<dims.size(); u++)
            shape[u] = dim_size( dims[u]);
        return shape;
    }
    std::vector<std::string> get_dims() const
    {
        if( !m_open)
            throw std::runtime_error( "Can't get dimension in a closed file!");
        NC_Error_Handle err;
        int ndims;
        err = nc_inq_ndims( m_grp, &ndims);
        // Dimension ids are 0, 1, 2, ... in the order in which the dimensions
        // were defined
        std::vector<std::string> dims;
        for ( int dimid = 0; dimid < ndims; dimid++)
        {
            char dimname [ NC_MAX_NAME+1];
            err = nc_inq_dimname( m_grp, dimid, dimname);
            dims[dimid] = dimname;
        }
        return dims;
    }
    bool dim_exists( std::string name) const
    {
        int dimid=0;
        int retval = nc_inq_dimid( m_grp, name.c_str(), &dimid);
        return retval == NC_NOERR;
    }
    /////////////// Attributes setters
    // Empty var string makes a global (to the group) attribute
    // Overwrites existing!?
    // Strong attribute set
    void set_att ( std::string id, const std::pair<std::string, nc_att_t>& att)
    {
        int varid = name2varid( id, "Can't set attribute in a closed file!");
        dg::file::set_att( m_grp, varid, att);
    }

    template<class S, class T> // T cannot be nc_att_t
    void set_att( std::string id, const std::tuple<S,nc_type, T>& att)
    {
        int varid = name2varid( id, "Can't set attribute in a closed file!");
        dg::file::set_att( m_grp, varid, att);
    }
    // Iterable can be e.g. std::vector<std::pair...>, std::map , etc.
    template<class Iterable> // *it must be usable in set_att
    void set_atts( std::string id, const Iterable& atts)
    {
        int varid = name2varid( id, "Can't set attributes in a closed file!");
        dg::file::set_atts( m_grp, varid, atts);
    }
    void set_atts( std::string id, const std::map<std::string, nc_att_t>& atts)
    {
        // Help compiler choose
        int varid = name2varid( id, "Can't set attributes in a closed file!");
        dg::file::set_atts( m_grp, varid, atts);
    }

    /////////////////// Attribute getters

    dg::file::nc_att_t get_att_t( std::string id, std::string att_name) const
    {
        int varid = name2varid( id, "Can't get attribute in a closed file!");
        return dg::file::get_att_t( m_grp, varid, att_name);
    }

    template<class T>
    T get_att_i( std::string id, std::string att_name, unsigned idx = 0) const
    {
        auto vec = get_att_v<T>( id, att_name);
        return vec[idx];
    }

    // This works for compound types
    template<class T>
    std::vector<T> get_att_v( std::string id, std::string att_name) const
    {
        int varid = name2varid( id, "Can't get attribute in a closed file!");
        return dg::file::get_att_v<T>( m_grp, varid, att_name);
    }
    template<class T>
    T get_att( std::string id, std::string att_name) const
    {
        nc_att_t att = get_att_t( id, att_name);
        return std::get<T>( att);
    }


    // Get all attributes of a given type
    template<class T> // T can be nc_att_t
    std::map<std::string, T> get_atts( std::string id = ".") const
    {
        int varid = name2varid( id, "Can't get attributes in a closed file!");
        return dg::file::get_atts<T>( m_grp, varid);
    }
    //std::vector<std::tuple<std::string, nc_type, std::any>> get_atts( std::string id = ".") const;

    /// Remove an attribute
    /// Note that you cannot delete attributes or dimensions or groups
    void del_att( std::string id, std::string att)
    {
        int varid = name2varid( id, "Can't delete attribute in a closed file!");
        auto name = att.c_str();
        NC_Error_Handle err;
        err = nc_del_att( m_grp, varid, name);
    }
    /// Rename an attribute
    void rename_att( std::string id, std::string old_att_name, std::string new_att_name)
    {
        int varid = name2varid( id, "Can't delete attribute in a closed file!");
        auto old_name = old_att_name.c_str();
        auto new_name = new_att_name.c_str();
        NC_Error_Handle err;
        err = nc_rename_att( m_grp, varid, old_name, new_name);
    }


    ////////////// Variables ////////////////////////
    // Overwrite existing?
    template<class T>
    void def_var( std::string name, std::vector<std::string> dim_names)
    {
        def_var( NcVariable{ name, detail::getNCDataType<T>(), dim_names});
    }
    void def_var( const NcVariable& var)
    {
        file::NC_Error_Handle err;
        std::vector<int> dimids( var.dims.size());
        for( unsigned u=0; u<var.dims.size(); u++)
            nc_inq_dimid( m_grp, var.dims[u].c_str(), &dimids[u]);
        int varid;
        err = nc_def_var( m_grp, var.name.c_str(), var.xtype,
                var.dims.size(), &dimids[0],
                &varid);
    }

    template<class ContainerType>
    void put_var( std::string name, const NcHyperslab& slab,
            const ContainerType& data)
    {
        int varid = name2varid( name, "Can't write variable in a closed file!");
        file::NC_Error_Handle err;
        if constexpr ( std::is_same_v<dg::get_execution_policy<ContainerType>,
            dg::CudaTag>)
        {
            using value_type = dg::get_value_type<ContainerType>;
            m_buffer.template set<value_type>( data.size());
            const auto& buffer = m_buffer.template get<value_type>( );
            dg::assign ( data, buffer);
            err = detail::put_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                buffer.data());
        }
        else
            err = detail::put_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                data.data());
    }

    template<class T>
    void put_var1( std::string name, const std::vector<size_t>& start, T data)
    {
        int varid = name2varid( name, "Can't write variable in a closed file!");
        file::NC_Error_Handle err;
        std::vector<size_t> count( start.size(), 1);
        err = detail::put_vara_T( m_grp, varid, &start[0], &count[0], &data);
    }


    template<class ContainerType>
    void get_var( std::string name, const NcHyperslab& slab,
            ContainerType& data) const
    {
        int varid = name2varid( name, "Can't write variable in a closed file!");
        file::NC_Error_Handle err;
        if constexpr ( std::is_same_v<dg::get_execution_policy<ContainerType>,
            dg::CudaTag>)
        {
            using value_type = dg::get_value_type<ContainerType>;
            m_buffer.template set<value_type>( data.size());
            const auto& buffer = m_buffer.template get<value_type>( );
            err = detail::get_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                buffer.data());
            dg::assign ( buffer, data);
        }
        else
            err = detail::get_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                data.data());
    }

    bool var_exists( std::string name) const
    {
        int varid=0;
        int retval = nc_inq_varid( m_grp, name.c_str(), &varid);
        return retval == NC_NOERR;
    }

    std::vector<NcVariable> get_vars() const
    {
        int num_vars = 0, num_dims;
        file::NC_Error_Handle err;
        err = nc_inq(m_grp, &num_dims, &num_vars, NULL, NULL);
        // https://docs.unidata.ucar.edu/netcdf-c/current/reading_unknown.html
        std::vector<NcVariable> vars;
        for( int i=0; i<num_vars; i++)
        {
            char name[NC_MAX_NAME+1]; // 256
            int xtype;
            int ndims;
            err = nc_inq_varndims( m_grp, i, &ndims);
            int dimIDs[ndims];
            err = nc_inq_var( m_grp, i, name, &xtype, NULL, dimIDs, NULL);
            std::vector<std::string> dim_names;
            for( unsigned u=0; u<(unsigned)ndims; u++)
            {
                char dim_name[NC_MAX_NAME+1]; // 256
                size_t len;
                err = nc_inq_dim( m_grp, dimIDs[u], dim_name, &len);
                dim_names.push_back( dim_name);
            }
            vars.push_back( {name, xtype, dim_names});
        }
        return vars;
    }
    template<class ContainerType>
    void put_var( std::string name, const ContainerType& data)
    {
        std::vector<size_t> count( 1, data.size());
        std::vector<size_t> start( 1, 0);
        put_var( name, { start, count}, data);
    }
    template<class T>
    void defput_dim( std::string name, size_t size,
            std::map<std::string, nc_att_t> atts)
    {
        def_dim( name, size);
        def_var<T>( name, {name});
        set_atts( name, atts);
    }

    template<class ContainerType>
    void defput_dim( std::string name,
            std::map<std::string, nc_att_t> atts,
            const ContainerType& abscissas)
    {
        def_dim( name, abscissas.size());
        def_var<dg::get_value_type<ContainerType>>( name, {name});
        set_atts( name, atts);
        std::vector<size_t> count( 1, abscissas.size());
        std::vector<size_t> start( 1, 0);
        put_var( name, { start, count}, abscissas);
    }

    private:
    int name2varid( std::string id, std::string error_message) const
    {
        // This is fast even for lots variables (1000 vars take <ms)
        if( !m_open)
            throw std::runtime_error( error_message );
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
    // Absolute current path
    std::filesystem::path get_grp_path( int ncid ) const
    {
        size_t len;
        NC_Error_Handle err;
        err = nc_inq_grpname_full( ncid, &len, NULL);
        std::string current( len, 'x');
        err = nc_inq_grpname_full( ncid, &len, &current[0]);
        return current;
    }
    // Absolute paths of subgroups
    std::map<int, std::filesystem::path> get_grps( int ncid ) const
    {
        NC_Error_Handle err;
        int num_grps;
        err = nc_inq_grps( ncid, &num_grps, NULL);
        if( num_grps == 0)
            return {};
        std::vector<int> group_ids( num_grps);
        err = nc_inq_grps( ncid, &num_grps, &group_ids[0]);
        std::map<int, std::filesystem::path> groups;
        for( int i=0; i<num_grps; i++)
        {
            size_t len;
            err = nc_inq_grpname_full( group_ids[i], &len, NULL);
            std::string name( len, 'z');
            err = nc_inq_grpname_full( group_ids[i], &len, &name[0]);
            groups[group_ids[i]] = name;
        }
        return groups;
    }
    std::map<int, std::filesystem::path> get_grps_r( int ncid) const
    {
        auto grps = get_grps(ncid);
        for( auto grp : grps)
        {
            auto subgrps = get_grps_r( grp.first);
            grps.merge( subgrps);
        }
        return grps;
    }

    bool m_open = false;
    int m_ncid = 0; // ncid can be different by opening the same file twice
    int m_grp = 0; // the currently active group (All group ids in open files are unique and thus group ids can be different by opening the same file twice), dims can be seen by all child groups

    // Buffer for device to host transfer, and dg::assign
    dg::detail::AnyVector<thrust::host_vector> m_buffer;
    std::map<std::string,std::pair<int,unsigned>> m_varids; //first is ID, second is the slice to write to next == length
};

#ifndef MPI_VERSION
using NcFile = SerialNcFile;
#endif // MPI_VERSION

}// namespace file
}// namespace dg
