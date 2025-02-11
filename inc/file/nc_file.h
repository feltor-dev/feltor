#pragma once

#include <filesystem>
#include <functional>
#include <list>
#include "../dg/blas.h"
#include "../dg/backend/memory.h"
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
/*! @class hide_grps_NetCDF_example
 * Groups can be thought of as directories in a NetCDF-4 file and we therefore
 * use \c std::filesystem::path and use bash equivalent operations to
 * manipulate them
 * @snippet{trimleft} nc_file_t.cpp groups
 */
/*! @class hide_dimension_hiding
 *
 * @note Dimensions are **visible** in a group and all of its subgroups.  Now,
 * a file can define multiple dimensions with the same name in subgroups. The
 * dimension in the highest group will then hide the ones in the lower groups
 * and e.g a call to \c def_var cannot currently use a dimension that is hidden
 * in such a way even though the NetCDF-C API would allow to do it using the
 * dimension ID. In contrast, \c get_var_dims may return dimension names that
 * are actually hidden. It is therefore highly recommended to use unique names
 * for all dimensions in the file.
 * Furthermore, notice that while a dimension is visible in subgroups,
 * its associated dimension variable is not.
 * @attention Paraview seems to have issues if dimensions in a subgrup in a
 * NetCDF file are defined before any of the dimensions in the root group.
 *
 * @class hide_attributes_overwrite
 * @note Attributes are silently overwritten. You need to manually
 * check with \c att_is_defined for existence if this is a concern
 *
 * @class hide_container_type
 * @tparam ContainerType May be anything that \c dg::get_tensor_category
 * recognises as a \c dg::SharedVectorTag. In \c defput* members the value type
 * determines the NetCDF type of the variable to define
 * @note In both \c put* and \c get* members it is possible for \c data to have
 * a different value type from the defined value type of the variable that is
 * being written/read, the NetCDF C-API simply converts it to
 * the requested type. For example a variable declared as \c NC_DOUBLE can be
 * read as/written from integer or float and vice versa.
 */
/*! @class hide_unlimited_issue
 *
 * @note There are a couple of subtle issues related to **unlimited dimensions** in NetCDF.
 * First of all, all variables that share a dimension in NetCDF have the same size
 * along that dimension.  Second, it is possible to write data to any index
 * (the start value of the hyperslab in the unlimited dimension can be
 * anything) of a variable with an unlimited dimension. For example you can do
 * @code{.cpp}
 * file.def_dim( "time", NC_UNLIMITED);
 * file.def_var_as<double>( "var", {"time"}, {});
 * file.put_var( "var", {5}, 42); // perfectly legal
 * @endcode
 * The size of an unlimited dimension is the maximum of the sizes of all
 * variables that share this dimension. If you are trying to read a variable at a slice
 * where data was never written, the NetCDF library just fills it up with Fill Values.
 * However, trying to read beyond the size of the unlimited dimension will fail.
 * If a user wants to keep all unlimited variables synchronised, they unfortunately have
 * to keep track of which variable was written themselves:
 * See <a href="https://github.com/Unidata/netcdf-c/issues/1898">NetCDF issue</a>
 */

/*! @brief NetCDF file format
 *
 * All Files are opened/ created in Netcdf-4 data format
 * @note If you are looking for an "nc_append" you can use
@code{.cpp}
auto nc_append = std::filesystem::exists(filename) ? nc_write : nc_noclobber;
@endcode
@ingroup utilities
*/
enum NcFileMode
{
    nc_nowrite,  //!< NC_NOWRITE Open an existing file for read-only access, fail if it does not exist
    nc_write,    //!< NC_WRITE Open an existing file for read and write access, fail if it does not exist
    nc_clobber, //!< NC_CLOBBER Create a new file for read and write access, overwrite if file exists
    nc_noclobber, //!< NC_NOCLOBBER Create new file for read and write access, fail if already exists
};

/*! @brief Serial NetCDF-4 file
 *
 * Our take on a modern C++ implementation of
<a href="https://docs.unidata.ucar.edu/netcdf-c/4.9.2/netcdf_data_model.html">the NetCDF-4 data model</a>
 *
 * See here a usage example
 * @snippet nc_utilities_t.cpp ncfile
 *
 * @note This is a singleton that cannot be copied/assigned but only
 * moved/move-assign
 * @note Only Netcdf-4 files are supported
 * @note The class hides all integer ids that the NetCDF C-library uses
 * ("Ids do not exist in the Netcdf-4 data model!")
 * @note Most member functions will throw if they are called on a closed file
 * @sa Conventions to follow are the
 <a href="http://cfconventions.org/Data/cf-conventions/cf-conventions-1.9/cf-conventions.html">CF-conventions</a>
 and
 <a href="https://docs.unidata.ucar.edu/nug/current/best_practices.html">netCDF conventions</a>
 * @ingroup ncfile
*/
struct SerialNcFile
{
    using Hyperslab = NcHyperslab;
    // ///////////////////////////// CONSTRUCTORS/DESTRUCTOR /////////
    /// Construct a File Handle not associated to any file
    /// @snippet{trimleft} nc_file_t.cpp default
    SerialNcFile () = default;
    /*!
     * @brief Open/Create a netCDF file.
     * @param filename  Name or path including the name of the netCDF file to
     * open or create. The path may be either absolute or **relative to the
     * execution path of the program** i.e. relative to \c
     * std::filesystem::current_path()
     * @param mode (see \c NcFileMode for nc_nowrite, nc_write, nc_clobber,
     * nc_noclobber)
     *
     * @snippet{trimleft} nc_file_t.cpp constructor
     * @sa NcFileMode
     */
    SerialNcFile(const std::filesystem::path& filename,
            enum NcFileMode mode = nc_nowrite)
    {
        open( filename, mode);
    }
    /*! @brief  There can only be exactly one file handle per physical file.
     *
     * The reason is that the destructor releases all resources and thus a copy
     * of the file that is subsequently destroyed leaves the original in an
     * invalid state
     */
    SerialNcFile(const SerialNcFile& rhs) = delete;
    ///@copydoc SerialNcFile::SerialNcFile(const SerialNcFile&)
    SerialNcFile& operator =(const SerialNcFile & rhs) = delete;

    /*! @brief Swap resources between two file handles
     */
    SerialNcFile(SerialNcFile&& rhs) = default;
    ///@copydoc SerialNcFile::SerialNcFile(SerialNcFile&&)
    SerialNcFile& operator =(SerialNcFile && rhs) = default;

    /// @brief Close open nc file and release all resources
    /// @note A destructor never throws any errors and we will just print a
    /// warning to \c std::cerr if something goes wrong
    ~SerialNcFile()
    {
        try
        {
            close();
        }
        catch (NC_Error &e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
    // /////////////////// open/close /////////

    /*! Explicitly open or create a netCDF file
     *
     * @param filename  Name or path including the name of the netCDF file to
     * open or create. The path may be either absolute or **relative to the
     * execution path of the program** i.e. relative to \c
     * std::filesystem::current_path()
     * @param mode (see \c NcFileMode for nc_nowrite, nc_write, nc_clobber,
     * nc_noclobber)
     * @note Just like \c std::fstream opening fails if a file is already
     * associated (\c is_open()) \c close() it before opening a new file.
     *
     * @snippet{trimleft} nc_file_t.cpp default
    */
    void open(const std::filesystem::path& filename,
            enum NcFileMode mode = nc_nowrite)
    {
        // Like a std::fstream opening fails if file already associated
        if( m_open)
            throw NC_Error( 1002);

        // TODO Test the pathing on Windows
        NC_Error_Handle err;
        switch (mode)
        {
            case nc_nowrite:
                err = nc_open( filename.string().c_str(), NC_NETCDF4 |
                        NC_NOWRITE, &m_ncid);
                break;
            case nc_write:
                err = nc_open( filename.string().c_str(), NC_NETCDF4 |
                        NC_WRITE, &m_ncid);
                break;
            case nc_noclobber:
                err = nc_create( filename.string().c_str(), NC_NETCDF4 |
                        NC_NOCLOBBER, &m_ncid);
                break;
            case nc_clobber:
                err = nc_create( filename.string().c_str(), NC_NETCDF4 |
                        NC_CLOBBER, &m_ncid);
                break;
        }
        m_open = true;
        m_grp = m_ncid;
    }
    /// Check if a file is associated (i.e. it is open)
    /// @snippet{trimleft} nc_file_t.cpp default
    bool is_open() const noexcept
    {
        // is_closed() == not is_open()
        return m_open;
    }

    /*! @brief Explicitly close a file
     *
     * Closing a file triggers all buffers to be written and memory to be
     * released. After closing a new file can be associated to this handle
     * again.
     *
     * @snippet{trimleft} nc_file_t.cpp default
     * @note This function may throw
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
        check_open();
        NC_Error_Handle err;
        err = nc_sync( m_ncid);
    }
    /*! @brief Get the \c ncid of the underlying NetCDF C-API
     *
     * Just if for whatever reason you want to call a NetCDF C-function
     * yourself ... just don't use it for something nasty, like
     * closing the file or whatever
     */
    int get_ncid() const noexcept{ return m_ncid;}

    // ///////////// Groups /////////////////
    /*! Define a group named \c name in the current group
     *
     * @copydoc hide_grps_NetCDF_example
     * Think of this as the bash command \c mkdir name
     * @param name of the new group
     * @note Just like in a filesystem 2 nested groups with same name can exist
     * but not 2 groups with the same name in the same group.
     */
    void def_grp( std::string name)
    {
        check_open();
        int new_grp_id = 0;
        NC_Error_Handle err;
        err = nc_def_grp( m_grp, name.c_str(), &new_grp_id);
        // we just forget the id => always need to ask netcdf for id
        // Is no performance hit as cached
    }

    /*! Define a group named \c path and all required intermediary groups
     *
     * @copydoc hide_grps_NetCDF_example
     * Think of this as the bash command \c mkdir -p path
     * @param path of the new group. Can be absolute or relative to the current
     * group
     * @note Just like in a filesystem 2 nested groups with same name can exist
     * but not 2 groups with the same name in the same group.
     */
    void def_grp_p( std::filesystem::path path)
    {
        check_open();
        if( not path.has_root_path()) // it is a relative path
        {
            auto current = get_current_path();
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
    /*! @brief Check for existence of the group given by path
     * @copydoc hide_grps_NetCDF_example
     * @param path Absolute or relative path to the current group
     */
    bool grp_is_defined( std::filesystem::path path) const
    {
        check_open();
        std::string name = path.generic_string();
        if( not path.has_root_path()) // it is a relative path
        {
            auto current = get_current_path();
            name = (current / path ).generic_string();
        }
        int grpid=0;
        int retval = nc_inq_grp_full_ncid( m_ncid, name.c_str(), &grpid);
        return retval == NC_NOERR;
    }
    /*! @brief Change group to \c path
     *
     * @copydoc hide_grps_NetCDF_example
     * All subsequent calls to atts, dims and vars are made to that group
     *
     * @param path can be absolute or relative to the current group.
     * Empty string or "/" goes back to root group. "." is the current group
     * and returns immediately.
     */
    void set_grp( std::filesystem::path path = "")
    {
        check_open();
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
                auto current = get_current_path();
                name = (current / path ).generic_string();
            }
            err = nc_inq_grp_full_ncid( m_ncid, name.c_str(), &m_grp);
        }
    }
    /// rename a subgroup in the current group from \c old_name to \c new_name
    /// @copydoc hide_grps_NetCDF_example
    void rename_grp( std::string old_name, std::string new_name)
    {
        check_open();
        NC_Error_Handle err;
        int old_grp;
        err = nc_inq_grp_ncid( m_grp, old_name.c_str(), &old_grp);
        err = nc_rename_grp( old_grp, new_name.c_str());
    }

    /// Get the NetCDF-C ID of the current group
    int get_grpid() const noexcept{ return m_grp;}

    /// Get the absolute path of the current group
    std::filesystem::path get_current_path( ) const
    {
        check_open();
        return get_grp_path( m_grp);
    }

    /// Get all subgroups in the current group as absolute paths
    /// @copydoc hide_grps_NetCDF_example
    std::list<std::filesystem::path> get_grps( ) const
    {
        check_open();
        auto grps = get_grps_abs(m_grp);
        std::list<std::filesystem::path> grps_v;
        for( auto grp : grps)
            grps_v.push_back( grp.second);
        return grps_v;

    }
    /*! @brief Get all subgroups recursively in the current group as absolute paths
     *
     * Think of this as \c ls -R
     * @copydoc hide_grps_NetCDF_example
     * @note Using the vector of paths it is possible to traverse
     * the entire filesystem
     *
     * @return All groups and subgroups
     * @attention Does not include the current group
     */
    std::list<std::filesystem::path> get_grps_r( ) const
    {
        auto grps = get_grps_abs_r(m_grp);
        std::list<std::filesystem::path> grps_v;
        for( auto grp : grps)
            grps_v.push_back( grp.second);
        return grps_v;
    }

    // //////////// Dimensions ////////////////////////
    /*! @brief Define a dimension named \c name of size \c size
     *
     * @note Remember that dimensions do not have attributes or types,
     * only variables and groups have attributes and only variables have types
     * @note One often defines an associated **dimension variable**
     * with the same name as the dimension.
     * @param name of the dimension. Cannot be the same name as a dimension
     * name already existing in the current group
     * @param size Size of the dimension to create. Use \c NC_UNLIMITED to
     * create an unlimited dimension
     * @copydoc hide_unlimited_issue
     * @copydoc hide_dimension_hiding
     */
    void def_dim( std::string name, size_t size)
    {
        check_open();
        NC_Error_Handle err;
        int dim_id;
        err = nc_def_dim( m_grp, name.c_str(), size, &dim_id);
    }
    /// Rename a dimension from \c old_name to \c new_name
    void rename_dim( std::string old_name, std::string new_name)
    {
        check_open();
        int dimid;
        NC_Error_Handle err;
        err = nc_inq_dimid( m_grp, old_name.c_str(), &dimid);
        err = nc_rename_dim( m_grp, dimid, new_name.c_str());
    }
    /*! @brief Get the size of the dimension named \c name
     *
     * @snippet{trimleft} nc_utilities_t.cpp get_dim_size
     * @note The size of an unlimited dimension is the maximum of the sizes
     * of all variables that share this dimension.
     * @copydoc hide_unlimited_issue
     */
    size_t get_dim_size( std::string name) const
    {
        check_open();
        NC_Error_Handle err;
        int dimid;
        err = nc_inq_dimid( m_grp, name.c_str(), &dimid);
        size_t len;
        err = nc_inq_dimlen( m_grp, dimid, &len);
        return len;

    }

    /// Get the size of each dimension in \c dims
    std::vector<size_t> get_dims_shape( const std::vector<std::string>& dims) const
    {
        std::vector<size_t> shape( dims.size());
        for( unsigned u=0; u<dims.size(); u++)
            shape[u] = get_dim_size( dims[u]);
        return shape;
    }
    /*! Get all visible dimension names in the current group
     *
     * The visible dimensions are all the dimensions in the current group
     * and all its parent group.
     * @copydoc hide_dimension_hiding
     * @param include_parents per default the parent groups will be included in
     * the search for dimensions, if \c false they are excluded
     * @return All visible dimension names. Hidden names do not appear.
     * @snippet{trimleft} nc_file_t.cpp get_dims
     */
    std::vector<std::string> get_dims( bool include_parents = true) const
    {
        check_open();
        NC_Error_Handle err;
        // There is a correct way of getting the dimensions in a group
        // https://github.com/Unidata/netcdf-c/issues/2873
        int ndims;
        err = nc_inq_dimids( m_grp, &ndims, NULL, include_parents);
        if( ndims == 0)
            return {};
        int dimids[ndims];
        err = nc_inq_dimids( m_grp, &ndims, dimids, include_parents);
        // Globally dimension ids are 0, 1, 2, ... in the order in which the
        // dimensions were defined
        std::vector<std::string> dims;
        for ( int i = 0; i < ndims; i++)
        {
            char dimname [ NC_MAX_NAME+1];
            err = nc_inq_dimname( m_grp, dimids[i], dimname);
            if( std::find(dims.begin(), dims.end(), std::string(dimname) ) ==
                    dims.end())
                dims.push_back(dimname);
        }
        return dims;
    }

    /*! Get all visible unlimited dimension names in the current group
     *
     * @copydoc hide_dimension_hiding
     * This function **does not** include the parent groups
     * @return All unlimited dimension names in current group.
     */
    std::vector<std::string> get_unlim_dims( ) const
    {
        check_open();
        int ndims;
        NC_Error_Handle err;
        err = nc_inq_unlimdims( m_grp, &ndims, NULL);
        if( ndims == 0)
            return {};
        int dimids[ndims];
        // Our tests indicate that this does not return the unlimited dimensions
        // of the parent group even though the documentation says so...
        err = nc_inq_unlimdims( m_grp, &ndims, dimids);
        std::vector<std::string> dims;
        for( int i=0; i<ndims; i++)
        {
            char dimname [ NC_MAX_NAME+1];
            err = nc_inq_dimname( m_grp, dimids[i], dimname);
            if( std::find(dims.begin(), dims.end(), std::string(dimname) ) ==
                    dims.end())
                dims.push_back(dimname);
        }
        return dims;
    }

    /// Check for existence of the dimension named \c name
    /// @snippet{trimleft} nc_utilities_t.cpp check_dim
    bool dim_is_defined( std::string name) const
    {
        int dimid=0;
        int retval = nc_inq_dimid( m_grp, name.c_str(), &dimid);
        return retval == NC_NOERR;
    }
    /////////////// Attributes setters
    /*! @brief Put an individual attribute
     * @snippet{trimleft} nc_file_t.cpp put_att
     * @param att Attribute consisting of name and value
     * @param id Variable name in the current group or empty string, in which
     * case the attribute refers to the current group
     * @copydoc hide_attributes_overwrite
     */
    void put_att( const std::pair<std::string, nc_att_t>& att, std::string id = "")
    {
        int varid = name2varid( id);
        dg::file::detail::put_att( m_grp, varid, att);
    }

    /*! @brief Put an individual attribute of preset type to variable id
     * @tparam S std::string or const char*
     * @tparam T Cannot be an nc_att_t
     * @param att Attribute consisting of name, type and value
     * @param id Variable name in the current group or empty string, in which
     * case the attribute refers to the current group
     * @copydoc hide_attributes_overwrite
     * @snippet{trimleft} nc_file_t.cpp put_att_x
     */
    template<class S, class T> // T cannot be nc_att_t
    void put_att( const std::tuple<S,nc_type, T>& att, std::string id = "")
    {
        int varid = name2varid( id);
        dg::file::detail::put_att( m_grp, varid, att);
    }
    /*! @brief Write a collection of attributes to a NetCDF variable or file
     *
     * Example code
     * @snippet{trimleft} nc_file_t.cpp put_atts
     * @note boolean values are mapped to byte NetCDF attributes (0b for true,
     * 1b for false)
     * @tparam Attributes Any \c Iterable whose values can be used in \c put_att
     * i.e. either a \c std::pair or \c std::tuple
     * @param atts An iterable containing all the attributes for the variable
     * or file. \c atts can be empty in which case no attribute is written.
     * @param id Variable name in the current group or empty string, in which
     * case the attributes refer to the current group
     * @copydoc hide_attributes_overwrite
     */
    template<class Attributes = std::map<std::string, nc_att_t> > // *it must be usable in put_att
    void put_atts(const Attributes& atts, std::string id = "")
    {
        int varid = name2varid( id);
        dg::file::detail::put_atts( m_grp, varid, atts);
    }

    // ///////////////// Attribute getters

    /*! @brief Get an attribute named \c att_name of the group or variable \c id
     *
     * @tparam T Any type in \c dg::file::nc_att_t or \c nc_att_t
     * in which case the type specific nc attribute getters are called
     * or \c std::vector<type> in which case the general \c nc_get_att is called
     * @param id Variable name in the current group or empty string, in which
     * case the attribute refers to the current group
     * @param att_name Name of the attribute
     * @return Attribute cast to type \c T
     * @snippet{trimleft} nc_file_t.cpp put_att_x
     */
    template<class T>
    T get_att_as(std::string att_name, std::string id = "") const
    {
        int varid = name2varid( id);
        return dg::file::detail::get_att_as<T>( m_grp, varid, att_name);
    }

    /// Short for <tt> get_att_as<std::vector<T>>( id, att_name);</tt>
    template<class T>
    std::vector<T> get_att_vec_as(std::string att_name, std::string id = "") const
    {
        return get_att_as<std::vector<T>>( att_name, id);
    }

    /*! @brief Read all NetCDF attributes of a certain type
     *
     * For example
     * @note byte attributes are mapped to boolean values (0b for true, 1b for false)
     * @return A Dictionary containing all the attributes of a certain type
     * for the variable or file. Can be empty if no attribute is present.
     * @param id Variable name in the current group or empty string, in which
     * case the attributes refer to the current group
     * @tparam T can be a primitive type like \c int or \c double or a vector
     * thereof \c std::vector<int> or a \c dg::file::nc_att_t in which case
     * attributes of heterogeneous types are captured
     */
    template<class T>
    std::map<std::string, T> get_atts_as( std::string id = "") const
    {
        int varid = name2varid( id);
        return dg::file::detail::get_atts_as<T>( m_grp, varid);
    }

    /// Short for <tt> get_atts_as<nc_att_t>( id) </tt>
    /// @snippet{trimleft} nc_file_t.cpp put_atts
    std::map<std::string, nc_att_t> get_atts( std::string id = "") const
    {
        return get_atts_as<nc_att_t>( id);
    }

    /*!
     * @brief Remove an attribute named \c att_name from variable \c id
     * @param att_name Attribute to delete
     * @param id Variable name in the current group or empty string, in which
     * case the attributes refer to the current group
     * @note Attributes are the only thing you can delete in a NetCDF file.
     * You cannot delete variables or dimensions or groups
     *
     * @snippet{trimleft} nc_file_t.cpp del_att
     */
    void del_att(std::string att_name, std::string id = "")
    {
        int varid = name2varid( id);
        auto name = att_name.c_str();
        NC_Error_Handle err;
        err = nc_del_att( m_grp, varid, name);
    }
    /// Check for existence of the attribute named \c att_name in variable \c id
    /// @snippet{trimleft} nc_file_t.cpp del_att
    bool att_is_defined(std::string att_name, std::string id = "") const
    {
        int varid = name2varid( id);
        int attid;
        int retval = nc_inq_attid( m_grp, varid, att_name.c_str(), &attid);
        return retval == NC_NOERR;
    }
    /// Rename an attribute
    /// @snippet{trimleft} nc_file_t.cpp rename_att
    void rename_att(std::string old_att_name, std::string
            new_att_name, std::string id = "")
    {
        int varid = name2varid( id);
        auto old_name = old_att_name.c_str();
        auto new_name = new_att_name.c_str();
        NC_Error_Handle err;
        err = nc_rename_att( m_grp, varid, old_name, new_name);
    }

    // //////////// Variables ////////////////////////
    /*! @brief Define a variable with given type, dimensions and (optionally)
     * attributes
     * @param name Name of the variable to define
     * @param dim_names Names of visible dimensions in the current group.
     * Can be empty which makes the defined variable a scalar w/o dimensions.
     * @copydoc hide_dimension_order
     * @copydoc hide_dimension_hiding
     * @tparam T set the type of the variable
     * @param atts Attributes to put for the variable
     *
     * @snippet{trimleft} nc_file_t.cpp def_var_as
     */
    template<class T, class Attributes = std::map<std::string, nc_att_t>>
    void def_var_as( std::string name,
        const std::vector<std::string>& dim_names,
        const Attributes& atts = {})
    {
        def_var( name, detail::getNCDataType<T>(), dim_names, atts);
    }

    /*! @brief Define a variable with given type, dimensions and (optionally)
     * attributes
     * @param name Name of the variable to define
     * @param xtype NetCDF typeid
     * @param dim_names Names of visible dimensions in the current group
     * Can be empty which makes the defined variable a scalar w/o dimensions.
     * @copydoc hide_dimension_order
     * @copydoc hide_dimension_hiding
     * @param atts Attributes to put for the variable
     * @note This function overload is useful if you want to use a compound type
     */
    template<class Attributes = std::map<std::string, nc_att_t>>
    void def_var( std::string name, nc_type xtype,
            const std::vector<std::string>& dim_names,
            const Attributes& atts = {})
    {
        file::NC_Error_Handle err;
        std::vector<int> dimids( dim_names.size());
        for( unsigned u=0; u<dim_names.size(); u++)
            err = nc_inq_dimid( m_grp, dim_names[u].c_str(), &dimids[u]);
        int varid;
        err = nc_def_var( m_grp, name.c_str(), xtype, dim_names.size(),
                &dimids[0], &varid);
        put_atts( atts, name);
    }

    /*! @brief Write data to a variable
     * @param name Name of the variable to write data to. Must be visible in
     * the current group
     * @param slab Define where the data is written. The dimension of the slab
     * \c slab.ndim() must match the number of dimensions of the variable
     * @param data to write. Size must be at least that of the slab
     * @copydoc hide_container_type
     * @copydoc hide_unlimited_issue
     *
     * @snippet{trimleft} nc_utilities_t.cpp put_var
     */
    template<class ContainerType, std::enable_if_t< dg::is_vector_v<
        ContainerType, SharedVectorTag>, bool > = true>
    void put_var( std::string name, const NcHyperslab& slab,
            const ContainerType& data)
    {
        int varid = name2varid( name);
        file::NC_Error_Handle err;
        int ndims;
        err = nc_inq_varndims( m_grp, varid, &ndims);
        assert( (unsigned)ndims == slab.ndim());
        if constexpr ( dg::has_policy_v<ContainerType, dg::CudaTag>)
        {
            using value_type = dg::get_value_type<ContainerType>;
            m_buffer.template set<value_type>( data.size());
            auto& buffer = m_buffer.template get<value_type>( );
            dg::assign ( data, buffer);
            err = detail::put_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                buffer.data());
        }
        else
            err = detail::put_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                thrust::raw_pointer_cast(data.data()));
    }

    /*! @brief Define and put a variable in one go
     *
     * Very convenient to "just write" a variable to a file:
     * @snippet{trimleft} nc_utilities_t.cpp defput_var
     *
     * Short for
     *@code{.cpp}
     *   def_var_as<dg::get_value_type<ContainerType>>( name, dim_names, atts);
     *   put_var( name, slab, data);
     *@endcode
     */
    template<class ContainerType, class Attributes = std::map<std::string, nc_att_t>,
        std::enable_if_t< dg::is_vector_v<ContainerType, SharedVectorTag>, bool > = true>
    void defput_var( std::string name, const std::vector<std::string>& dim_names,
            const Attributes& atts, const NcHyperslab& slab,
            const ContainerType& data)
    {
        def_var_as<dg::get_value_type<ContainerType>>( name, dim_names, atts);
        put_var( name, slab, data);
    }

    /*! @brief Write a single data point
     * @param name Name of the variable to write data to. Must be visible in
     * the current group
     * @param start The coordinates (one for each dimension) to which to write
     * data to
     * @param data to write
     * @tparam T must be convertible to the datatype of the variable \c name
     *
     * @snippet{trimleft} nc_utilities_t.cpp def_dimvar
     * @copydoc hide_unlimited_issue
     */
    template<class T, std::enable_if_t< dg::is_scalar_v<T>, bool> = true>
    void put_var( std::string name, const std::vector<size_t>& start, T data)
    {
        int varid = name2varid( name);
        file::NC_Error_Handle err;
        std::vector<size_t> count( start.size(), 1);
        err = detail::put_vara_T( m_grp, varid, &start[0], &count[0], &data);
    }

    /*! @brief Define a dimension and dimension variable in one go
     *
     * Short for
     * @code{.cpp}
     * def_dim( name, size);
     * def_var_as<T>( name, {name}, atts);
     * @endcode
     * @param name Name of the dimension and associated dimension variable
     * @param size Size of the dimension to create. Use \c NC_UNLIMITED to
     * create an unlimited dimension
     * @param atts Suggested attribute is "axis" : "T" which enable paraview to
     * recognize the dimension as the time axis
     * @note This function is mainly intended to define an unlimited dimension
     * as in
     * @snippet{trimleft} nc_utilities_t.cpp def_dimvar
     * @copydoc hide_unlimited_issue
     * @copydoc hide_dimension_hiding
     */
    template<class T, class Attributes = std::map<std::string, nc_att_t>>
    void def_dimvar_as( std::string name, size_t size, const Attributes& atts)
    {
        def_dim( name, size);
        def_var_as<T>( name, {name}, atts);
    }

    /*!
     * @brief Define a dimension and define and write to a dimension variable
     * in one go
     *
     * @param name Name of the dimension and associated dimension variable
     * @param atts Suggested attributes is for example "axis" : "X" which
     * enable paraview to recognize the dimension as the x axis and "long_name"
     * : "X-coordinate in Cartesian coordinates"
     * @param abscissas values to write to the dimension variable (the
     * dimension size is inferred from \c abscissas.size() and the type is
     * inferred from \c ContainerType
     * @note This function is mainly intended to define dimensions from a grid
     * as in
     * @snippet{trimleft} nc_utilities_t.cpp defput_dim
     * @copydoc hide_container_type
     */
    template<class ContainerType, class Attributes = std::map<std::string, nc_att_t>>
    void defput_dim( std::string name,
            const Attributes& atts,
            const ContainerType& abscissas)
    {
        def_dimvar_as<dg::get_value_type<ContainerType>>( name,
            abscissas.size(), atts);
        put_var( name, {abscissas}, abscissas);
    }

    /*! @brief Read hyperslab \c slab from variable named \c name into
     * container \c data
     *
     * @snippet{trimleft} nc_utilities_t.cpp get_var
     * @param name of previously defined variable
     * @param slab Hyperslab to read
     * @param data Result on output
     * @copydoc hide_container_type
     */
    template<class ContainerType, std::enable_if_t< dg::is_vector_v<
        ContainerType, SharedVectorTag>, bool > = true>
    void get_var( std::string name, const NcHyperslab& slab,
            ContainerType& data) const
    {
        int varid = name2varid( name);
        file::NC_Error_Handle err;
        int ndims;
        err = nc_inq_varndims( m_grp, varid, &ndims);
        assert( (unsigned)ndims == slab.ndim());
        if constexpr ( dg::has_policy_v<ContainerType, dg::CudaTag>)
        {
            using value_type = dg::get_value_type<ContainerType>;
            m_buffer.template set<value_type>( data.size());
            auto& buffer = m_buffer.template get<value_type>( );
            err = detail::get_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                buffer.data());
            dg::assign ( buffer, data);
        }
        else
            err = detail::get_vara_T( m_grp, varid, slab.startp(), slab.countp(),
                thrust::raw_pointer_cast(data.data()));
    }

    /*! @brief Read scalar from position \c start from variable named \c name
     *
     * @snippet{trimleft} nc_file_t.cpp get_var
     * @param name of previously defined variable
     * @param start coordinate to take scalar from (can be empty for scalar
     * variable)
     * @param data Result on output
     * into container \c data
     */
    template<class T, std::enable_if_t< dg::is_scalar_v<T>, bool> = true>
    void get_var( std::string name, const std::vector<size_t>& start, T& data) const
    {
        int varid = name2varid( name);
        file::NC_Error_Handle err;
        int ndims;
        err = nc_inq_varndims( m_grp, varid, &ndims);
        assert( (unsigned)ndims == start.size());
        if( ndims == 0)
            err = detail::get_vara_T( m_grp, varid, NULL, NULL, &data);
        else
        {
            std::vector<size_t> count( start.size(), 1);
            err = detail::get_vara_T( m_grp, varid, &start[0], &count[0], &data);
        }
    }

    /// Check if variable named \c name is defined in the current group
    /// @snippet{trimleft} nc_file_t.cpp var_is_defined
    bool var_is_defined( std::string name) const
    {
        check_open();
        int varid=0;
        int retval = nc_inq_varid( m_grp, name.c_str(), &varid);
        return retval == NC_NOERR;
    }

    /// Get the NetCDF typeid of the variable named \c name
    /// @snippet{trimleft} nc_file_t.cpp get_var_type
    nc_type get_var_type(std::string name) const
    {
        int varid = name2varid( name);
        int xtype;
        file::NC_Error_Handle err;
        err = nc_inq_vartype( m_grp, varid, &xtype);
        return xtype;
    }
    /*! @brief Get the dimension names associated to variable \c name
     *
     * @snippet{trimleft} nc_file_t.cpp get_var_dims
     * @param name of the variable
     * @return list of dimension names associated with variable
     * @copydoc hide_dimension_hiding
     */
    std::vector<std::string> get_var_dims(std::string name) const
    {
        int varid = name2varid( name);
        file::NC_Error_Handle err;

        int ndims;
        err = nc_inq_varndims( m_grp, varid, &ndims);
        if( ndims == 0)
            return {};
        int dimids[ndims];
        err = nc_inq_vardimid( m_grp, varid, dimids);

        std::vector<std::string> dims(ndims);
        for( int i=0; i<ndims; i++)
        {
            char dimname [ NC_MAX_NAME+1];
            err = nc_inq_dimname( m_grp, dimids[i], dimname);
            dims[i] = dimname;
        }
        return dims;
    }

    /*!
     * @brief Get a list of variable names in the current group
     *
     * We use \c std::list here because of how easy it is
     * to sort or filter elemenets there
     * For example
     * @snippet{trimleft} nc_file_t.cpp get_var_names
     * @return list of variable names in current group
     */
    std::list<std::string> get_var_names() const
    {
        check_open();
        return get_var_names_private( m_grp);
    }

    private:
    int name2varid( std::string id) const
    {
        // Variable ids are persistent once created
        // Attribute ids can change if one deletes attributes
        // This is fast even for lots variables (1000 vars take <ms)
        check_open();
        NC_Error_Handle err;
        int varid;
        if ( id == "")
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
    std::map<int, std::filesystem::path> get_grps_abs( int ncid ) const
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
    std::map<int, std::filesystem::path> get_grps_abs_r( int ncid) const
    {
        auto grps = get_grps_abs(ncid);
        for( auto grp : grps)
        {
            auto subgrps = get_grps_abs_r( grp.first);
            grps.merge( subgrps);
        }
        return grps;
    }
    // Need a different name so mpi_file can invoke
    std::list<std::string> get_var_names_private(int grpid) const
    {
        std::list<std::string> vars;
        int num_vars = 0, num_dims;
        file::NC_Error_Handle err;
        err = nc_inq(grpid, &num_dims, &num_vars, NULL, NULL);
        // https://docs.unidata.ucar.edu/netcdf-c/current/reading_unknown.html
        for( int i=0; i<num_vars; i++)
        {
            char name[NC_MAX_NAME+1]; // 256
            err = nc_inq_varname( grpid, i, name);
            vars.push_back( name);
        }
        return vars;
    }


    void check_open() const
    {
        if( !m_open)
            throw NC_Error( 1000);
    }

    bool m_open = false;
    int m_ncid = 0; // ncid can change by closing and re-opening a file
    int m_grp = 0;
    // the currently active group (All group ids in all open files are unique
    // and thus group ids can be different by opening the same file twice),
    // dims can be seen by all child groups

    // Buffer for device to host transfer, and dg::assign
    mutable dg::detail::AnyVector<thrust::host_vector> m_buffer;
    // ABANDONED: Variable tracker (persists on closing and opening a different
    // file) The problem with trying to track is
    // 1) do we track every file that is opened?
    // 2) we cannot prevent someone from opening closing a file here and simply
    // opening the same file somewhere else. The info is lost or corrupted then
    //std::map<std::filesystem::path, std::map<std::filesystem::path, NcVarInfo>>
    //    m_vars;
};

#ifndef MPI_VERSION
/// Convenience typedef for platform independent code
/// @ingroup ncfile
using NcFile = SerialNcFile;
#endif // MPI_VERSION

}// namespace file
}// namespace dg
