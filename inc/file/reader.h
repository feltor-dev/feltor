#pragma once

#include "../dg/backend/typedefs.h"
#include "../dg/topology/fast_interpolation.h"

#include "easy_dims.h"
#include "easy_input.h"
#include "json_netcdf.h"

namespace dg
{
namespace file
{

///@addtogroup Cpp
///@{

/**
 * @brief Manage dimensions and variable ids for reading
 *
 * @copydoc hide_topology_tparam
 * @note In an MPI program all processes must construct the class.
 * @copydoc hide_parallel_read
 */
template<class Topology>
struct Reader
{
    Reader() = default;
    /**
     * @brief Consruct a Reader
     *
     * @copydoc hide_ncid_param
     * @copydoc hide_g_param
     * @copydoc hide_dim_names_param
     */
    Reader( const int& ncid, const Topology& g, std::vector<std::string> dim_names) :
        m_ncid( &ncid), m_grid( g)
    {
        m_dims.resize( dim_names.size());
        if( g.ndim() == dim_names.size())
        {
            if( !dg::file::check_dimensions( ncid, &m_dims[0], m_grid, dim_names))
                throw std::runtime_error( "Requested dimensions do not exist");
        }
        else if( g.ndim()+1 == dim_names.size())
        {
            int tvarID;
            if( !dg::file::check_dimensions( ncid, &m_dims[0], &tvarID, m_grid, dim_names))
                throw std::runtime_error( "Requested dimensions do not exist");
        }
        else
            throw std::runtime_error( "Number of dimension names "+std::to_string(dim_names.size())+" must be either the same or one larger than grid dimensions "+std::to_string(g.ndim())+"\n");
        int num_vars = 0, num_dims;
        file::NC_Error_Handle err;
        err = nc_inq(ncid, &num_dims, &num_vars, NULL, NULL);
        // https://docs.unidata.ucar.edu/netcdf-c/current/reading_unknown.html
        for( int i=0; i<num_vars; i++)
        {
            char name[NC_MAX_NAME]; // 256
            int xtype;
            int ndims;
            err = nc_inq_varndims( ncid, i, &ndims);
            int dimIDs[ndims];
            err = nc_inq_var( ncid, i, name, &xtype, NULL, dimIDs, NULL);

            bool match = true;
            if( ndims != (int)m_dims.size() ||
                xtype != getNCDataType<typename Topology::value_type>() )
            {
                match = false;
            }
            else
            {
                for( unsigned i=0; i<m_dims.size(); i++)
                    if( m_dims[i] != dimIDs[i])
                        match = false;
            }
            if ( match)
                m_varids[name] = i;
        }
    }

    ///@copydoc Writer::grid()
    const typename Topology::host_grid grid() const{ return m_grid;}

    /**
     * @brief All variables names found that match the dimensions in the constructor
     *
     * @return A list of names existing in the file with matching dimensions
     */
    std::vector<std::string> names() const{
        std::vector<std::string> names;
        for( const auto& pair : m_varids)
            names.push_back( pair.first);
        return names;
    }

    /**
     * @brief The size of the unlimited dimension if it exists, 1 else
     *
     * @copydoc hide_comment_slice
     * @return The size of the unlimited dimension if it exists, 1 else
     */
    unsigned size( ) const{
        // https://github.com/Unidata/netcdf-c/issues/1898
        // The size of the unlimited dimension is the maximum of the sizes of all the variables
        // sharing that unlimited dimension
        // If reading a variable not written to that extension yet, filling values will be returned.
        size_t length;
        if( m_grid.ndim() == m_dims.size())
            return 1;
        file::NC_Error_Handle err;
        err = nc_inq_dimlen( *m_ncid, m_dims[0], &length);
        return unsigned(length);
    }

    /**
     * @brief Read all attributes of given variable
     *
     * @param name the name of the variable to define. If the name is not managed by this class we throw.
     * @return All attributes found for given variable
     */
    dg::file::JsonType get_attrs( std::string name) const
    {
        return dg::file::nc_attrs2json( *m_ncid, m_varids.at(name));
    }

    /**
     * @brief Read data from given variable
     *
     * @param name the name of the variable to read data from. If the name is not managed by this class we throw.
     * @copydoc hide_tparam_host_vector
     * @param data the data to read (must have grid size)
     * @param slice (ignored for time-independent variables). The number of the
     * time-slice to read (first element of the \c startp array in \c
     * nc_put_vara).
     * @copydoc hide_comment_slice
     */
    template<class host_vector>
    void get( std::string name, host_vector& data, unsigned slice=0) const
    {
        if( m_grid.ndim() == m_dims.size())
            dg::file::get_var( *m_ncid, m_varids.at(name), m_grid, data);
        else
            dg::file::get_vara( *m_ncid, m_varids.at(name), slice, m_grid, data);
    }
    /**
     * @brief Get attributes and data in one go
     *
     * Same as
     * @code
        auto atts = get( name);
        get( name, data, 0);
     * @endcode
     *
     * @param name the name of the variable to read data from. If the name is not managed by this class we throw.
     * @param atts [write-only] All attributes found for given variable
     * @copydoc hide_tparam_host_vector
     * @param data the data to read (must have grid size)
     */
    template<class host_vector>
    void get( std::string name, dg::file::JsonType& atts, host_vector& data) const
    {
        atts = get( name);
        get( name, data, 0);
    }
    private:
    const int* m_ncid;
    std::vector<int> m_dims;
    std::map<std::string,int> m_varids;
    typename Topology::host_grid m_grid;
};

///@}

}//namespace file
}//namespace dg
