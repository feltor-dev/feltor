#pragma once

#include "nc_utilities.h"
#include "probes_params.h"

#ifdef MPI_VERSION
#include "dg/topology/mpi_projection.h"
#endif //MPI_VERSION

namespace dg
{
namespace file
{

///@cond
namespace detail
{
// helpers to define a C++-17 if constexpr
template<class Geometry, unsigned ndim>
struct CreateInterpolation{};


template<class Geometry>
struct CreateInterpolation<Geometry,1>
{
    auto call( const std::vector<dg::HVec>& x, const Geometry& g)
    {
        return dg::create::interpolation( x[0], g, g.bcx());

    }
};
template<class Geometry>
struct CreateInterpolation<Geometry,2>
{
auto call( const std::vector<dg::HVec>& x, const Geometry& g)
{
    return dg::create::interpolation( x[0], x[1], g, g.bcx(), g.bcy());

}
};
template<class Geometry>
struct CreateInterpolation<Geometry,3>
{
auto call( const std::vector<dg::HVec>& x, const Geometry& g)
{
    return dg::create::interpolation( x[0], x[1], x[2], g, g.bcx(), g.bcy(), g.bcz());

}
};
}
///@endcond
//

/**
 * @brief Facilitate output at selected points
 *
 * This class is a high level synthetic diagnostics package
 * Typically, it works together with the \c dg::file::parse_probes function
 *
 * Instead of writing to file every time one desires probe outputs, an internal
 * buffer stores the probe values when the \c buffer member is called. File
 * writes happen only when calling \c flush
 * @note in an MPI program all processes have to create the class and call its methods. The
 * class automatically takes care of which threads write to file.
 * @ingroup netcdf
 */
struct Probes
{
    Probes() = default;
    // if coords[i] are empty then all functions simply return immediately only master threads coords count
    /**
     * @brief Construct from parameter struct
     *
     * @param ncid netcdf id; a "probes" group will be generated that contains all fields this class writes to file
     * (probe dimensions are called "time" and "dim"). The file must be open.
     * @param grid The interpolation matrix is generated with the \c grid and \c paraams.coords . \c grid.ndim
     * must equal \c param.coords.size()
     * @param params Typically read in from file with \c dg::file::parse_probes
     * @param probe_list The list of variables later used in \c write
     *
     */
    template<class Geometry, class ListClass>
    Probes(
        int ncid,
        const Geometry& grid,
        const ProbesParams& params, // do nothing if probes is false
        const ListClass& probe_list
        )
    {
        m_probes = params.probes;
        if( !params.probes) return;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION

        if ( params.coords.size() != grid.ndim())
            throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in coords!");
        m_num_pins = params.get_coords_sizes();

        m_probe_interpolate = detail::CreateInterpolation<Geometry,
                            Geometry::ndim()>().call( params.coords, grid);
        // Create helper storage probe variables
#ifdef MPI_VERSION
        m_simple_probes = dg::MHVec(params.coords[0],
                grid.communicator());
#else //MPI_VERSION
        m_simple_probes = dg::HVec(m_num_pins);
#endif
        for( auto& record : probe_list)
            m_simple_probes_intern[record.name] = std::vector<dg::x::HVec>(); // empty vectors
        m_resultD = dg::evaluate( dg::zero, grid);
        m_resultH = dg::evaluate( dg::zero, grid);
        define_nc_variables( ncid, params.format, params.coords, params.coords_names,
                probe_list);
    }

    /*! @brief Directly write results of a list of callback functions to file
     *
     * Each item in the list consists of a name, a long name and a callback function that
     * is called with a host vector as first argument and the given list of Params as additional arguments.
     * @code
        for ( auto& record : diag_static_list)
        {
            record.name;
            record.long_name;
            record.function( resultH, ps...);
        }
     * @endcode
     * The host vector has the size of the grid given in the constructor of the Probes class.
     * The callback function is supposed to write its result into the given host vector.
     *
     * The result is then interpolated to the probe positions and stored in the
     * netcdf file in the probes group under the given variable name, with the
     * long name as attribute ("long_name") and the "dim" probes dimension.
     * @note The netcdf file must be open when this method is called.
     * @note If \c param.probes was \c false in the constructor this function returns immediately
     */
    template<class HostList, class ...Params>
    void static_write( const HostList& diag_static_list, Params&& ... ps)
    {
        if(!m_probes) return;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        dg::file::NC_Error_Handle err;
        for ( auto& record : diag_static_list)
        {
            int vecID;
            DG_RANK0 err = nc_def_var( m_probe_grp_id, record.name.data(), NC_DOUBLE, 1,
                &m_probe_dim_ids[1], &vecID);
            DG_RANK0 err = nc_put_att_text( m_probe_grp_id, vecID,
                "long_name", record.long_name.size(), record.long_name.data());
            record.function( m_resultH, std::forward<Params>(ps)...);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
            DG_RANK0 nc_put_var_double( m_probe_grp_id, vecID,
#ifdef MPI_VERSION
                m_simple_probes.data().data()
#else
                m_simple_probes.data()
#endif
                    );
        }
    }

    // netcdf file must be open for this
    /*! @brief Write (time-dependent) results of a list of callback functions to internal buffer
     *
     * The \c probe_list must be the same as the one used in the constructor, where
     * the corresponding variables (with one unlimited time-dimension) are created
     *
     * @param time The time value to store
     * @param probe_list the list of records that was given in the constructor
     * @param ps The parameters forwarded to the \c record.function( resultD,
     * ps...) The function is supposed to store its result into the given
     * device vector
     * @note No data is written to file and the netcdf file does not need to be open.
     * @note If \c param.probes was \c false in the constructor this function returns immediately
     */
    template<class DeviceList, class ...Params>
    void buffer( double time, const DeviceList& probe_list, Params&& ... ps)
    {
        if(!m_probes) return;
        m_time_intern.push_back(time);
        for( auto& record : probe_list)
        {
            record.function( m_resultD, std::forward<Params>(ps)...);
            dg::assign( m_resultD, m_resultH);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
            m_simple_probes_intern.at(record.name).push_back(m_simple_probes);
        }
    }

    /*! @brief Flush the buffer to file
     *
     * Write all contents of the buffer to the netcdf file and reset the buffer.
     *
     * @note If \c param.probes was \c false in the constructor this function returns immediately
     * @attention the netcdf file needs to be open when calling this function
     */
    void flush()
    {
        if(!m_probes) return;
        // else we write the internal buffer to file
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        size_t probe_count[] = {1, m_num_pins};
        dg::file::NC_Error_Handle err;

        for( unsigned j=0; j<m_time_intern.size(); j++)
        {
            DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                    m_probe_timevarID, &m_probe_start[0] , &probe_count[0],
                    &m_time_intern[j]);
            for( auto& field : m_simple_probes_intern)
            {
                DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                        m_probe_id_field.at(field.first), m_probe_start,
                        probe_count,
#ifdef MPI_VERSION
                        field.second[j].data().data()
#else
                        field.second[j].data()
#endif
                        );
            }
            m_probe_start[0] ++;
        }
        // flush internal buffer
        m_time_intern.clear();
        for( auto& field : m_simple_probes_intern)
            field.second.clear();
    }

    /*! @brief Same as \c buffer followed by \c flush
     *
     * @code
        buffer( time, probe_list, std::forward<Params>(ps)...);
        flush();
     * @endcode
     */
    template<class DeviceList, class ...Params>
    void write( double time, const DeviceList& probe_list, Params&& ... ps)
    {
        buffer( time, probe_list, std::forward<Params>(ps)...);
        flush();
    }

    private:
    bool m_probes = false;
    int m_probe_grp_id;
    int m_probe_dim_ids[2];
    int m_probe_timevarID;
    std::map<std::string, int> m_probe_id_field;
    unsigned m_num_pins;
    dg::x::IHMatrix m_probe_interpolate;
    dg::x::HVec m_simple_probes;
    std::map<std::string, std::vector<dg::x::HVec>> m_simple_probes_intern;
    std::vector<double> m_time_intern;
    dg::x::DVec m_resultD;
    dg::x::HVec m_resultH;
    size_t m_probe_start[2] = {0,0}; // always point to where we currently can write
    unsigned m_iter = 0; // the number of calls to write

    template<class ListClass>
    void define_nc_variables( int ncid, const std::string& format,
        const std::vector<dg::HVec>& coords,
        const std::vector<std::string> & coords_names,
        const ListClass& probe_list)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        if( rank == 0)
        {
#endif //MPI_VERSION

        dg::file::NC_Error_Handle err;
        err = nc_def_grp(ncid,"probes",&m_probe_grp_id);
        err = nc_put_att_text( m_probe_grp_id, NC_GLOBAL,
            "format", format.size(), format.data());
        dg::Grid1d g1d( 0,1,1,m_num_pins);
        err = dg::file::define_dimensions( m_probe_grp_id,
                m_probe_dim_ids, &m_probe_timevarID, g1d, {"time", "dim"});
        std::vector<int> pin_id;
        for( unsigned i=0; i<coords.size(); i++)
        {
            int pin_id;
            err = nc_def_var(m_probe_grp_id, coords_names[i].data(),
                NC_DOUBLE, 1, &m_probe_dim_ids[1], &pin_id);
            err = nc_put_var_double( m_probe_grp_id, pin_id, coords[i].data());
        }
        for( auto& record : probe_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            m_probe_id_field[name] = 0;//creates a new id4d entry for all processes
            err = nc_def_var( m_probe_grp_id, name.data(),
                    NC_DOUBLE, 2, m_probe_dim_ids,
                    &m_probe_id_field.at(name));
            err = nc_put_att_text( m_probe_grp_id,
                    m_probe_id_field.at(name), "long_name", long_name.size(),
                    long_name.data());
        }
#ifdef MPI_VERSION
        }
#endif // MPI_VERSION
    }

};

} //namespace file
}//namespace dg
