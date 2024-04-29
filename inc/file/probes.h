#pragma once


#include "dg/topology/interpolation.h"
#ifdef MPI_VERSION
#include "dg/topology/mpi_projection.h"
#endif //MPI_VERSION

#include "probes_params.h"
#include "writer.h"

namespace dg
{
namespace file
{

///@cond
namespace detail
{
// helpers to define a C++-17 if constexpr
template<class Topology, unsigned ndim>
struct CreateInterpolation{};


template<class Topology>
struct CreateInterpolation<Topology,1>
{
    static auto call( const std::vector<dg::HVec>& x, const Topology& g)
    {
        return dg::create::interpolation( x[0], g, g.bcx());

    }
};
template<class Topology>
struct CreateInterpolation<Topology,2>
{
static auto call( const std::vector<dg::HVec>& x, const Topology& g)
{
    return dg::create::interpolation( x[0], x[1], g, g.bcx(), g.bcy());

}
};
template<class Topology>
struct CreateInterpolation<Topology,3>
{
static auto call( const std::vector<dg::HVec>& x, const Topology& g)
{
    return dg::create::interpolation( x[0], x[1], x[2], g, g.bcx(), g.bcy(), g.bcz());

}
};

template<class Topology, class Enable = void>
class Helper {};

template<class Topology>
struct Helper<Topology, std::enable_if_t<dg::is_shared_grid<Topology>::value >>
{
    using IHMatrix = dg::IHMatrix_t<typename Topology::value_type>;
    using Writer0d = dg::file::Writer<dg::RealGrid0d<typename Topology::value_type>>;
    using Writer1d = dg::file::Writer<dg::RealGrid1d<typename Topology::value_type>>;
    static dg::RealGrid1d<typename Topology::value_type> create( unsigned num_pins)
    {
        return {0,1,1,num_pins};
    }
    static typename Topology::host_vector probes_vec( const dg::HVec& coord, const Topology& grid)
    {
        return typename Topology::host_vector(coord);
    }
    static void def_group( int ncid, std::string format, int& grpid)
    {
        dg::file::NC_Error_Handle err;
        err = nc_def_grp(ncid,"probes",&grpid);
        err = nc_put_att_text( grpid, NC_GLOBAL,
            "format", format.size(), format.data());
    }
    static void open_group( int ncid, int& grpid)
    {
        dg::file::NC_Error_Handle err;
        err = nc_inq_grp_ncid( ncid, "probes", &grpid);
    }
};

#ifdef MPI_VERSION
template<class Topology>
struct Helper<Topology, std::enable_if_t<dg::is_mpi_grid<Topology>::value >>
{
    using IHMatrix = dg::MIHMatrix_t<typename Topology::value_type>;
    using Writer0d = dg::file::Writer<dg::RealMPIGrid0d<typename Topology::value_type>>;
    using Writer1d = dg::file::Writer<dg::RealMPIGrid1d<typename Topology::value_type>>;
    static dg::RealMPIGrid1d<typename Topology::value_type> create( unsigned num_pins)
    {
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank); // a private variable
        MPI_Comm comm1d, comm_cart1d;
        MPI_Comm_split( MPI_COMM_WORLD, rank, rank, &comm1d);
        int dims[1] = {1};
        int periods[1] = {true};
        MPI_Cart_create( comm1d, 1, dims, periods, false, &comm_cart1d);
        dg::MPIGrid1d g1d ( 0,1,1, rank == 0 ? num_pins : 1, comm_cart1d);
        return g1d;
    }
    static typename Topology::host_vector probes_vec( const dg::HVec& coord, const Topology& grid)
    {
        typename Topology::host_vector::container_type vec(coord);
        return typename Topology::host_vector(vec, grid.communicator());
    }
    static void def_group( int ncid, std::string format, int& grpid)
    {
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        DG_RANK0 Helper<dg::RealGrid1d<typename Topology::value_type>>::def_group(ncid, format, grpid);
    }
    static void open_group( int ncid, int& grpid)
    {
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        DG_RANK0 Helper<dg::Grid1d>::open_group(ncid, grpid);
    }
};
#endif
}
///@endcond
//

/**
 * @brief Facilitate output at selected points
 *
 * This class is a high level synthetic diagnostics package.
 * Typically, it works together with the \c dg::file::parse_probes function
 *
 * Instead of writing to file every time one desires probe outputs, an internal
 * buffer stores the probe values when the \c buffer member is called. File
 * writes happen only when calling \c flush
 * @note in an MPI program all processes have to create the class and call its methods.
 * Only the master thread writes to file and needs to open the file
 * @copydoc hide_tparam_topology
 * @note It is the topology of the simulation grid that is needed here, i.e.
 * the Topology **from which** to interpolate, not the topology of the 1d probe
 * array. The class automatically constructs the latter itself.
 * @attention Because the paraview NetCDF reader is faulty, it is recommended that
 * \c Probes is constructed only after all other root dimensions in the file
 * are defined. This is because of the dimension numbering in NetCDF-4.
 * @ingroup Cpp
 */
template<class Topology>
struct Probes
{
    Probes() = default;
    /**
     * @brief Construct from parameter struct
     *
     * @param ncid NetCDF id; a "probes" group will be generated that contains all fields this class writes to file
     * (probe dimensions are called "ptime" and "pdim"). The file must be open.
     * @param grid The interpolation matrix is generated with the \c grid and \c params.coords . \c grid.ndim
     * must equal \c param.coords.size()
     * @param params Typically read in from file with \c dg::file::parse_probes
     */
    Probes(
        const int& ncid,
        const Topology& grid,
        const ProbesParams& params // do nothing if probes is false
        ) : m_ncid(&ncid)
    {
        m_probes = params.probes;
        if( !params.probes) return;

        if ( params.coords.size() != grid.ndim())
            throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in coords!");
        unsigned num_pins = params.get_coords_sizes();

        m_probe_interpolate = detail::CreateInterpolation<Topology,
                            Topology::ndim()>::call( params.coords, grid);
        // Create helper storage probe variables
        m_simple_probes = detail::Helper<Topology>::probes_vec( params.coords[0], grid);
        m_resultH = dg::evaluate( dg::zero, grid);

        detail::Helper<Topology>::def_group( ncid, params.format, m_probe_grp_id);

        auto g1d = detail::Helper<Topology>::create( num_pins);
        typename detail::Helper<Topology>::Writer1d
            writer_coords( m_probe_grp_id, g1d, {"pdim"});
        for( unsigned i=0; i<params.coords.size(); i++)
            writer_coords.def_and_put( params.coords_names[i], {},
                detail::Helper<Topology>::probes_vec( params.coords[i], grid));
        m_writer0d = { m_probe_grp_id, {}, {"ptime"}};
        m_writer1d = { m_probe_grp_id, g1d, {"ptime", "pdim"}};
    }

    /*! @brief Directly write results of a list of callback functions to file
     *
     * Each item in the list consists of a name, a long name and a callback function that
     * is called with a host vector as first argument and the given list of Params as additional arguments.
     * @code
        for ( auto& record : records)
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
     * @copydoc hide_tparam_listclass
     */
    template<class ListClass, class ...Params>
    void static_write( const ListClass& records, Params&& ... ps)
    {
        if(!m_probes) return;

        typename detail::Helper<Topology>::Writer1d
            write( m_probe_grp_id, m_writer1d.grid(), {"pdim"});
        auto result =
            dg::construct<get_first_argument_type_t<typename ListClass::value_type::Signature>>(
                m_resultH);
        for ( auto& record : records)
        {
            record.function( result, std::forward<Params>(ps)...);
            dg::assign( result, m_resultH);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
            write.def_and_put( record.name, dg::file::long_name(
                record.long_name), m_simple_probes);
        }
    }

    /*! @brief Write (time-dependent) results of a list of callback functions to internal buffer
     *
     * The \c probe_list must be the same as the one used in the constructor, where
     * the corresponding variables (with one unlimited time-dimension) are created
     *
     * @param time The time value to store
     * @param probe_list the list of records to store (variables are defined in file on first write)
     * @param ps The parameters forwarded to the \c record.function( resultD,
     * ps...) The function is supposed to store its result into the given
     * device vector
     * @note No data is written to file and the netcdf file does not need to be open.
     * @note If \c param.probes was \c false in the constructor this function returns immediately
     */
    template<class ListClass, class ...Params>
    void buffer( double time, const ListClass& probe_list, Params&& ... ps)
    {
        if(!m_probes) return;
        m_time_intern.push_back(time);
        auto result =
            dg::construct<get_first_argument_type_t<typename ListClass::value_type::Signature>>(
                m_resultH);
        if( m_simple_probes_intern.empty())
            init_buffer( probe_list);
        if( m_probe_start == 0)
            init_writer( probe_list);
        for( auto& record : probe_list)
        {
            record.function( result, std::forward<Params>(ps)...);
            dg::assign( result, m_resultH);
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
        detail::Helper<Topology>::open_group( *m_ncid, m_probe_grp_id);

        // else we write the internal buffer to file
        for( unsigned j=0; j<m_time_intern.size(); j++)
        {
            m_writer0d.put( "ptime", m_time_intern[j], m_probe_start);
            for( auto& field : m_simple_probes_intern)
                m_writer1d.put( field.first, field.second[j], m_probe_start);
            m_probe_start ++;
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
    template<class ListClass, class ...Params>
    void write( double time, const ListClass& probe_list, Params&& ... ps)
    {
        buffer( time, probe_list, std::forward<Params>(ps)...);
        flush();
    }

    private:
    template<class ListClass>
    void init_buffer( const ListClass& probe_list)
    {
        for( auto& record : probe_list)
            m_simple_probes_intern[record.name] = std::vector<typename Topology::host_vector>(); // empty vectors
    }
    template<class ListClass>
    void init_writer( const ListClass& probe_list)
    {
        for( auto& record : probe_list)
            m_writer1d.def( record.name, dg::file::long_name( record.long_name));
    }

    typename detail::Helper<Topology>::Writer1d m_writer1d;
    typename detail::Helper<Topology>::Writer0d m_writer0d;
    bool m_probes = false;
    const int* m_ncid;
    int m_probe_grp_id;
    typename detail::Helper<Topology>::IHMatrix m_probe_interpolate;
    typename Topology::host_vector m_simple_probes;
    std::map<std::string, std::vector<typename Topology::host_vector>> m_simple_probes_intern;
    std::vector<double> m_time_intern;
    //Container m_resultD;
    typename Topology::host_vector m_resultH;
    size_t m_probe_start = 0; // always point to where we currently can write
};

} //namespace file
}//namespace dg
