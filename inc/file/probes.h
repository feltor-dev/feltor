#pragma once


#include "dg/topology/interpolation.h"
#ifdef MPI_VERSION
#include "dg/topology/mpi_projection.h"
#include "nc_mpi_file.h"
#endif //MPI_VERSION

#include "probes_params.h"
#include "nc_file.h"
#include "records.h"

namespace dg
{
namespace file
{
/**
 * @class hide_tparam_listclass
 * @tparam ListClass
 * A Type whose <tt> ListClass::value_type </tt> equals a %Record class (e.g. \c dg::file::Record)
 * The Signature <tt> ListClass::value_type::Signature </tt> must have either \c void as return type
 * or a primitive type. The latter indicates a scalar output and must coincide
 * with <tt> Topology::ndim() == 0</tt>. If the return type is void then the
 * **first argument type** must be a Vector type constructible from \c
 * Topology::host_vector e.g. a \c dg::DVec.
 */

/**
 * @brief Facilitate output at selected points
 *
 * This class is a high level synthetic diagnostics package.
 * Typically, it works together with the \c dg::file::parse_probes function
 *
 * Instead of writing to file every time one desires probe outputs, an internal
 * buffer stores the probe values when the \c buffer member is called. File
 * writes happen only when calling \c flush
 * @note in an MPI program all processes in the \c file.communicator() have to
 * create the class and call its methods.
 * @note It is the topology of the simulation grid that is needed here, i.e.
 * the Topology **from which** to interpolate, not the topology of the 1d probe
 * array. The class automatically constructs the latter itself.
 * @attention Because the paraview NetCDF reader is faulty, it is recommended that
 * \c Probes is constructed only after all other root dimensions in the file
 * are defined. This is because of the dimension numbering in NetCDF-4.
 * @ingroup probes
 */
template<class NcFile, class Topology>
struct Probes
{
    Probes() = default;
    /**
     * @brief Construct from parameter struct
     *
     * @param file NetCDF file; a "probes" group will be generated in the
     * current group that contains all fields this class writes to file (probe
     * dimensions are called "ptime" and "pdim"). The file must be open.
     * @param grid The interpolation matrix is generated with the \c grid and \c params.coords . \c grid.ndim
     * must equal \c param.coords.size()
     * @param params Typically read in from file with \c dg::file::parse_probes
     */
    Probes(
        NcFile& file,
        const Topology& grid,
        const ProbesParams& params // do nothing if probes is false
        ) : m_file(&file)
    {
        m_probes = params.probes;
        if( !params.probes) return;

        if ( params.coords.size() != grid.ndim())
            throw std::runtime_error( "Need "+std::to_string(grid.ndim())+" values in coords!");
        unsigned num_pins = params.get_coords_sizes();
        // params.coords is empty on ranks other than master

// TODO We could think about distributing the coords among ranks using grid.contains ...
        m_probe_interpolate = dg::create::interpolation( params.coords, grid,
            grid.get_bc(), "dg");
        // Create helper storage probe variables
        m_simple_probes = create_probes_vec( params.coords[0], grid);
        m_resultH = dg::evaluate( dg::zero, grid);

        file.def_grp( "probes");
        m_grp = file.get_current_path() / "probes";
        file.set_grp( m_grp);
        file.put_att( {"format", params.format});
        file.def_dim( "pdim", num_pins);
        file.template def_dimvar_as<double>( "ptime", NC_UNLIMITED, {{"axis" , "T"}});
        for( unsigned i=0; i<params.coords.size(); i++)
        {
            auto probes_vec = create_probes_vec( params.coords[i], grid);
            file.defput_var( params.coords_names[i], {"pdim"},
                {{"long_name" , "Coordinate variable for probe position"}},
                {probes_vec}, probes_vec);
        }
        file.set_grp("..");
    }

    /*! @brief Directly write results of a list of callback functions to file
     *
     * Each item in the list consists of a name, attributes and a callback function that
     * is called with \c result as first argument and the given list of Params as additional arguments.
     * @code
        for ( auto& record : records)
        {
            record.name;
            record.atts;
            record.function( result, ps...);
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
        using Result = get_first_argument_type_t<typename ListClass::value_type::Signature>;
        auto result = dg::construct<Result>( m_resultH);
        static_assert( dg::is_vector_v<Result>, "Result must be a vector type");
        m_file->set_grp( m_grp);

        for ( auto& record : records)
        {
            record.function( result, std::forward<Params>(ps)...);
            dg::assign( result, m_resultH);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);

            m_file->defput_var( record.name, {"pdim"}, record.atts,
                {m_simple_probes}, m_simple_probes);
        }
        m_file->set_grp("..");
    }

    /*! @brief Write (time-dependent) results of a list of callback functions to internal buffer
     *
     * @param time The time value to store
     * @param probe_list the list of records to store (variables are defined in file on first write)
     * @param ps The parameters forwarded to the \c record.function( resultD,
     * ps...) The function is supposed to store its result into the given
     * device vector
     * @note No data is written to file and the netcdf file does not need to be open.
     * @note If \c param.probes was \c false in the constructor this function returns immediately
     * @copydoc hide_tparam_listclass
     */
    template<class ListClass, class ...Params>
    void buffer( double time, const ListClass& probe_list, Params&& ... ps)
    {
        if(!m_probes) return;
        using Result = get_first_argument_type_t<typename ListClass::value_type::Signature>;
        auto result = dg::construct<Result>( m_resultH);
        m_time_intern.push_back(time);
        if( m_simple_probes_intern.empty())
            init<dg::get_value_type<Result>>( probe_list);

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
        // else we write the internal buffer to file
        m_file->set_grp( m_grp);
        for( unsigned j=0; j<m_time_intern.size(); j++)
        {
            m_file->put_var( "ptime", {m_probe_start}, m_time_intern[j]);
            for( auto& field : m_simple_probes_intern)
                m_file->put_var( field.first, {m_probe_start, field.second[j]},
                    field.second[j]);
            m_probe_start ++;
        }
        // clear internal buffers
        m_time_intern.clear();
        for( auto& field : m_simple_probes_intern)
            field.second.clear();
        m_file->set_grp( "..");
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
    template<class T=Topology, std::enable_if_t<dg::is_vector_v<typename
    T::host_vector, dg::SharedVectorTag>, bool> = true>
    auto create_probes_vec( const std::vector<double>& coord, const T&)
    {
        return typename Topology::host_vector(coord);
    }
    template<class T=Topology, std::enable_if_t<dg::is_vector_v<typename
    T::host_vector, dg::MPIVectorTag>, bool> = true>
    auto create_probes_vec( const std::vector<double>& coord, const T& grid)
    {
        return typename Topology::host_vector(coord, grid.communicator());
    }
    template<class value_type, class ListClass>
    void init( const ListClass& probe_list)
    {
        m_file->set_grp( m_grp);
        for( auto& record : probe_list)
        {
            m_simple_probes_intern[record.name] = {}; // empty vectors
            m_file->template def_var_as<value_type>( record.name, {"ptime", "pdim"});
            m_file->put_atts( record.atts, record.name);
        }
        m_file->set_grp( "..");
    }
    bool m_probes = false;
    NcFile* m_file;
    std::filesystem::path m_grp;
    int m_probe_grp_id;
    typename Topology::host_vector m_simple_probes;
#ifdef MPI_VERSION
    std::conditional_t<
        dg::is_vector_v<typename Topology::host_vector, dg::SharedVectorTag>,
        dg::IHMatrix_t< typename Topology::value_type>,
        dg::MIHMatrix_t<typename Topology::value_type> > m_probe_interpolate;
#else
    dg::IHMatrix_t< typename Topology::value_type> m_probe_interpolate;
#endif
    std::map<std::string, std::vector<typename Topology::host_vector>> m_simple_probes_intern;
    std::vector<double> m_time_intern;
    //Container m_resultD;
    typename Topology::host_vector m_resultH;
    size_t m_probe_start = 0; // always point to where we currently can write
};

} //namespace file
}//namespace dg
