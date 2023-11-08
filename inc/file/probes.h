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
 * @ingroup netcdf
 */
struct Probes
{
    Probes() = default;
    // if coords[i] are empty then all functions simply return immediately only master threads coords count
    template<class Geometry, class ListClass>
    Probes(
        int ncid,
        unsigned itstp,
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
        if(rank==0) m_simple_probes = dg::MHVec(params.coords[0],
                grid.communicator());
#else //MPI_VERSION
        m_simple_probes = dg::HVec(m_num_pins);
#endif
        for( auto& record : probe_list)
            m_simple_probes_intern[record.name] = std::vector<dg::x::HVec>(itstp, m_simple_probes);
        m_time_intern.resize(itstp);
        m_resultD = dg::evaluate( dg::zero, grid);
        m_resultH = dg::evaluate( dg::zero, grid);
        define_nc_variables( ncid, params.format, params.coords, params.coords_names,
                probe_list);
    }

    // record.name, record.long_name, record.function( resultH, ps...)
    template<class ListClass, class ...Params>
    void static_write( const ListClass& diag_static_list, Params&& ... ps)
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

    // record.name, record.long_name, record.function( resultH, ps...)
    template<class ListClass, class ...Params>
    void write( double time, const ListClass& probe_list, Params&& ... ps)
    {
        if(!m_probes) return;
        dg::file::NC_Error_Handle err;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        /// Probes FIRST output ///
        size_t probe_count[] = {1, m_num_pins};
        DG_RANK0 err = nc_put_vara_double( m_probe_grp_id, m_probe_timevarID,
                &m_probe_start[0], &probe_count[0], &time);
        for( auto& record : probe_list)
        {
            record.function( m_resultD, std::forward<Params>(ps)...);
            dg::assign( m_resultD, m_resultH);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
            DG_RANK0 err = nc_put_vara_double( m_probe_grp_id,
                    m_probe_id_field.at(record.name), m_probe_start, probe_count,
#ifdef MPI_VERSION
                    m_simple_probes.data().data()
#else
                    m_simple_probes.data()
#endif
            );
        }
         /// End probes output ///

    }
    // is thought to be called itstp times before write
    template<class ListClass, class ...Params>
    void save( double time, unsigned iter, const ListClass& probe_list, Params&& ... ps)
    {
        if(!m_probes) return;
        for( auto& record : probe_list)
        {
            record.function( m_resultD, std::forward<Params>(ps)...);
            dg::assign( m_resultD, m_resultH);
            dg::blas2::symv( m_probe_interpolate, m_resultH, m_simple_probes);
            m_simple_probes_intern.at(record.name)[iter]=m_simple_probes;
            m_time_intern[iter]=time;
        }
    }
    void write_after_save()
    {
        if( !m_probes) return;
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        size_t probe_count[] = {1, m_num_pins};
        //OUTPUT OF PROBES
        dg::file::NC_Error_Handle err;
        for( unsigned j=0; j<m_time_intern.size(); j++)
        {
            m_probe_start[0] += 1;
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
        }
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
    size_t m_probe_start[2] = {0,0};

    template<class ListClass>
    void define_nc_variables( int ncid, const std::string& format,
        const std::vector<dg::HVec>& coords,
        const std::vector<std::string> & coords_names,
        const ListClass& probe_list)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        dg::file::NC_Error_Handle err;
        DG_RANK0 err = nc_def_grp(ncid,"probes",&m_probe_grp_id);
        DG_RANK0 err = nc_put_att_text( m_probe_grp_id, NC_GLOBAL,
            "format", format.size(), format.data());
        dg::Grid1d g1d( 0,1,1,m_num_pins);
        DG_RANK0 err = dg::file::define_dimensions( m_probe_grp_id,
                m_probe_dim_ids, &m_probe_timevarID, g1d, {"time", "x"});
        std::vector<int> pin_id;
        for( unsigned i=0; i<coords.size(); i++)
        {
            int pin_id;
            DG_RANK0 err = nc_def_var(m_probe_grp_id, coords_names[i].data(),
                NC_DOUBLE, 1, &m_probe_dim_ids[1], &pin_id);
            DG_RANK0 err = nc_put_var_double( m_probe_grp_id, pin_id, coords[i].data());
        }
        for( auto& record : probe_list)
        {
            std::string name = record.name;
            std::string long_name = record.long_name;
            m_probe_id_field[name] = 0;//creates a new id4d entry for all processes
            DG_RANK0 err = nc_def_var( m_probe_grp_id, name.data(),
                    NC_DOUBLE, 2, m_probe_dim_ids,
                    &m_probe_id_field.at(name));
            DG_RANK0 err = nc_put_att_text( m_probe_grp_id,
                    m_probe_id_field.at(name), "long_name", long_name.size(),
                    long_name.data());
        }
    }

};

} //namespace file
}//namespace dg
