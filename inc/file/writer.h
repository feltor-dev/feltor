#pragma once

#include "../dg/backend/typedefs.h"
#include "../dg/topology/fast_interpolation.h"
#include "easy_output.h"

namespace dg
{
namespace file
{


/**
 * @brief Create netcdf variable ids from records list
 *
 * For each record in \c record_list create a variable named \c record.name with
 * attribute \c record.long_name of dimension \c ndim in group \c ncid with
 * dimensions given by \c dim_ids
 * @tparam ndim Number of dimensions of variables
 * @tparam ListClass
 * @param ncid root or group id in a netcdf file
 * @param dim_ids ndim dimension ids associated to each variable
 * @param record_list list of records to put into ncid
 * @note This function compiles for MPI automatically if the macro
 * \c MPI_VERSION is defined. Then only the master rank 0 writes to file
 */
template<unsigned ndim, class ListClass>
std::map<std::string, int> create_varids( int ncid, int* dim_ids, const ListClass& record_list)
{
#ifdef MPI_VERSION
    int rank;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
    dg::file::NC_Error_Handle err;
    std::map<std::string, int> ids;
    for( auto& record : record_list)
    {
        std::string name = record.name;
        std::string long_name = record.long_name;
        ids[name] = 0;
        DG_RANK0 err = nc_def_var( ncid, name.data(), NC_DOUBLE, ndim,
            dim_ids, &ids.at(name));
        DG_RANK0 err = nc_put_att_text( ncid, ids.at(name), "long_name",
            long_name.size(), long_name.data());
    }
    return ids;
}

/**
 * @brief Write netcdf variable created from records list
 *
 * For each record in \c record_list create a variable named \c record.name with
 * attribute \c record.long_name of dimension \c ndim in group \c ncid with
 * dimensions given by \c dim_ids and data given by \c record.function( resultH, ps ...)
 * where \c resultH is a host vector given by \c grid_out
 * @tparam ndim Number of dimensions of variables
 * @tparam Geometry A topology with ndim dimensions
 * @tparam ListClass
 * @tparam Params
 * @param ncid root or group id in a netcdf file
 * @param dim_ids ndim dimension ids associated to each variable
 * @param grid gives the shape of the output variable (shape must be consistent with \c dim_ids)
 * @param record_list list of records to put into ncid
 * @param ps Parameters forwarded to \c record.function( resultH, ps...)
 */
template<unsigned ndim, class Geometry, class ListClass, class ...Params>
void write_static_records_list( int ncid, int* dim_ids, const Geometry& grid,
    const ListClass& record_list, Params&& ... ps)
{
    static_assert( ndim == Geometry::ndim());
    auto ids = create_varids<ndim,ListClass>( ncid, dim_ids, record_list);
    auto transferH = dg::evaluate(dg::zero, grid);
    for ( auto& record : record_list)
    {
        record.function( transferH, std::forward<Params>(ps)...);
        dg::file::put_var_double( ncid, ids.at(record.name), grid, transferH);
    }
}

/**
 * @brief A class to write time-dependent variables from a record list into a netcdf file
 * @tparam ndim Number of dimensions of variables
 */
template<unsigned ndim>
struct WriteRecordsList
{

    ///@brief Default constructor
    WriteRecordsList() = default;

    /**
     * @brief Create variables ids
     *
     * For each record in \c record_list create a variable named \c record.name with
     * attribute \c record.long_name of dimension \c ndim in group \c ncid with
     * dimensions given by \c dim_ids
     * @tparam ListClass
     * @param ncid root or group id in a netcdf file
     * @param dim_ids ndim dimension ids associated to each variable
     * @param record_list list of records to put into ncid
     */
    template<class ListClass>
    WriteRecordsList( int ncid, int* dim_ids, const ListClass& record_list) : m_start(0)
    {
        m_ids = create_varids<ndim>( ncid, dim_ids, record_list);
    }

    /**
     * @brief Write variables created from record list and projected to smaller grid
     *
     * For each record in \c record_list call \c record.function( resultD, ps...)
     * where \c resultD is a \c dg::x::DVec of size given by \c grid and write
     * its projection to \c grid_out into \c ncid
     * @tparam Geometry A topology with ndim-1 dimensions
     * @tparam ListClass
     * @tparam Params
     * @param ncid root or group id in a netcdf file
     * @param grid gives the shape of the result of \c record.function
     * @param grid_out gives the shape of the output variable (shape must be
     * consistent with \c dim_ids in constructor)
     * @param record_list list of records to put into ncid
     * @param ps Parameters forwarded to \c record.function( resultD, ps...)
     */
    template<class Geometry, class ListClass, class ... Params >
    void project_write( int ncid, const Geometry& grid, const Geometry& grid_out,
        const ListClass& record_list, Params&& ...ps)
    {
        static_assert( ndim>1);
        static_assert( ndim == Geometry::ndim() +1);
        dg::x::DVec resultD = dg::evaluate( dg::zero, grid);
        dg::x::DVec transferD( dg::evaluate(dg::zero, grid_out));
        dg::x::HVec transferH( dg::evaluate(dg::zero, grid_out));
        dg::MultiMatrix<dg::x::DMatrix,dg::x::DVec> projectD =
            dg::create::fast_projection( grid, grid.n()/grid_out.n(),
                grid.Nx()/grid_out.Nx(), grid.Ny()/grid_out.Ny());
        for( auto& record : record_list)
        {
            record.function( resultD, std::forward<Params>(ps)...);
            dg::blas2::symv( projectD, resultD, transferD);
            dg::assign( transferD, transferH);
            dg::file::put_vara_double( ncid, m_ids.at(record.name), m_start, grid_out, transferH);
            m_start++;
        }
    }
    /**
     * @brief Write variables created from record list
     *
     * For each record in \c record_list call \c record.function( resultD, ps...)
     * where \c resultD is a \c dg::x::DVec of size given by \c grid and write
     * into \c ncid
     * @tparam Geometry A topology with ndim dimensions
     * @tparam ListClass
     * @tparam Params
     * @param ncid root or group id in a netcdf file
     * @param grid gives the shape of the output variable (shape must be
     * consistent with \c dim_ids in constructor)
     * @param record_list list of records to put into ncid
     * @param ps Parameters forwarded to \c record.function( resultD, ps...)
     */
    template<class Geometry, class ListClass, class ... Params >
    void write( int ncid, const Geometry& grid, const
        ListClass& record_list, Params&& ...ps)
    {
        dg::x::DVec resultD = dg::evaluate( dg::zero, grid);
        dg::x::HVec resultH = dg::evaluate( dg::zero, grid);
        for( auto& record : record_list)
        {
            record.function( resultD, std::forward<Params>(ps)...);
            dg::assign( resultD, resultH);
            dg::file::put_vara_double( ncid, m_ids.at(record.name), m_start, grid, resultH);
            m_start++;
        }
    }
    private:
    size_t m_start;
    std::map<std::string, int> m_ids;
};

/// @brief A specialisation for 0-dimensional time-dependent data
template<>
struct WriteRecordsList<1>
{
    WriteRecordsList() = default;
    template<class ListClass>
    WriteRecordsList( int ncid, int* dim_ids, const ListClass& record_list) : m_start(0)
    {
        m_ids = create_varids<1>( ncid, dim_ids, record_list);
    }
    template<class ListClass, class ...Params>
    void write( int ncid, const ListClass& record_list, Params&& ... ps)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
#endif //MPI_VERSION
        size_t count = 1;
        for( auto& record : record_list)
        {
            double result = record.function( std::forward<Params>(ps)...);
            DG_RANK0 nc_put_vara_double( ncid, m_ids.at(record.name), &m_start, &count, &result);
            m_start++;
        }
    }
    private:
    size_t m_start;
    std::map<std::string, int> m_ids;
};

}//namespace file
}//namespace dg
