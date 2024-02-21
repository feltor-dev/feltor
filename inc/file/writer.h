#pragma once

#include <functional>
#include <map>

#include "../dg/topology/fast_interpolation.h"

#include "easy_dims.h"
#include "easy_output.h"
#include "json_netcdf.h"


namespace dg
{
namespace file
{

///@addtogroup Cpp
///@{

template<class Topology>
struct Writer
{
    Writer( const int& ncid, const Topology& g, std::vector<std::string> dim_names) :
        m_ncid( &ncid), m_grid( g)
    {
        m_dims.resize( dim_names.size());
        if( g.ndim() == dim_names.size())
        {
            dg::file::define_dimensions( ncid, &m_dims[0], m_grid, dim_names, true);
        }
        else if( g.ndim()+1 == dim_names.size())
        {
            int tvarID = 0;
            dg::file::define_dimensions( ncid, &m_dims[0], &tvarID, m_grid, dim_names, true);
            if( g.ndim() == 0) // enable time writing for 0d Writer
            {
                m_varids[dim_names[0]] = tvarID;
            }
        }
        else
            throw std::runtime_error( "Number of dimension names "+std::to_string(dim_names.size())+" must be either the same or one larger than grid dimensions "+std::to_string(g.ndim())+"\n");
    }
    const typename Topology::host_grid grid() const{ return m_grid;}

    void def( std::string name, const dg::file::JsonType& atts = {})
    {
        // this enables time writing !
        if ( m_varids.find( name) == m_varids.end()) // we did not find it
        {
            do_def( name, atts);
        }
    }
    template<class HostContainer>
    void put( std::string name, const HostContainer& data, unsigned slice=0) const
    {
        if( m_grid.ndim() == m_dims.size())
            dg::file::put_var( *m_ncid, m_varids.at(name), m_grid, data);
        else
            dg::file::put_vara( *m_ncid, m_varids.at(name), slice, m_grid, data);
    }
    template<class HostContainer>
    void def_and_put( std::string name, const dg::file::JsonType& atts, const HostContainer& data)
    {
        def( name, atts);
        put( name, data, 0);
    }
    private:
#ifdef MPI_VERSION
    // Help SFINAE
    //https://stackoverflow.com/questions/11531989/what-happened-to-my-sfinae-redux-conditional-template-class-members
    template<class T = Topology>
    std::enable_if_t<dg::is_mpi_grid<T>::value,void >
         do_def( std::string name, const dg::file::JsonType& atts)
    {
        m_varids[name] = 0;// all processes are aware that variable exists
        int rank;
        MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        dg::file::NC_Error_Handle err;
        if( rank==0)
        {
            err = nc_def_var( *m_ncid, name.data(), getNCDataType<typename Topology::value_type>(),
                m_dims.size(), &m_dims[0], &m_varids.at(name));
            dg::file::json2nc_attrs( atts, *m_ncid, m_varids.at(name));
        }
    }
#endif // MPI_VERSION
    template<class T = Topology>
    std::enable_if_t<dg::is_shared_grid<T>::value,void >
        do_def( std::string name, const dg::file::JsonType& atts)
    {
        dg::file::NC_Error_Handle err;
        err = nc_def_var( *m_ncid, name.data(), getNCDataType<typename Topology::value_type>(),
            m_dims.size(), &m_dims[0], &m_varids[name]);
        dg::file::json2nc_attrs( atts, *m_ncid, m_varids.at(name));
    }

    const int* m_ncid;
    std::vector<int> m_dims;
    std::map<std::string,int> m_varids;
    typename Topology::host_grid m_grid;
};


///@cond
namespace detail
{

template<class Signature>
struct get_first_argument_type;

template<class R, class Arg1, class ...A>
struct get_first_argument_type<R(Arg1, A...)>
{
    using type = Arg1;
};
template<class Signature>
using get_first_argument_type_t = std::decay_t<typename get_first_argument_type<Signature>::type>;

template<class Signature>
using get_result_type_t = typename std::function<Signature>::result_type;
}//namespace detail
///@endcond

template<class Signature>
struct Record
{
    using SignatureType = Signature;
    std::string name, long_name;
    std::function<Signature> function;
};


/**
 * @brief A class to write (time-dependent) variables from a record list into a netcdf file
 * @note in an MPI program all processes have to create the class and call its methods. The
 * class automatically takes care of which threads write to file.
 */
template<class Topology>
struct WriteRecordsList
{
    /**
     * @brief Create variables ids
     *
     * For each record in \c records create a variable named \c record.name with
     * attribute \c record.long_name of dimension \c ndim in group \c ncid with
     * dimensions given by \c dim_ids
     * @tparam ListClass
     * @param records list of records to put into ncid
     */
    template<class ListClass>
    WriteRecordsList( const int& ncid, const Topology& g, std::vector<std::string> dim_names, const ListClass& records) : m_start(0), m_writer( ncid, g, dim_names)
    {
        for( auto& record : records)
        {
            dg::file::JsonType att;
            att["long_name"] = record.long_name;
            m_writer.def( record.name, att);
        }
    }

    /**
     * @brief Write variables created from record list
     *
     * For each record in \c records call \c record.function( resultD, ps...)
     * where \c resultD is a \c dg::x::DVec of size given by \c grid and write
     * into \c ncid
     * @tparam ListClass
     * @tparam Params
     * @param records list of records to put into ncid
     * @param ps Parameters forwarded to \c record.function( resultD, ps...) or \c result = record.function( ps...)
     */
    template< class ListClass, class ... Params >
    void write( const ListClass& records, Params&& ...ps)
    {
        do_write( records, std::forward<Params>(ps)...);
    }
    private:
    template< class ListClass, class ... Params >
    std::enable_if_t<std::is_same<detail::get_result_type_t<typename ListClass::value_type::SignatureType> ,void>::value >  do_write( const ListClass& records, Params&& ...ps)
    {
        //vector write
        auto resultD =
            dg::construct<detail::get_first_argument_type_t<typename ListClass::value_type::SignatureType>>(
                dg::evaluate( dg::zero, m_writer.grid()));
        auto resultH = dg::evaluate( dg::zero, m_writer.grid());
        for( auto& record : records)
        {
            record.function( resultD, std::forward<Params>(ps)...);
            dg::assign( resultD, resultH);
            m_writer.put( record.name, resultH, m_start);
        }
        m_start++;
    }
    template< class ListClass, class ... Params >
    std::enable_if_t<!std::is_same<detail::get_result_type_t<typename ListClass::value_type::SignatureType> ,void>::value >  do_write( const ListClass& records, Params&& ...ps)
    {
        // scalar writes
        for( auto& record : records)
        {
            auto result = record.function( std::forward<Params>(ps)...);
            m_writer.put( record.name, result, m_start);
        }
        m_start++;
    }

    size_t m_start;
    Writer<Topology>  m_writer;
};

/**
 * @brief Write variables created from record list and projected to smaller grid
 *
 * For each record in \c records call \c record.function( resultD, ps...)
 * where \c resultD is a \c dg::x::DVec of size given by \c grid and write
 * its projection to \c grid_out into \c ncid
 * @tparam Topology A topology
 * @tparam ListClass
 * @tparam Params
 * @param ncid root or group id in a netcdf file
 * @param grid gives the shape of the result of \c record.function
 * @param grid_out gives the shape of the output variable (shape must be
 * consistent with \c dim_ids in constructor)
 * @param records list of records to put into ncid
 * @param ps Parameters forwarded to \c record.function( resultD, ps...)
 */
template<class Topology, class MatrixType, class ContainerType>
struct ProjectRecordsList
{
    template<class ListClass>
    ProjectRecordsList( const int& ncid, const Topology& grid, const Topology& grid_out, std::vector<std::string> dim_names, const ListClass& records): m_writer( ncid, grid_out, dim_names)
    {
        m_projectD =
            dg::create::fast_projection( grid, grid.n()/grid_out.n(),
                grid.Nx()/grid_out.Nx(), grid.Ny()/grid_out.Ny());
        m_resultD = dg::evaluate( dg::zero, grid);
        m_transferD = dg::evaluate(dg::zero, grid_out);
        for( auto& record : records)
        {
            dg::file::JsonType att;
            att["long_name"] = record.long_name;
            m_writer.def( record.name, att);
        }
    }
    template<class ListClass, class ... Params >
    void write( const ListClass& records, Params&& ...ps)
    {
        auto transferH = dg::evaluate(dg::zero, m_writer.grid());
        for( auto& record : records)
        {
            record.function( m_resultD, std::forward<Params>(ps)...);
            dg::blas2::symv( m_projectD, m_resultD, m_transferD);
            dg::assign( m_transferD, transferH);
            m_writer.put( record.name, transferH);
        }
        m_start++;
    }

    private:
    size_t m_start;
    typename Topology::host_vector m_resultH;
    ContainerType m_resultD, m_transferD;
    dg::file::Writer<Topology> m_writer;
    dg::MultiMatrix<MatrixType,ContainerType> m_projectD;

};
///@}


}//namespace file
}//namespace dg
