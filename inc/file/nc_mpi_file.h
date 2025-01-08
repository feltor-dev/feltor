#pragma once

#include "nc_file.h"
#include "dg/backend/mpi_datatype.h"

/*!@file
 *
 * Our take on a modern C++ implementation of the NetCDF-4 data model
 * MPI - Version
 */

namespace dg
{
namespace file
{

struct MPINcFile
{
    /////////////////////////////// CONSTRUCTORS/DESTRUCTOR /////////
    MPINcFile () = default;
    /// mpi-mode = shared, MPI_Comm = MPI_COMM_WORLD
    MPINcFile(const std::string& path, enum NcFileMode mode = nc_nowrite)
    {
        open( path, mode);
    }
    MPINcFile(const MPINcFile& rhs) = delete;
    MPINcFile& operator =(const MPINcFile & rhs) = delete;

    MPINcFile(MPINcFile&& rhs) = default;
    MPINcFile& operator =(MPINcFile && rhs) = default;

    ~MPINcFile() = default;
    ///////////////////// open/close /////////

    void open(const std::string& path, enum NcFileMode mode = nc_nowrite)
    {
        m_comm = MPI_COMM_WORLD;
        int rank;
        MPI_Comm_rank( m_comm, &rank);
        m_rank0 = (rank == 0);
        m_readonly = ( mode == nc_nowrite);
        // Classic file access, one process writes, everyone else reads
        if( m_rank0)
            m_file.open( path, mode);
        MPI_Barrier( m_comm);
        if( !m_rank0)
            m_file.open( path, nc_nowrite);
    }
    /// Check if a file is open
    bool is_open() const
    {
        return m_file.is_open();
    }

    void close() { m_file.close(); }

    void sync()
    {
        // Also the readers need to sync to update information
        m_file.sync();
        MPI_Barrier( m_comm);
    }

    /////////////// Groups /////////////////
    void def_grp( std::string name)
    {
        if( m_rank0)
            m_file.def_grp(name);
    }
    void set_grp( std::string name = "")
    {
        if( m_rank0)
            m_file.set_grp(name);
    }
    void rename_grp( std::string old_name, std::string new_name)
    {
        if( m_rank0)
            m_file.rename_grp(old_name, new_name);
    }

    std::vector<std::string> get_grps( ) const
    {
        readsync();
        return m_file.get_grps();
    }

    ////////////// Dimensions ////////////////////////
    void def_dim( std::string name, size_t size)
    {
        if( m_rank0)
            m_file.def_dim( name, size);
    }
    void rename_dim( std::string old_name, std::string new_name)
    {
        if( m_rank0)
            m_file.rename_dim( old_name, new_name);
    }
    size_t dim_size( std::string name) const
    {
        readsync();
        return m_file.dim_size( name);
    }

    std::vector<size_t> dims_shape( const std::vector<std::string>& dims) const
    {
        readsync();
        return m_file.dims_shape( dims);
    }
    std::vector<std::string> get_dims() const
    {
        readsync();
        return m_file.get_dims();
    }
    /////////////// Attributes setters
    void set_att ( std::string id, const std::pair<std::string, nc_att_t>& att)
    {
        if( m_rank0)
            m_file.set_att( id, att);
    }

    template<class S, class T> // T cannot be nc_att_t
    void set_att( std::string id, const std::tuple<S,nc_type, T>& att)
    {
        if( m_rank0)
            m_file.set_att( id, att);
    }
    // Iterable can be e.g. std::vector<std::pair...>, std::map , etc.
    template<class Iterable> // *it must be usable in set_att
    void set_atts( std::string id, const Iterable& atts)
    {
        if( m_rank0)
            m_file.set_atts( id, atts);
    }
    void set_atts( std::string id, const std::map<std::string, nc_att_t>& atts)
    {
        if( m_rank0)
            m_file.set_atts( id, atts);
    }

    /////////////////// Attribute getters

    dg::file::nc_att_t get_att_t( std::string id, std::string att_name) const
    {
        readsync();
        return m_file.get_att_t(id, att_name);
    }

    template<class T>
    T get_att_i( std::string id, std::string att_name, unsigned idx = 0) const
    {
        readsync();
        return m_file.get_att_i<T>(id, att_name, idx);
    }

    // This works for compound types
    template<class T>
    std::vector<T> get_att_v( std::string id, std::string att_name) const
    {
        readsync();
        return m_file.get_att_v<T>(id, att_name);
    }
    template<class T>
    T get_att( std::string id, std::string att_name) const
    {
        readsync();
        return m_file.get_att<T>(id, att_name);
    }

    template<class T>
    std::map<std::string, T> get_atts( std::string id = ".") const
    {
        readsync();
        return m_file.get_atts<T>(id);
    }
    //std::vector<std::tuple<std::string, nc_type, std::any>> get_atts( std::string id = ".") const;

    /// Remove an attribute
    void rm_att( std::string id, std::string att)
    {
        if( m_rank0)
            m_file.rm_att( id, att);
    }
    /// Rename an attribute
    void rename_att( std::string id, std::string old_att_name, std::string new_att_name)
    {
        if( m_rank0)
            m_file.rename_att( id, old_att_name, new_att_name);
    }


    ////////////// Variables ////////////////////////
    template<class T>
    void def_var( std::string name, std::vector<std::string> dim_names)
    {
        if( m_rank0)
            m_file.def_var<T>( name, dim_names);
    }
    void def_var( const NcVariable& var)
    {
        if( m_rank0)
            m_file.def_var( var);
    }
    template<class ContainerType>
    void put_var( std::string name, const MPINcHyperslab& slab,
            const ContainerType& data)
    {
        int grpid = 0, varid = 0;
        if( m_rank0)
        {
            grpid = m_file.get_grpid();
            file::NC_Error_Handle err;
            err = nc_inq_varid( grpid, name.c_str(), &varid);
        }
        using value_type = dg::get_value_type<ContainerType>;
        m_receive.template set<value_type>(0);
        auto& receive = m_receive.template get<value_type>( );
        const auto& data_ref = get_ref( data, dg::get_tensor_category<ContainerType>());
        if constexpr ( std::is_same_v<dg::get_execution_policy<ContainerType>,
            dg::CudaTag>)
        {
            m_buffer.template set<value_type>( data.size());
            const auto& buffer = m_buffer.template get<value_type>( );
            dg::assign ( data_ref, buffer);
            detail::put_vara_detail( grpid, varid, slab, buffer, receive);
        }
        else
            detail::put_vara_detail( grpid, varid, slab, data_ref, receive);
    }

    template<class ContainerType>
    void put_var( std::string name, const MPI_Vector<ContainerType>& data)
    {
        // Only works for 1d variable in MPI
        if( m_rank0)
        {
            int varid = 0, ndims = 0;
            int retval = nc_inq_varid( m_file.get_grpid(), name.c_str(), &varid);
            if( retval != NC_NOERR )
                throw std::runtime_error( "Variable does not exist!");
            retval = nc_inq_varndims( m_file.get_grpid(), varid, &ndims);
            assert( ndims == 1);
        }

        int count = data.size();
        MPI_Comm comm = data.communicator();
        int rank, size;
        MPI_Comm_rank( comm, &rank);
        MPI_Comm_size( comm, &size);
        std::vector<int> counts ( size);
        MPI_Allgather( &count, 1, MPI_INT, &counts[0], 1, MPI_INT, comm);
        std::vector<size_t> start(1, 0);
        for( int r=0; r<rank; r++)
            start[0] += counts[r];
        put_var( name, { start, std::vector<size_t>(1,count), comm}, data);
    }

    template<class T>
    void defput_dim( std::string name, size_t size,
            std::map<std::string, nc_att_t> atts)
    {
        if( m_rank0)
            m_file.defput_dim<T>( name, size, atts);
    }
    template<class ContainerType>
    void defput_dim( std::string name,
            std::map<std::string, nc_att_t> atts,
            const MPI_Vector<ContainerType>& abscissas)  // implicitly assume ordered by rank
    {
        if( m_rank0)
            m_file.defput_dim<dg::get_value_type<ContainerType>>( name,
                abscissas.size(), atts);
        put_var( name, abscissas);
    }

    template<class T>
    void put_var1( std::string name, const std::vector<size_t>& start, T data)
    {
        if(m_rank0)
            m_file.put_var1( name, start, data);
    }


    template<class ContainerType>
    void get_var( std::string name, const MPINcHyperslab& slab,
            ContainerType& data) const
    {
        // Reading is always possible in parallel
        readsync();
        file::NC_Error_Handle err;
        int grpid = 0, varid = 0;
        grpid = m_file.get_grpid();
        err = nc_inq_varid( grpid, name.c_str(), &varid);

        using value_type = dg::get_value_type<ContainerType>;
        auto& data_ref = get_ref( data, dg::get_tensor_category<ContainerType>());

        if constexpr ( std::is_same_v<dg::get_execution_policy<ContainerType>,
            dg::CudaTag>)
        {
            m_buffer.template set<value_type>( data.size());
            const auto& buffer = m_buffer.template get<value_type>( );
            err = detail::get_vara_T( grpid, varid,
                slab.startp(), slab.countp(), buffer.data());
            dg::assign ( buffer, data_ref);
        }
        else
            err = detail::get_vara_T( grpid, varid,
                slab.startp(), slab.countp(), data_ref.data());
    }

    bool is_defined( std::string name) const
    {
        readsync();
        return m_file.is_defined(name);
    }

    std::vector<NcVariable> get_vars() const
    {
        readsync();
        return m_file.get_vars();
    }


    private:
    void readsync() const
    {
        if( !m_readonly) // someone is writing
        {
            if( !m_file.is_open())
                throw std::runtime_error( "Can't sync a closed file!");
            NC_Error_Handle err;
            err = nc_sync( m_file.get_ncid());

            MPI_Barrier( m_comm);
        }
    }
    template<class ContainerType>
    const ContainerType& get_ref( const MPI_Vector<ContainerType>& x, dg::MPIVectorTag)
    {
        return x.data();
    }
    template<class ContainerType>
    const ContainerType& get_ref( const ContainerType& x, dg::AnyVectorTag)
    {
        return x;
    }
    template<class ContainerType>
    ContainerType& get_ref( MPI_Vector<ContainerType>& x, dg::MPIVectorTag)
    {
        return x.data();
    }
    template<class ContainerType>
    ContainerType& get_ref( ContainerType& x, dg::AnyVectorTag)
    {
        return x;
    }


    bool m_rank0;
    bool m_readonly;
    MPI_Comm m_comm;
    SerialNcFile m_file;
    // Buffer for device to host transfer, and dg::assign
    dg::detail::AnyVector<thrust::host_vector> m_buffer, m_receive;
};

using NcFile = MPINcFile;

}// namespace file
}// namespace dg
