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

    //TODO where is file MPI_Comm in put_var?

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
        // General problem: HDF5 may use file locks to prevent multiple processes
        // from opening the same file for write at the same time
        m_comm = MPI_COMM_WORLD;
        int rank;
        MPI_Comm_rank( m_comm, &rank);
        m_rank0 = (rank == 0);
        m_readonly = ( mode == nc_nowrite);
        // Classic file access, one process writes, everyone else reads
        if( m_readonly or m_rank0)
        {
            m_file.open( path, mode);
        }
    }
    /// Check if a file is open
    bool is_open() const
    {
        bool boolean;
        if( m_readonly or m_rank0)
            boolean = m_file.is_open();
        return mpi_bcast( boolean);
    }

    void close()
    {
        m_file.close();
        MPI_Barrier( m_comm); // all ranks agree that file is closed
        // removes lock from file
    }

    void sync()
    {
        if( m_readonly or m_rank0)
            m_file.sync();
    }

    /////////////// Groups /////////////////
    void def_grp( std::string name)
    {
        if( m_rank0)
            m_file.def_grp(name);
    }
    void def_grp_p( std::filesystem::path path)
    {
        if( m_rank0)
            m_file.def_grp_p(path);
    }
    bool grp_exists( std::filesystem::path path) const
    {
        bool boolean;
        if( m_readonly or m_rank0)
            boolean = m_file.grp_exists(path);
        return mpi_bcast( boolean);
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

    int get_grpid() const
    {
        int integer;
        if( m_readonly or m_rank0)
            integer = m_file.get_grpid();
        return mpi_bcast( integer);
    }

    std::filesystem::path get_grp_path( ) const
    {
        std::filesystem::path path;
        if( m_readonly or m_rank0)
            path = m_file.get_grp_path();
        return mpi_bcast( path);
    }

    std::vector<std::filesystem::path> get_grps( ) const
    {
        std::vector<std::filesystem::path> grps;
        if( m_readonly or m_rank0)
            grps = m_file.get_grps();
        return mpi_bcast( grps);
    }
    std::vector<std::filesystem::path> get_grps_r( ) const
    {
        std::vector<std::filesystem::path> grps;
        if( m_readonly or m_rank0)
            grps = m_file.get_grps_r();
        return mpi_bcast( grps);
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
        size_t size;
        if( m_readonly or m_rank0)
            size = m_file.dim_size( name);
        return mpi_bcast( size);
    }

    std::vector<size_t> dims_shape( const std::vector<std::string>& dims) const
    {
        std::vector<size_t> size;
        if( m_readonly or m_rank0)
            size = m_file.dims_shape( dims);
        return mpi_bcast( size);
    }
    std::vector<std::string> get_dims() const
    {
        std::vector<std::string> strings;
        if( m_readonly or m_rank0)
            strings = m_file.get_dims( );
        return mpi_bcast( strings) ;
    }
    bool dim_exists( std::string name) const
    {
        bool boolean;
        if( m_readonly or m_rank0)
            boolean = m_file.dim_exists(name);
        return mpi_bcast( boolean);
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
        nc_att_t att;
        if( m_readonly or m_rank0)
            att = m_file.get_att_t( id, att_name);
        return mpi_bcast( att);
    }

    template<class T>
    T get_att_i( std::string id, std::string att_name, unsigned idx = 0) const
    {
        T att;
        if( m_readonly or m_rank0)
            att = m_file.get_att_i<T>( id, att_name, idx);
        return mpi_bcast( att);
    }

    // This works for compound types
    template<class T>
    std::vector<T> get_att_v( std::string id, std::string att_name) const
    {
        std::vector<T> att;
        if( m_readonly or m_rank0)
            att = m_file.get_att_v<T>( id, att_name);
        return mpi_bcast( att);
    }
    template<class T>
    T get_att( std::string id, std::string att_name) const
    {
        T att;
        if( m_readonly or m_rank0)
            att = m_file.get_att<T>( id, att_name);
        return mpi_bcast( att);
    }

    template<class T>
    std::map<std::string, T> get_atts( std::string id = ".") const
    {
        std::map<std::string, T> atts;
        if( m_readonly or m_rank0)
            atts = m_file.get_atts<T>( id);
        return mpi_bcast( atts);
    }
    //std::vector<std::tuple<std::string, nc_type, std::any>> get_atts( std::string id = ".") const;

    /// Remove an attribute
    void del_att( std::string id, std::string att)
    {
        if( m_rank0)
            m_file.del_att( id, att);
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
        unsigned size = abscissas.size(), global_size = 0;
        MPI_Reduce( &size, &global_size, 1, MPI_UNSIGNED, MPI_SUM, 0, m_comm);
        if( m_rank0)
            m_file.defput_dim<dg::get_value_type<ContainerType>>( name,
                global_size, atts);
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
        file::NC_Error_Handle err;
        int grpid = 0, varid = 0;
        grpid = m_file.get_grpid();
        err = nc_inq_varid( grpid, name.c_str(), &varid);

        using value_type = dg::get_value_type<ContainerType>;
        auto& receive = m_receive.template get<value_type>( );
        auto& data_ref = get_ref( data, dg::get_tensor_category<ContainerType>());

        if constexpr ( std::is_same_v<dg::get_execution_policy<ContainerType>,
            dg::CudaTag>)
        {
            m_buffer.template set<value_type>( data.size());
            const auto& buffer = m_buffer.template get<value_type>( );
            if( m_readonly)
                err = detail::get_vara_T( grpid, varid,
                    slab.startp(), slab.countp(), buffer.data());
            else
                detail::get_vara_detail( grpid, varid, slab, buffer, receive);
            dg::assign ( buffer, data_ref);
        }
        else
        {
            if( m_readonly)
                err = detail::get_vara_T( grpid, varid,
                    slab.startp(), slab.countp(), data_ref.data());
            else
                detail::get_vara_detail( grpid, varid, slab, data_ref, receive);
        }
    }

    bool var_exists( std::string name) const
    {
        bool boolean;
        if( m_readonly or m_rank0)
            boolean = m_file.var_exists(name);
        return mpi_bcast( boolean);
    }

    std::vector<NcVariable> get_vars() const
    {
        std::vector<NcVariable> vars;
        if( m_readonly or m_rank0)
            vars = m_file.get_vars();
        return mpi_bcast( vars);
    }


    private:
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

    template<class T>
    T mpi_bcast( T data) const
    {
        if( not m_readonly)
            MPI_Bcast( &data, 1, dg::getMPIDataType<T>(), 0, m_comm);
        return data;
    }
    std::string mpi_bcast( std::string data) const
    {
        if( not m_readonly)
        {
            size_t len = data.size();
            MPI_Bcast( &len, 1, dg::getMPIDataType<size_t>(), 0, m_comm);
            data.resize( len, 'x');
            MPI_Bcast( &data[0], len, MPI_CHAR, 0, m_comm);
        }
        return data;
    }
    std::filesystem::path mpi_bcast( std::filesystem::path data) const
    {
        if( not m_readonly)
        {
            std::string name = data.generic_string();
            name = mpi_bcast( name);
            data = name;
        }
        return data;
    }
    NcVariable mpi_bcast( NcVariable data)
    {
        if( not m_readonly)
        {
            std::string name = data.name;
            data.name = mpi_bcast( name);
            int xtype = data.xtype;
            data.xtype = mpi_bcast( xtype);
            auto dims = data.dims;
            data.dims = mpi_bcast( dims);
        }
        return data;
    }

    template<class T>
    std::vector<T> mpi_bcast( std::vector<T> data) const
    {
        if( not m_readonly)
        {
            size_t len = data.size();
            MPI_Bcast( &len, 1, dg::getMPIDataType<size_t>(), 0, m_comm);
            data.resize( len);
            for( unsigned u=0; u<len; u++)
                data[u] = mpi_bcast( data[u]);
        }
        return data;
    }
    template<class K, class T>
    std::map<K, T> mpi_bcast( const std::map<K,T>& data) const
    {
        if( not m_readonly)
        {
            size_t len = data.size();
            MPI_Bcast( &len, 1, dg::getMPIDataType<size_t>(), 0, m_comm);
            std::vector<K> keys;
            std::vector<T> values;
            if( m_rank0)
            {
                for ( const auto& pair : data)
                {
                    keys.push_back( pair.first);
                    values.push_back( pair.second);
                }
            }
            keys = mpi_bcast( keys);
            values = mpi_bcast( values);
            if( not m_rank0)
            {
                std::map<K,T> tmp;
                for( unsigned u=0; u<len; u++)
                    tmp[keys[u]] = values[u];
                return tmp;
            }
        }
        return data;
    }
    nc_att_t mpi_bcast( const nc_att_t& data) const
    {
        nc_att_t tmp;
        if( not m_readonly)
        {
            tmp = std::visit( [this]( auto&& arg) { return nc_att_t(mpi_bcast(arg)); }, data);
        }
        return tmp;
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
