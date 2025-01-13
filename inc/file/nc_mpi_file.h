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

/*! @brief MPI NetCDF-4 file based on **serial** NetCDF
 *
 * by funneling get and put operations through the master rank of the given communicator
 */
struct MPINcFile
{
    /////////////////////////////// CONSTRUCTORS/DESTRUCTOR /////////
    MPINcFile (MPI_Comm comm = MPI_COMM_WORLD)
    : m_comm(comm)
    {}
    MPINcFile(const std::string& path, enum NcFileMode mode = nc_nowrite, MPI_Comm comm = MPI_COMM_WORLD)
    : m_comm(comm)
    {
        open( path, mode);
    }
    MPINcFile(const MPINcFile& rhs) = delete;
    MPINcFile& operator =(const MPINcFile & rhs) = delete;

    MPINcFile(MPINcFile&& rhs) = default;
    MPINcFile& operator =(MPINcFile && rhs)
    {
        if( this!= &rhs)
        {
            // We need to check that the rhs has the same communicator
            // else we need to think about different groups (do all ranks in both
            // groups call this function?)
            int result;
            MPI_Comm_compare( this->m_comm, rhs.m_comm, &result);
            // congruent, similar and ident all should be fine
            assert( result != MPI_UNEQUAL);
            this->m_comm  = rhs.m_comm;
            this->m_rank0 = rhs.m_rank0;
            this->m_readonly = rhs.m_readonly;
            this->m_file    = std::move( rhs.m_file);
            this->m_buffer  = std::move( rhs.m_buffer);
            this->m_receive = std::move( rhs.m_receive);
        }
        return *this;
    }

    ~MPINcFile() = default;
    ///////////////////// open/close /////////

    void open(const std::string& path, enum NcFileMode mode = nc_nowrite)
    {
        // General problem: HDF5 may use file locks to prevent multiple processes
        // from opening the same file for write at the same time
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
        return mpi_invoke( &SerialNcFile::is_open, m_file);
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
    int get_ncid() const
    {
        return mpi_invoke( &SerialNcFile::get_ncid, m_file);
    }

    MPI_Comm communicator() const { return m_comm;}

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
    bool grp_is_defined( std::filesystem::path path) const
    {
        return mpi_invoke( &SerialNcFile::grp_is_defined, m_file, path);
    }
    void set_grp( std::filesystem::path path = "")
    {
        if( m_rank0)
            m_file.set_grp(path);
    }
    void rename_grp( std::string old_name, std::string new_name)
    {
        if( m_rank0)
            m_file.rename_grp(old_name, new_name);
    }

    int get_grpid() const
    {
        return mpi_invoke( &SerialNcFile::get_grpid, m_file);
    }

    std::filesystem::path get_current_path( ) const
    {
        return mpi_invoke( &SerialNcFile::get_current_path, m_file);
    }

    std::vector<std::filesystem::path> get_grps( ) const
    {
        return mpi_invoke( &SerialNcFile::get_grps, m_file);
    }
    std::vector<std::filesystem::path> get_grps_r( ) const
    {
        return mpi_invoke( &SerialNcFile::get_grps_r, m_file);
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
    size_t get_dim_size( std::string name) const
    {
        return mpi_invoke( &SerialNcFile::get_dim_size, m_file, name);
    }

    std::vector<size_t> get_dims_shape( const std::vector<std::string>& dims) const
    {
        return mpi_invoke( &SerialNcFile::get_dims_shape, m_file, dims);
    }
    std::vector<std::string> get_dims(bool include_parents = true) const
    {
        return mpi_invoke( &SerialNcFile::get_dims, m_file, include_parents);
    }
    std::vector<std::string> get_unlim_dims() const
    {
        return mpi_invoke( &SerialNcFile::get_unlim_dims, m_file);
    }
    bool dim_is_defined( std::string name) const
    {
        return mpi_invoke( &SerialNcFile::dim_is_defined, m_file, name);
    }
    /////////////// Attributes setters
    void put_att ( std::string id, const std::pair<std::string, nc_att_t>& att)
    {
        if( m_rank0)
            m_file.put_att( id, att);
    }

    template<class S, class T> // T cannot be nc_att_t
    void put_att( std::string id, const std::tuple<S,nc_type, T>& att)
    {
        if( m_rank0)
            m_file.put_att( id, att);
    }
    // Iterable can be e.g. std::vector<std::pair...>, std::map , etc.
    template<class Iterable> // *it must be usable in put_att
    void put_atts( std::string id, const Iterable& atts)
    {
        if( m_rank0)
            m_file.put_atts( id, atts);
    }
    void put_atts( std::string id, const std::map<std::string, nc_att_t>& atts)
    {
        if( m_rank0)
            m_file.put_atts( id, atts);
    }

    /////////////////// Attribute getters

    template<class T>
    T get_att_as( std::string id, std::string att_name) const
    {
        return mpi_invoke( &SerialNcFile::get_att_as<T>, m_file, id, att_name);
    }
    template<class T>
    std::vector<T> get_att_vec_as( std::string id, std::string att_name) const
    {
        return mpi_invoke( &SerialNcFile::get_att_vec_as<T>, m_file, id,
            att_name);
    }

    template<class T>
    std::map<std::string, T> get_atts_as( std::string id = ".") const
    {
        return mpi_invoke( &SerialNcFile::get_atts_as<T>, m_file, id);
    }
    /// Short for <tt> get_atts_as<nc_att_t>( id)
    std::map<std::string, nc_att_t> get_atts( std::string id = ".") const
    {
        return get_atts_as<nc_att_t>( id);
    }

    /// Remove an attribute
    void del_att( std::string id, std::string att)
    {
        if( m_rank0)
            m_file.del_att( id, att);
    }
    /// Check for existence of the attribute named \c att_name
    bool att_is_defined( std::string id, std::string att_name) const
    {
        return mpi_invoke( &SerialNcFile::att_is_defined, m_file, id, att_name);
    }
    /// Rename an attribute
    void rename_att( std::string id, std::string old_att_name, std::string new_att_name)
    {
        if( m_rank0)
            m_file.rename_att( id, old_att_name, new_att_name);
    }


    ////////////// Variables ////////////////////////
    template<class T>
    void def_var_as( std::string name, std::vector<std::string> dim_names)
    {
        if( m_rank0)
            m_file.def_var_as<T>( name, dim_names);
    }
    void def_var( std::string name, nc_type xtype,
            std::vector<std::string> dim_names)
    {
        if( m_rank0)
            m_file.def_var( name, xtype, dim_names);
    }
    /*! @brief
     *
     * @attention Only works for 1d variables in MPI, in which
     * case the rank of the calling process determines where the data is written to
     */
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

    template<class ContainerType, typename = std::enable_if_t<dg::is_not_scalar<ContainerType>::value>>
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
            detail::put_vara_detail( grpid, varid, slab, buffer, receive, m_comm);
        }
        else
            detail::put_vara_detail( grpid, varid, slab, data_ref, receive, m_comm);
    }

    template<class T, typename = std::enable_if_t<dg::is_scalar<T>::value>>
    void put_var( std::string name, const std::vector<size_t>& start, T data)
    {
        if(m_rank0)
            m_file.put_var( name, start, data);
    }

    template<class T>
    void defput_dim_as( std::string name, size_t size,
            const std::map<std::string, nc_att_t>& atts)
    {
        if( m_rank0)
            m_file.defput_dim_as<T>( name, size, atts);
    }
    template<class ContainerType>
    void defput_dim( std::string name,
            std::map<std::string, nc_att_t> atts,
            const MPI_Vector<ContainerType>& abscissas)  // implicitly assume ordered by rank
    {
        unsigned size = abscissas.size(), global_size = 0;
        MPI_Reduce( &size, &global_size, 1, MPI_UNSIGNED, MPI_SUM, 0, m_comm);
        if( m_rank0)
            m_file.defput_dim_as<dg::get_value_type<ContainerType>>( name,
                global_size, atts);
        put_var( name, abscissas);
    }


// The comm in MPINcHyperslab must be at least a subgroup of m_comm
    template<class ContainerType, typename = std::enable_if_t<dg::is_not_scalar<ContainerType>::value>>
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
                detail::get_vara_detail( grpid, varid, slab, buffer, receive, m_comm);
            dg::assign ( buffer, data_ref);
        }
        else
        {
            if( m_readonly)
                err = detail::get_vara_T( grpid, varid,
                    slab.startp(), slab.countp(), data_ref.data());
            else
                detail::get_vara_detail( grpid, varid, slab, data_ref, receive, m_comm);
        }
    }

    template<class T, typename = std::enable_if_t<dg::is_scalar<T>::value>>
    void get_var( std::string name, const std::vector<size_t>& start, T& data) const
    {
        if( m_readonly or m_rank0)
            m_file.get_var( name, start, data);
        if( not m_readonly)
            mpi_bcast( data);
    }

    bool var_is_defined( std::string name) const
    {
        return mpi_invoke( &SerialNcFile::var_is_defined, m_file, name);
    }

    nc_type get_var_type(std::string name) const
    {
        return mpi_invoke( &SerialNcFile::get_var_type, m_file, name);
    }

    std::vector<std::string> get_var_dims(std::string name) const
    {
        return mpi_invoke( &SerialNcFile::get_var_dims, m_file, name);
    }

    std::vector<std::string> get_vars() const
    {
        return mpi_invoke( &SerialNcFile::get_vars, m_file);
    }

    std::map<std::filesystem::path, std::vector<std::string>> get_vars_r()
        const
    {
        return mpi_invoke( &SerialNcFile::get_vars_r, m_file);
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
    void mpi_bcast( T& data) const
    {
        MPI_Bcast( &data, 1, dg::getMPIDataType<T>(), 0, m_comm);
    }
    void mpi_bcast( std::string& data) const
    {
        size_t len = data.size();
        MPI_Bcast( &len, 1, dg::getMPIDataType<size_t>(), 0, m_comm);
        data.resize( len, 'x');
        MPI_Bcast( &data[0], len, MPI_CHAR, 0, m_comm);
    }
    void mpi_bcast( std::filesystem::path& data) const
    {
        std::string name = data.generic_string();
        mpi_bcast( name);
        data = name;
    }

    template<class T>
    void mpi_bcast( std::vector<T>& data) const
    {
        size_t len = data.size();
        MPI_Bcast( &len, 1, dg::getMPIDataType<size_t>(), 0, m_comm);
        data.resize( len);

        for( unsigned u=0; u<len; u++)
            if constexpr ( std::is_same_v<T, bool>)
            {
                bool b = data[u];
                mpi_bcast( b);
                data[u] = b;
            }
            else
                mpi_bcast( data[u]);
    }
    template<class K, class T>
    void mpi_bcast( std::map<K,T>& data) const
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
        mpi_bcast( keys);
        mpi_bcast( values);
        if( not m_rank0)
        {
            for( unsigned u=0; u<len; u++)
                data[keys[u]] = values[u];
        }
    }
    void mpi_bcast( nc_att_t& data) const
    {
        std::visit( [this]( auto&& arg) { mpi_bcast(arg); }, data);
    }

    template<class F, class ... Args>
    std::invoke_result_t<F, Args...> mpi_invoke( F&& f, Args&& ...args) const
    {
        using R = std::invoke_result_t<F, Args...>;
        if ( m_readonly)
        {
            return std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        }
        R r;
        if( m_rank0)
        {
            r = std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
        }
        mpi_bcast( r);
        return r;
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
