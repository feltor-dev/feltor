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

/*! @brief MPI NetCDF-4 file based on **serial** NetCDF
 *
 * by funneling all file operations through the rank 0 of the given
 * communicator. In general, only the rank 0 actually opens, reads and writes
 * to the file on disk with the exception of the \c nc_nowrite file mode where
 * all ranks open the file. When metadata like attributes, groups and
 * dimensions are read all ranks receive the same information (by \c MPI_Bcast
 * i.e. there is MPI communication, or a parallel read without communication
 * for \c nc_nowrite) and all ranks should call the write members with the same
 * information even if only the rank 0 actually uses it.
 *
 * When variables are read and written all ranks write/read a different chunk of the data
 * except for scalar variables which are broadcast to all ranks on read.
 * If \c nc_nowrite the reads are parallel and involve no MPI communication, otherwise
 * all data is communicated to/from rank 0.
 *
 * @note When compiling you thus need to link only to the normal **serial**
 * NetCDF-C library, **not parallel** NetCDF
 * @attention All ranks in the communicator \c comm given in the constructor
 * must participate in **all** member function calls. No exceptions!. Even if
 * e.g. the data to write only lies distributed only on a subgroup of ranks.
 * @sa SerialNcFile
 * @ingroup Cpp
 */
struct MPINcFile
{
    // ///////////////////////////// CONSTRUCTORS/DESTRUCTOR /////////
    /*! @brief Construct a File Handle not associated to any file
     *
     * @param comm All ranks in comm must participate in all subsequent member
     * function calls
     */
    MPINcFile (MPI_Comm comm = MPI_COMM_WORLD)
    : m_comm(comm)
    {}
    /*! @copydoc SerialNcFile::SerialNcFile(const std::filesystem::path&,enum NcFileMode)
     * @param comm All ranks in comm must participate in all subsequent member
     * function calls
     */
    MPINcFile(const std::filesystem::path& filename, enum NcFileMode mode =
        nc_nowrite, MPI_Comm comm = MPI_COMM_WORLD)
    : m_comm(comm)
    {
        open( filename, mode);
    }
    ///@copydoc SerialNcFile::SerialNcFile(const SerialNcFile&)
    MPINcFile(const MPINcFile& rhs) = delete;
    ///@copydoc SerialNcFile::operator=(const SerialNcFile&)
    MPINcFile& operator =(const MPINcFile & rhs) = delete;

    ///@copydoc SerialNcFile::SerialNcFile(SerialNcFile&&)
    MPINcFile(MPINcFile&& rhs) = default;
    ///@copydoc SerialNcFile::SerialNcFile(SerialNcFile&&)
    ///@note The communicator gruops of \c *this and \c rhs must the same
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

    ///@copydoc SerialNcFile::~SerialNcFile
    ~MPINcFile() = default;
    // /////////////////// open/close /////////

    /*!@copydoc SerialNcFile::open
     * @note if <tt> mode == nc_nowrite</tt> all ranks in comm open the file and
     *  the read member functions involve no communication
     * @note May invoke \c MPI_Barrier so that all ranks see the existence of a
     * possibly new file
     */
    void open(const std::filesystem::path& filename,
            enum NcFileMode mode = nc_nowrite)
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
            m_file.open( filename, mode);
        }
        MPI_Barrier( m_comm); // all ranks agree that file exists
    }
    ///@copydoc SerialNcFile::is_open
    ///  All MPI ranks agree if a file is open
    bool is_open() const
    {
        return mpi_invoke( &SerialNcFile::is_open, m_file);
    }

    /*!@copydoc SerialNcFile::close
     * @note May invoke \c MPI_Barrier so that \c file.open with \c nc_nowrite
     *  can be called directly afterwards (the HDF-5 file-lock needs to be released)
     */
    void close()
    {
        m_file.close();
        MPI_Barrier( m_comm); // all ranks agree that file is closed
        // removes lock from file
    }

    ///@copydoc SerialNcFile::sync
    void sync()
    {
        if( m_readonly or m_rank0)
            m_file.sync();
    }
    ///@copydoc SerialNcFile::get_ncid
    /// In MPI all ranks get an ncid but only rank 0 can actually use it,
    ///except for \c nc_nowrite / when every rank can use it
    int get_ncid() const
    {
        return mpi_invoke( &SerialNcFile::get_ncid, m_file);
    }

    /// Return MPI communicator set in constructor
    MPI_Comm communicator() const { return m_comm;}

    // ///////////// Groups /////////////////
    ///@copydoc SerialNcFile::def_grp
    void def_grp( std::string name)
    {
        if( m_rank0)
            m_file.def_grp(name);
    }
    ///@copydoc SerialNcFile::def_grp_p
    void def_grp_p( std::filesystem::path path)
    {
        if( m_rank0)
            m_file.def_grp_p(path);
    }
    ///@copydoc SerialNcFile::grp_is_defined
    bool grp_is_defined( std::filesystem::path path) const
    {
        return mpi_invoke( &SerialNcFile::grp_is_defined, m_file, path);
    }
    ///@copydoc SerialNcFile::set_grp
    void set_grp( std::filesystem::path path = "")
    {
        if( m_rank0)
            m_file.set_grp(path);
    }
    ///@copydoc SerialNcFile::rename_grp
    void rename_grp( std::string old_name, std::string new_name)
    {
        if( m_rank0)
            m_file.rename_grp(old_name, new_name);
    }

    ///@copydoc SerialNcFile::get_grpid
    int get_grpid() const
    {
        return mpi_invoke( &SerialNcFile::get_grpid, m_file);
    }

    ///@copydoc SerialNcFile::get_current_path
    std::filesystem::path get_current_path( ) const
    {
        return mpi_invoke( &SerialNcFile::get_current_path, m_file);
    }

    ///@copydoc SerialNcFile::get_grps
    std::vector<std::filesystem::path> get_grps( ) const
    {
        return mpi_invoke( &SerialNcFile::get_grps, m_file);
    }
    ///@copydoc SerialNcFile::get_grps_r
    std::vector<std::filesystem::path> get_grps_r( ) const
    {
        return mpi_invoke( &SerialNcFile::get_grps_r, m_file);
    }

    // //////////// Dimensions ////////////////////////
    ///@copydoc SerialNcFile::def_dim
    void def_dim( std::string name, size_t size)
    {
        if( m_rank0)
            m_file.def_dim( name, size);
    }
    ///@copydoc SerialNcFile::rename_dim
    void rename_dim( std::string old_name, std::string new_name)
    {
        if( m_rank0)
            m_file.rename_dim( old_name, new_name);
    }
    ///@copydoc SerialNcFile::get_dim_size
    size_t get_dim_size( std::string name) const
    {
        return mpi_invoke( &SerialNcFile::get_dim_size, m_file, name);
    }

    ///@copydoc SerialNcFile::get_dims_shape
    std::vector<size_t> get_dims_shape( const std::vector<std::string>& dims) const
    {
        return mpi_invoke( &SerialNcFile::get_dims_shape, m_file, dims);
    }
    ///@copydoc SerialNcFile::get_dims
    std::vector<std::string> get_dims(bool include_parents = true) const
    {
        return mpi_invoke( &SerialNcFile::get_dims, m_file, include_parents);
    }
    ///@copydoc SerialNcFile::get_unlim_dims
    std::vector<std::string> get_unlim_dims() const
    {
        return mpi_invoke( &SerialNcFile::get_unlim_dims, m_file);
    }
    ///@copydoc SerialNcFile::dim_is_defined
    bool dim_is_defined( std::string name) const
    {
        return mpi_invoke( &SerialNcFile::dim_is_defined, m_file, name);
    }
    // ///////////// Attributes setters
    ///@copydoc SerialNcFile::put_att
    void put_att ( std::string id, const std::pair<std::string, nc_att_t>& att)
    {
        if( m_rank0)
            m_file.put_att( id, att);
    }
    ///@copydoc SerialNcFile::put_att<S,T>
    template<class S, class T>
    void put_att( std::string id, const std::tuple<S,nc_type, T>& att)
    {
        if( m_rank0)
            m_file.put_att( id, att);
    }
    ///@copydoc SerialNcFile::put_atts(std::string,const Iterable&)
    template<class Attributes = std::map<std::string, nc_att_t> > // *it must be usable in put_att
    void put_atts( std::string id, const Attributes& atts)
    {
        if( m_rank0)
            m_file.put_atts( id, atts);
    }
    ///@copydoc SerialNcFile::put_atts(std::string,const std::map<std::string,nc_att_t>&)
    void put_atts( std::string id, const std::map<std::string, nc_att_t>& atts)
    {
        if( m_rank0)
            m_file.put_atts( id, atts);
    }

    // ///////////////// Attribute getters

    ///@copydoc SerialNcFile::get_att_as
    template<class T>
    T get_att_as( std::string id, std::string att_name) const
    {
        return mpi_invoke( &SerialNcFile::get_att_as<T>, m_file, id, att_name);
    }
    ///@copydoc SerialNcFile::get_att_vec_as
    template<class T>
    std::vector<T> get_att_vec_as( std::string id, std::string att_name) const
    {
        return mpi_invoke( &SerialNcFile::get_att_vec_as<T>, m_file, id,
            att_name);
    }

    ///@copydoc SerialNcFile::get_atts_as
    template<class T>
    std::map<std::string, T> get_atts_as( std::string id = ".") const
    {
        return mpi_invoke( &SerialNcFile::get_atts_as<T>, m_file, id);
    }
    ///@copydoc SerialNcFile::get_atts
    std::map<std::string, nc_att_t> get_atts( std::string id = ".") const
    {
        return get_atts_as<nc_att_t>( id);
    }

    ///@copydoc SerialNcFile::del_att
    void del_att( std::string id, std::string att)
    {
        if( m_rank0)
            m_file.del_att( id, att);
    }
    ///@copydoc SerialNcFile::att_is_defined
    bool att_is_defined( std::string id, std::string att_name) const
    {
        return mpi_invoke( &SerialNcFile::att_is_defined, m_file, id, att_name);
    }
    ///@copydoc SerialNcFile::rename_att
    void rename_att( std::string id, std::string old_att_name, std::string new_att_name)
    {
        if( m_rank0)
            m_file.rename_att( id, old_att_name, new_att_name);
    }

    // //////////// Variables ////////////////////////
    ///@copydoc SerialNcFile::def_var_as
    template<class T, class Attributes = std::map<std::string, nc_att_t>>
    void def_var_as( std::string name,
        const std::vector<std::string>& dim_names,
        const Attributes& atts = {})
    {
        if( m_rank0)
            m_file.def_var_as<T>( name, dim_names, atts);
    }
    ///@copydoc SerialNcFile::def_var
    template<class Attributes = std::map<std::string, nc_att_t>>
    void def_var( std::string name, nc_type xtype,
            const std::vector<std::string>& dim_names,
            const Attributes& atts = {})
    {
        if( m_rank0)
            m_file.def_var( name, xtype, dim_names, atts);
    }

    ///@copydoc SerialNcFile::put_var(std::string,const NcHyperslab&,const ContainerType&)
    /// @note The \c ContainerType in MPI can have either a \c
    ///dg::SharedVectorTag or \c dg::MPIVectorTag (It is the communicator of
    ///the slab that counts, the data communicator if present is ignored)
    template<class ContainerType, typename = std::enable_if_t<
        dg::is_vector_v<ContainerType, dg::SharedVectorTag> or
        dg::is_vector_v<ContainerType, dg::MPIVectorTag>>>
    void put_var( std::string name, const MPINcHyperslab& slab,
            const ContainerType& data)
    {
        int grpid = 0, varid = 0;
        if( m_rank0)
        {
            grpid = m_file.get_grpid();
            file::NC_Error_Handle err;
            err = nc_inq_varid( grpid, name.c_str(), &varid);
            int ndims;
            err = nc_inq_varndims( grpid, varid, &ndims);
            assert( (unsigned)ndims == slab.ndim());
        }
        using value_type = dg::get_value_type<ContainerType>;
        m_receive.template set<value_type>(0);
        auto& receive = m_receive.template get<value_type>( );
        const auto& data_ref = get_ref( data, dg::get_tensor_category<ContainerType>());
        if constexpr ( dg::has_policy_v<ContainerType, dg::CudaTag>)
        {
            m_buffer.template set<value_type>( data.size());
            auto& buffer = m_buffer.template get<value_type>( );
            dg::assign ( data_ref, buffer);
            detail::put_vara_detail( grpid, varid, slab, buffer, receive, m_comm);
        }
        else
            detail::put_vara_detail( grpid, varid, slab, data_ref, receive, m_comm);
    }

    ///@copydoc SerialNcFile::put_var(std::string,const std::vector<size_t>&,T)
    /// @note In MPI only the rank 0 writes data
    template<class T, typename = std::enable_if_t<dg::is_scalar_v<T>>>
    void put_var( std::string name, const std::vector<size_t>& start, T data)
    {
        if(m_rank0)
            m_file.put_var( name, start, data);
    }

    ///@copydoc SerialNcFile::defput_dim_as
    template<class T, class Attributes = std::map<std::string, nc_att_t>>
    void defput_dim_as( std::string name, size_t size, const Attributes& atts)
    {
        if( m_rank0)
            m_file.defput_dim_as<T>( name, size, atts);
    }
    /*! @copydoc SerialNcFile::defput_dim
     *
     * @note We use \c MPI_Reduce with \c abscissas.size() and \c
     * abscissas.communicator() to get the size of the dimension in MPI.
     */
    template<class ContainerType, class Attributes = std::map<std::string, nc_att_t>>
    void defput_dim( std::string name,
            const Attributes& atts,
            const MPI_Vector<ContainerType>& abscissas)  // implicitly assume ordered by rank
    {
        unsigned size = abscissas.size(), global_size = 0;
        MPI_Reduce( &size, &global_size, 1, MPI_UNSIGNED, MPI_SUM, 0,
            abscissas.communicator());
        if( m_rank0)
            m_file.defput_dim_as<dg::get_value_type<ContainerType>>( name,
                global_size, atts);
        put_var( name, {abscissas}, abscissas);
    }

    ///@copydoc SerialNcFile::get_var(std::string,const NcHyperslab&,ContainerType&,ContainerType&)
    /// @note The comm in \c MPINcHyperslab must be at least a subgroup of \c communicator()
    /// @note The \c ContainerType in MPI can have either a \c dg::SharedVectorTag or \c dg::MPIVectorTag
    template<class ContainerType, typename = std::enable_if_t<
        dg::is_vector_v<ContainerType, dg::SharedVectorTag> or
        dg::is_vector_v<ContainerType, dg::MPIVectorTag>>>
    void get_var( std::string name, const MPINcHyperslab& slab,
            ContainerType& data) const
    {
        int grpid = 0, varid = 0;
        grpid = m_file.get_grpid();
        file::NC_Error_Handle err;
        if( m_readonly or m_rank0)
        {
            err = nc_inq_varid( grpid, name.c_str(), &varid);
            int ndims;
            err = nc_inq_varndims( grpid, varid, &ndims);
            assert( (unsigned)ndims == slab.ndim());
        }

        using value_type = dg::get_value_type<ContainerType>;
        auto& receive = m_receive.template get<value_type>( );
        auto& data_ref = get_ref( data, dg::get_tensor_category<ContainerType>());

        if constexpr ( dg::has_policy_v<ContainerType, dg::CudaTag>)
        {
            m_buffer.template set<value_type>( data.size());
            auto& buffer = m_buffer.template get<value_type>( );
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

    ///@copydoc SerialNcFile::get_var(std::string,const std::vector<size_t>&,T&)
    template<class T, typename = std::enable_if_t<dg::is_scalar_v<T>> >
    void get_var( std::string name, const std::vector<size_t>& start, T& data) const
    {
        if( m_readonly or m_rank0)
            m_file.get_var( name, start, data);
        if( not m_readonly)
            mpi_bcast( data);
    }

    ///@copydoc SerialNcFile::var_is_defined
    bool var_is_defined( std::string name) const
    {
        return mpi_invoke( &SerialNcFile::var_is_defined, m_file, name);
    }

    ///@copydoc SerialNcFile::get_var_type
    nc_type get_var_type(std::string name) const
    {
        return mpi_invoke( &SerialNcFile::get_var_type, m_file, name);
    }

    ///@copydoc SerialNcFile::get_var_dims
    std::vector<std::string> get_var_dims(std::string name) const
    {
        return mpi_invoke( &SerialNcFile::get_var_dims, m_file, name);
    }

    ///@copydoc SerialNcFile::get_vars
    std::vector<std::string> get_vars() const
    {
        return mpi_invoke( &SerialNcFile::get_vars, m_file);
    }

    ///@copydoc SerialNcFile::get_vars_r
    auto get_vars_r() const
    {
        return mpi_invoke( &SerialNcFile::get_vars_r, m_file);
    }


    private:
    template<class ContainerType>
    const ContainerType& get_ref( const MPI_Vector<ContainerType>& x, dg::MPIVectorTag) const
    {
        return x.data();
    }
    template<class ContainerType>
    const ContainerType& get_ref( const ContainerType& x, dg::AnyVectorTag) const
    {
        return x;
    }
    template<class ContainerType>
    ContainerType& get_ref( MPI_Vector<ContainerType>& x, dg::MPIVectorTag) const
    {
        return x.data();
    }
    template<class ContainerType>
    ContainerType& get_ref( ContainerType& x, dg::AnyVectorTag) const
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
    mutable dg::detail::AnyVector<thrust::host_vector> m_buffer, m_receive;
};

/// Convenience typedef for platform independent code
/// @ingrpup Cpp
using NcFile = MPINcFile;

}// namespace file
}// namespace dg
