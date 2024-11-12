#pragma once

#include "dg/backend/typedefs.h"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/mpi_collective.h"
#include "mpi_grid.h"
#include "projection.h"

/*! @file
  @brief Useful MPI typedefs and overloads of interpolation and projection
  */

namespace dg
{
///@cond
namespace detail{
//given global indices -> make a sorted unique indices vector + a gather map into the unique vector:
//@param buffer_idx -> (gather map/ new column indices) same size as global_idx ( can alias global_idx, index into unique_global_idx
//@param unique_global_idx -> (list of unique global indices to be used in a Collective Communication object)
static void global2bufferIdx(
    const cusp::array1d<int, cusp::host_memory>& global_idx,
    cusp::array1d<int, cusp::host_memory>& buffer_idx,
    thrust::host_vector<int>& locally_unique_global_idx)
{
    thrust::host_vector<int> index(global_idx.begin(), global_idx.end()), m_global_idx(index);
    thrust::sequence( index.begin(), index.end());
    //1. sort input global indices
    thrust::stable_sort_by_key( m_global_idx.begin(), m_global_idx.end(), index.begin());//this changes both global_idx and index
    //2. now reduce on multiple indices
    thrust::host_vector<int> ones( index.size(), 1);
    thrust::host_vector<int> unique_global( index.size()), howmany( index.size());
    typedef typename thrust::host_vector<int>::iterator iterator;
    thrust::pair<iterator, iterator> new_end;
    new_end = thrust::reduce_by_key( m_global_idx.begin(), m_global_idx.end(), ones.begin(), unique_global.begin(), howmany.begin());
    //3. copy unique indices
    locally_unique_global_idx.assign( unique_global.begin(), new_end.first);
    //4. manually make gather map into locally_unique_global_idx
    thrust::host_vector<int> gather_map;
    for( int i=0; i<(int)locally_unique_global_idx.size(); i++)
        for( int j=0; j<howmany[i]; j++)
            gather_map.push_back(i);
    assert( gather_map.size() == global_idx.size());
    //5. buffer idx is the new index
    buffer_idx.resize( global_idx.size());
    thrust::scatter( gather_map.begin(), gather_map.end(), index.begin(), buffer_idx.begin());
}
}//namespace detail
///@endcond

/**
 * @brief Convert a (row-distributed) matrix with local row and global column indices to a row distributed MPI matrix
 *
@code{.cpp}
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation(
        g_new.local(), g_old.global(), method);
    // mat is row distributed
    // mat has local row and global column indices
    auto mpi_mat = dg::convert(  mat, g_old);
@endcode
 * @tparam ConversionPolicy (can be one of the MPI %grids ) has to have the members:
 *  - <tt> bool global2localIdx(unsigned,unsigned&,unsigned&) const; </tt>
 * where the first parameter is the global index and the
 * other two are the output pair (localIdx, rank).
   return true if successful, false if global index is not part of the grid
 *  - <tt> MPI_Comm %communicator() const; </tt>  returns the communicator to use in the gather/scatter
 *  - <tt> local_size(); </tt> return the local vector size
 * @param global the local part of the matrix (different on each process) with **global column indices** and num_cols but **local row indices** and num_rows
 * @param policy the conversion object
 *
 * @return a row distributed MPI matrix. If no MPI communication is needed the collective communicator will have zero size.
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a ConversionPolicy
 * @ingroup mpi_structures
 */
template<class ConversionPolicy, class real_type>
dg::MIHMatrix_t<real_type> convert( const dg::IHMatrix_t<real_type>& global, const ConversionPolicy& policy)
{
    dg::iHVec unique_global_idx;
    cusp::array1d<int, cusp::host_memory> buffer_idx;
    dg::detail::global2bufferIdx( global.column_indices, buffer_idx, unique_global_idx);
    dg::GeneralComm<dg::iHVec, thrust::host_vector<real_type>> comm( unique_global_idx, policy);
    if( !comm.isCommunicating() )
    {
        cusp::array1d<int, cusp::host_memory> local_idx(global.column_indices), pids(local_idx);
        bool success = true;
        for(unsigned i=0; i<local_idx.size(); i++)
            success = policy.global2localIdx(global.column_indices[i], local_idx[i], pids[i]);
        assert( success);
        dg::IHMatrix_t<real_type> local( global.num_rows, policy.local_size(), global.values.size());
        // It is more memory efficient to leave the global communicator empty
        comm = dg::GeneralComm< dg::iHVec, thrust::host_vector<real_type>>();
        local.row_offsets=global.row_offsets;
        local.column_indices=local_idx;
        local.values=global.values;
        return dg::MIHMatrix_t<real_type>( local, comm, dg::row_dist);
    }
    dg::IHMatrix_t<real_type> local( global.num_rows, comm.buffer_size(), global.values.size());
    local.row_offsets=global.row_offsets;
    local.column_indices=buffer_idx;
    local.values=global.values;
    dg::MIHMatrix_t<real_type> matrix(   local, comm, dg::row_dist);
    return matrix;
}

/**
 * @brief Convert a (column-distributed) matrix with global row and column indices to a row distributed matrix
 *
 * Send all elements with a global row-index that does not belong to the calling process to the
 * process where it belongs to.
 * This can be used to convert a column distributed matrix to a row-distributed matrix as in
 * @code{.cpp}
    dg::IHMatrix_t<real_type> mat = dg::create::projection(
        g_new.global(), g_old.local(), method);
    // mat is column distributed
    // mat has global rows and local column indices
    dg::convertLocal2GlobalCols( mat, g_old);
    // now mat has global rows and global column indices
    auto mat_loc = dg::convertGlobal2LocalRows( mat, g_new);
    // now mat is row distributed with global column indices
    auto mpi_mat = dg::convert(  mat_loc, g_old);
 * @endcode
 * @tparam ConversionPolicy (can be one of the MPI %grids ) has to have the members:
 *  - <tt> bool global2localIdx(unsigned,unsigned&,unsigned&) const; </tt>
 * where the first parameter is the global index and the
 * other two are the output pair (localIdx, rank).
   return true if successful, false if global index is not part of the grid
 *  - <tt> MPI_Comm %communicator() const; </tt>  returns the communicator to use in the gather/scatter
 * @param global the column indices and num_cols need to be global, the row indices and num_rows local
 * @param policy the conversion object
 *
 * @return a row distributed MPI matrix. If no MPI communication is needed it simply has row-indices
 * converted from global to local indices.
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a ConversionPolicy
 * @ingroup mpi_structures
 */
template<class ConversionPolicy, class real_type>
dg::IHMatrix_t<real_type> convertGlobal2LocalRows( const dg::IHMatrix_t<real_type>& global, const ConversionPolicy& policy)
{
    // 0. Convert rows to coo rows
    cusp::array1d<int, cusp::host_memory> rows( global.column_indices.size()), local_rows(rows);
    for( unsigned i=0; i<global.num_rows; i++)
        for( int k = global.row_offsets[i]; k < global.row_offsets[i+1]; k++)
            rows[k] = i;
    // 1. For all rows determine pid to which it belongs
    thrust::host_vector<int> pids(rows.size());
    bool success = true;
    for(unsigned i=0; i<rows.size(); i++)
        if( !policy.global2localIdx(rows[i], local_rows[i], pids[i]) ) success = false;
    assert( success);
    // 2. Construct BijectiveComm with it
    dg::BijectiveComm<dg::iHVec, thrust::host_vector<real_type>> comm( pids, policy.communicator());
    // 3. Gather all rows/cols/vals triplets local to matrix
    auto rowsV = dg::construct<thrust::host_vector<real_type> >( local_rows);
    auto colsV = dg::construct<thrust::host_vector<real_type> >( global.column_indices);
    auto row_buffer = comm.global_gather( &rowsV[0]);
    auto col_buffer = comm.global_gather( &colsV[0]);
    auto val_buffer = comm.global_gather( &global.values[0]);
    int local_num_rows = dg::blas1::reduce( row_buffer, (real_type)0, thrust::maximum<real_type>())+1;
    cusp::coo_matrix<int, real_type, cusp::host_memory> A( local_num_rows, global.num_cols, row_buffer.size());
    A.row_indices = dg::construct<cusp::array1d<int, cusp::host_memory>>( row_buffer);
    A.column_indices = dg::construct<cusp::array1d<int, cusp::host_memory>>( col_buffer);
    A.values = val_buffer;
    A.sort_by_row_and_column();
    return dg::IHMatrix_t<real_type>(A);
    // 4. Reduce on identical rows/cols
    // .... Maybe later
}

/**
 * @brief Convert a matrix with local column indices to a matrix with global column indices
 *
 * Simply call policy.local2globalIdx for every column index
 * @code{.cpp}
    dg::IHMatrix_t<real_type> mat = dg::create::projection(
        g_new.global(), g_old.local(), method);
    // mat is column distributed
    // mat has global rows and local column indices
    dg::convertLocal2GlobalCols( mat, g_old);
    // now mat has global rows and global column indices
    auto mat_loc = dg::convertGlobal2LocalRows( mat, g_new);
    // now mat is row distributed with global column indices
    auto mpi_mat = dg::convert(  mat_loc, g_old);
 * @endcode
 * @tparam ConversionPolicy (can be one of the MPI %grids ) has to have the members:
 *  - <tt> bool local2globalIdx(unsigned,unsigned&,unsigned&) const; </tt>
 * where the first two parameters are the pair (localIdx, rank).
 * and the last one is the global index and the
   return true if successful, false if index is not part of the grid
 *  - <tt> unsigned %size() const; </tt>  returns what will become the new \c num_cols
 * @param local the column indices and num_cols need to be local, will be global on output
 * @param policy the conversion object
 *
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a ConversionPolicy
 * @ingroup mpi_structures
 */
template<class ConversionPolicy, class real_type>
void convertLocal2GlobalCols( dg::IHMatrix_t<real_type>& local, const ConversionPolicy& policy)
{
    // 1. For all rows determine pid to which it belongs
    int rank=0;
    MPI_Comm_rank( policy.communicator(), &rank);

    bool success = true;
    for(unsigned i=0; i<local.column_indices.size(); i++)
        if( !policy.local2globalIdx(local.column_indices[i], rank, local.column_indices[i]) ) success = false;
    assert( success);
    local.num_cols = policy.size();
}

namespace create
{

///@addtogroup interpolation
//
///@{
///@copydoc dg::create::interpolation(const RealGrid1d&,const RealGrid1d&,std::string)
template<class MPITopology, typename = std::enable_if_t<is_mpi_grid<MPITopology>::value >>
dg::MIHMatrix_t<typename MPITopology::value_type> interpolation( const MPITopology&
        g_new, const MPITopology& g_old, std::string method = "dg")
{
    dg::IHMatrix_t<typename MPITopology::value_type> mat = dg::create::interpolation(
        g_new.local(), g_old.global(), method);
    return convert(  mat, g_old);
}
// This is now dg::create::prolongation
/////@copydoc dg::create::interpolation(const RealGrid1d&,const RealGrid1d&,std::string)
//template<class real_type>
//dg::MIHMatrix_t<real_type> interpolation( const aRealMPITopology3d<real_type>&
//        g_new, const aRealMPITopology2d<real_type>& g_old,std::string method = "dg")
//{
//    dg::IHMatrix_t<real_type> mat = dg::create::interpolation(
//        g_new.local(), g_old.global(), method);
//    return convert(  mat, g_old);
//}

///@copydoc dg::create::projection(const RealGrid1d&,const RealGrid1d&,std::string)
template<class MPITopology, typename = std::enable_if_t<is_mpi_grid<MPITopology>::value >>
dg::MIHMatrix_t<typename MPITopology::value_type> projection( const MPITopology&
        g_new, const MPITopology& g_old, std::string method = "dg")
{
    dg::IHMatrix_t<typename MPITopology::value_type> mat = dg::create::projection(
        g_new.global(), g_old.local(), method);
    convertLocal2GlobalCols( mat, g_old);
    auto mat_loc = convertGlobal2LocalRows( mat, g_new);
    return convert(  mat_loc, g_old);
}

/**
 * @brief Create an MPI row distributed interpolation 1d matrix
 *
 * @note In the MPI version each process creates a local interpolation matrix
 * with local row and global column indices using the given points and
 * @code
 * auto mat = dg::create::interpolation ( x, g.global(), bcx, method);
 * return dg::convert( mat, g);
 * @endcode
 *
 * @copydetails interpolation(const thrust::host_vector<real_type>&,const RealGrid1d<real_type>&,dg::bc,std::string)
 */
template<class real_type>
dg::MIHMatrix_t<real_type> interpolation(
        const thrust::host_vector<real_type>& x,
        const RealMPIGrid1d<real_type>& g,
        dg::bc bcx = dg::NEU,
        std::string method = "dg")
{
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x, g.global(),
            bcx, method);
    return convert(  mat, g);
}
/**
 * @brief Create an MPI row distributed interpolation 2d matrix
 *
 * @note In the MPI version each process creates a local interpolation matrix
 * with local row and global column indices using the given points and
 * @code
 * auto mat = dg::create::interpolation ( x,y, g.global(), bcx, bcy, method);
 * return dg::convert( mat, g);
 * @endcode
 *
 * @copydetails interpolation(const thrust::host_vector<real_type>&,const thrust::host_vector<real_type>&,const aRealTopology2d<real_type>&,dg::bc,dg::bc,std::string)
 */
template<class real_type>
dg::MIHMatrix_t<real_type> interpolation(
        const thrust::host_vector<real_type>& x,
        const thrust::host_vector<real_type>& y,
        const aRealMPITopology2d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU,
        std::string method = "dg")
{
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x,y, g.global(),
            bcx, bcy, method);
    return convert(  mat, g);
}

/**
 * @brief Create an MPI row distributed interpolation 3d matrix
 *
 * @note In the MPI version each process creates a local interpolation matrix
 * with local row and global column indices using the given points
 * @code
 * auto mat = dg::create::interpolation ( x,y,z, g.global(), bcx, bcy, bcz, method);
 * return dg::convert( mat, g);
 * @endcode
 *
 * @copydetails interpolation(const thrust::host_vector<real_type>&,const thrust::host_vector<real_type>&,const thrust::host_vector<real_type>&,const aRealTopology3d<real_type>&,dg::bc,dg::bc,dg::bc,std::string)
 */
template<class real_type>
dg::MIHMatrix_t<real_type> interpolation(
        const thrust::host_vector<real_type>& x,
        const thrust::host_vector<real_type>& y,
        const thrust::host_vector<real_type>& z,
        const aRealMPITopology3d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU, dg::bc bcz = dg::PER,
        std::string method = "linear")
{
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x,y,z,
            g.global(), bcx, bcy, bcz, method);
    return convert(  mat, g);
}



///@}

}//namespace create
}//namespace dg
