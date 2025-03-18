#pragma once

#include "dg/backend/typedefs.h"
#include "dg/backend/index.h"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/mpi_gather.h"
#include "mpi_grid.h"
#include "projection.h"

/*! @file
  @brief Useful MPI typedefs and overloads of interpolation and projection
  */

namespace dg
{

/**
 * @brief Convert a (row-distributed) matrix with local row and global column
 * indices to a row distributed MPI matrix
 *
@code{.cpp}
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation(
        g_new.local(), g_old.global(), method);
    // mat is row distributed
    // mat has local row and global column indices
    auto mpi_mat = dg::make_mpi_matrix(  mat, g_old);
@endcode
 * @tparam ConversionPolicy (can be one of the MPI %grids ) has to have the
 * members:
 *  - <tt>bool global2localIdx(unsigned,unsigned&,unsigned&) const;</tt> where
 *  the first parameter is the global index and the other two are the output
 *  pair (localIdx, rank).
   return true if successful, false if global index is not part of the grid
 *  - <tt>MPI_Comm %communicator() const;</tt>  returns the communicator to use
 *  in the gather/scatter
 *  - <tt>unsigned local_size() const;</tt> return the local vector size
 *  .
 * @param global_cols the local part of the matrix (different on each process) with
 * **global column indices** and num_cols but **local row indices** and
 * num_rows
 * @param col_policy the conversion object
 *
 * @return a row distributed MPI matrix. If no MPI communication is needed the
 * collective communicator will have zero size.
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a
 * ConversionPolicy
 * @ingroup mpi_matvec
 */
template<class ConversionPolicy, class real_type>
dg::MIHMatrix_t<real_type> make_mpi_matrix(
    const dg::IHMatrix_t<real_type>& global_cols, const ConversionPolicy& col_policy)
{
    // Our approach here is:
    // 1. Convert to coo format
    // 2. mark rows in global_cols that are communicating
    // we need to grab the entire row to ensure reproducibility (which is guaranteed by order of computations)
    // 3. Convert inner rows to local and move outer rows to vectors (invalidating them in inner)
    // 4. Use global columns vector to construct MPIGather
    // 5. Use buffer index to construct local CooMat
    int rank;
    MPI_Comm_rank( col_policy.communicator(), &rank);

    const dg::IHMatrix_t<real_type>& A = global_cols;
    std::vector<int> outer_row( A.num_rows, 0);
    // 1st pass determine local rows
    for(int i = 0; i < (int)A.num_rows; i++)
    for (int jj = A.row_offsets[i]; jj < A.row_offsets[i+1]; jj++)
    {
        int lIdx=0, pid = 0;
        assert( col_policy.global2localIdx( A.column_indices[jj], lIdx, pid));
        if( pid != rank)
            outer_row[i] = 1;
    }

    // 2nd pass: distribute entries
    thrust::host_vector<std::array<int,2>> outer_col;
    cusp::array1d<int, cusp::host_memory> inner_row, inner_col, buffer_row;
    cusp::array1d<real_type, cusp::host_memory> inner_val, outer_val;
    int buffer_row_idx = 0;
    thrust::host_vector<int> row_scatter;
    for(int i = 0; i < (int)A.num_rows; i++)
    {
        for (int jj = A.row_offsets[i]; jj < A.row_offsets[i+1]; jj++)
        {
            int lIdx=0, pid = 0;
            col_policy.global2localIdx( A.column_indices[jj], lIdx, pid);
            if( outer_row[i] == 1)
            {
                buffer_row.push_back( buffer_row_idx);
                outer_col.push_back( {pid,lIdx});
                outer_val.push_back( A.values[jj]);
            }
            else
            {
                inner_row.push_back( i);
                inner_col.push_back( lIdx);
                inner_val.push_back( A.values[jj]);
            }
        }
        if( outer_row[i] == 1)
        {
            buffer_row_idx++;
            row_scatter.push_back(i);
        }
    }
    // 3. Now make MPI Gather object
    thrust::host_vector<int> lColIdx;
    auto gather_map = dg::gIdx2unique_idx( outer_col, lColIdx);
    MPIGather<thrust::host_vector> mpi_gather( gather_map,
            col_policy.communicator());
    cusp::coo_matrix<int, real_type, cusp::host_memory> inner( A.num_rows,
            col_policy.local_size(), inner_val.size());
    inner.row_indices    = inner_row;
    inner.column_indices = inner_col;
    inner.values         = inner_val;
    if( inner.row_indices.size() > 0)
        inner.sort_by_row_and_column();
    cusp::coo_matrix<int, real_type, cusp::host_memory> outer( buffer_row_idx,
            mpi_gather.buffer_size(), outer_val.size());
    outer.row_indices    = buffer_row;
    outer.column_indices = lColIdx;
    outer.values         = outer_val;
    if( outer.row_indices.size() > 0)
        outer.sort_by_row_and_column();
    return { inner, outer, mpi_gather, row_scatter}; // implicitly convert
}


/**
 * @brief Convert a (column-distributed) matrix with global row and column
 * indices to a row distributed matrix
 *
 * Send all elements with a global row-index that does not belong to the
 * calling process to the process where it belongs to.  This can be used to
 * convert a column distributed matrix to a row-distributed matrix as in
 * @code{.cpp}
    dg::IHMatrix_t<real_type> mat = dg::create::projection(
        g_new.global(), g_old.local(), method);
    // mat is column distributed
    // mat has global rows and local column indices
    dg::convertLocal2GlobalCols( mat, g_old);
    // now mat has global rows and global column indices
    auto mat_loc = dg::convertGlobal2LocalRows( mat, g_new);
    // now mat is row distributed with global column indices
    auto mpi_mat = dg::make_mpi_matrix(  mat_loc, g_old);
 * @endcode
 * @tparam ConversionPolicy (can be one of the MPI %grids ) has to have the
 * members:
 *  - <tt> bool global2localIdx(unsigned,unsigned&,unsigned&) const; </tt>
 * where the first parameter is the global index and the other two are the
 * output pair (localIdx, rank).  return true if successful, false if global
 * index is not part of the grid
 *  - <tt> MPI_Comm %communicator() const; </tt>  returns the communicator to
 *  use in the gather/scatter
 *  - <tt> local_size(); </tt> return the local vector size
 * @param global the row indices and num_rows need to be global
 * @param row_policy the conversion object
 *
 * @return a row distributed MPI matrix. If no MPI communication is needed it
 * simply has row-indices converted from global to local indices. \c num_cols
 * is the one from \c global
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a
 * ConversionPolicy;
 * @ingroup mpi_matvec
 */
template<class ConversionPolicy, class real_type>
dg::IHMatrix_t<real_type> convertGlobal2LocalRows( const
    dg::IHMatrix_t<real_type>& global, const ConversionPolicy& row_policy)
{
    // 0. Convert to coo matrix
    cusp::coo_matrix<int, real_type, cusp::host_memory> A = global;
    // 1. For all rows determine pid to which it belongs
    auto gIdx = dg::gIdx2gIdx( A.row_indices, row_policy);
    std::map<int, cusp::array1d<int, cusp::host_memory>> rows, cols;
    std::map<int, cusp::array1d<real_type, cusp::host_memory>> vals;
    for( unsigned u=0; u<gIdx.size(); u++)
    {
        rows[gIdx[u][0]].push_back( gIdx[u][1]);
        cols[gIdx[u][0]].push_back( A.column_indices[u]);
        vals[gIdx[u][0]].push_back( A.values[u]);
    }
    // 2. Now send those rows to where they belong
    auto row_buf = dg::mpi_permute( rows, row_policy.communicator());
    auto col_buf = dg::mpi_permute( cols, row_policy.communicator());
    auto val_buf = dg::mpi_permute( vals, row_policy.communicator());

    cusp::coo_matrix<int, real_type, cusp::host_memory> B(
        row_policy.local_size(), global.num_cols,
        dg::detail::flatten_map(row_buf).size());
    B.row_indices    = dg::detail::flatten_map(row_buf);
    B.column_indices = dg::detail::flatten_map(col_buf);
    B.values         = dg::detail::flatten_map(val_buf);
    if( B.row_indices.size() > 0) // BugFix
        B.sort_by_row_and_column();
    return dg::IHMatrix_t<real_type>(B);
    // 4. Reduce on identical rows/cols
    // .... Maybe later
}

/**
 * @brief Convert a matrix with local column indices to a matrix with global
 * column indices
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
    auto mpi_mat = dg::make_mpi_matrix(  mat_loc, g_old);
 * @endcode
 * @tparam ConversionPolicy (can be one of the MPI %grids ) has to have the
 * members:
 *  - <tt>bool local2globalIdx(unsigned,unsigned&,unsigned&) const;</tt>
 * where the first two parameters are the pair (localIdx, rank).
 * and the last one is the global index and the
   return true if successful, false if index is not part of the grid
 *  - <tt>unsigned %size() const;</tt>  returns what will become the new \c
 *  num_cols
 * @param local the column indices and num_cols need to be local, will be
 * global on output
 * @param policy the conversion object
 *
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a
 * ConversionPolicy
 * @ingroup mpi_matvec
 */
template<class ConversionPolicy, class real_type>
void convertLocal2GlobalCols( dg::IHMatrix_t<real_type>& local, const ConversionPolicy& policy)
{
    // 1. For all columns determine pid to which it belongs
    int rank=0;
    MPI_Comm_rank( policy.communicator(), &rank);

    for(unsigned i=0; i<local.column_indices.size(); i++)
        assert( policy.local2globalIdx(local.column_indices[i], rank, local.column_indices[i]) );
    local.num_cols = policy.size();
}

namespace create
{

///@addtogroup interpolation
//
///@{
///@copydoc interpolation(const aRealTopology<real_type,Nd>&,const aRealTopology<real_type,Nd>&,std::string)
template<class MPITopology, typename = std::enable_if_t<dg::is_vector_v<
    typename MPITopology::host_vector, MPIVectorTag>>>
dg::MIHMatrix_t<typename MPITopology::value_type> interpolation( const MPITopology&
        g_new, const MPITopology& g_old, std::string method = "dg")
{
    dg::IHMatrix_t<typename MPITopology::value_type> mat = dg::create::interpolation(
        g_new.local(), g_old.global(), method);
    return make_mpi_matrix(  mat, g_old);
}
// This is now dg::create::prolongation
/////@copydoc dg::create::interpolation(const RealGrid1d&,const RealGrid1d&,std::string)
//template<class real_type>
//dg::MIHMatrix_t<real_type> interpolation( const aRealMPITopology3d<real_type>&
//        g_new, const aRealMPITopology2d<real_type>& g_old,std::string method = "dg")
//{
//    dg::IHMatrix_t<real_type> mat = dg::create::interpolation(
//        g_new.local(), g_old.global(), method);
//    return make_mpi_matrix(  mat, g_old);
//}

///@copydoc projection(const aRealTopology<real_type,Nd>&,const aRealTopology<real_type,Nd>&,std::string)
template<class MPITopology, typename = std::enable_if_t<dg::is_vector_v<
    typename MPITopology::host_vector, MPIVectorTag>>>
dg::MIHMatrix_t<typename MPITopology::value_type> projection( const MPITopology&
        g_new, const MPITopology& g_old, std::string method = "dg")
{
    dg::IHMatrix_t<typename MPITopology::value_type> mat = dg::create::projection(
        g_new.global(), g_old.local(), method);
    convertLocal2GlobalCols( mat, g_old);
    auto mat_loc = convertGlobal2LocalRows( mat, g_new);
    return make_mpi_matrix(  mat_loc, g_old);
}

///@copydoc interpolation(const RecursiveHostVector&,const aRealTopology<real_type,Nd>&,std::array<dg::bc,Nd>,std::string)
template<class RecursiveHostVector, class real_type, size_t Nd>
dg::MIHMatrix_t<real_type> interpolation(
        const RecursiveHostVector& x,
        const aRealMPITopology<real_type, Nd>& g,
        std::array<dg::bc, Nd> bcx,
        std::string method = "dg")
{
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x, g.global(),
            bcx, method);
    return make_mpi_matrix(  mat, g);
}

/**
 * @brief Create an MPI row distributed interpolation 1d matrix
 *
 * @note In the MPI version each process creates a local interpolation matrix
 * with local row and global column indices using the given points and
 * @code
 * auto mat = dg::create::interpolation ( x, g.global(), bcx, method);
 * return dg::make_mpi_matrix( mat, g);
 * @endcode
 *
 * @copydetails interpolation(const host_vector&,const RealGrid1d<real_type>&,dg::bc,std::string)
 */
template<class host_vector, class real_type>
dg::MIHMatrix_t<real_type> interpolation(
        const host_vector& x,
        const RealMPIGrid1d<real_type>& g,
        dg::bc bcx = dg::NEU,
        std::string method = "dg")
{
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x, g.global(),
            bcx, method);
    return make_mpi_matrix(  mat, g);
}
/**
 * @brief Create an MPI row distributed interpolation 2d matrix
 *
 * @note In the MPI version each process creates a local interpolation matrix
 * with local row and global column indices using the given points and
 * @code
 * auto mat = dg::create::interpolation ( x,y, g.global(), bcx, bcy, method);
 * return dg::make_mpi_matrix( mat, g);
 * @endcode
 *
 * @copydetails interpolation(const host_vector&,const host_vector&,const aRealTopology2d<real_type>&,dg::bc,dg::bc,std::string)
 */
template<class host_vector, class real_type>
dg::MIHMatrix_t<real_type> interpolation(
        const host_vector& x,
        const host_vector& y,
        const aRealMPITopology2d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU,
        std::string method = "dg")
{
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x,y, g.global(),
            bcx, bcy, method);
    return make_mpi_matrix(  mat, g);
}

/**
 * @brief Create an MPI row distributed interpolation 3d matrix
 *
 * @note In the MPI version each process creates a local interpolation matrix
 * with local row and global column indices using the given points
 * @code
 * auto mat = dg::create::interpolation ( x,y,z, g.global(), bcx, bcy, bcz, method);
 * return dg::make_mpi_matrix( mat, g);
 * @endcode
 *
 * @copydetails interpolation(const host_vector&,const host_vector&,const host_vector&,const aRealTopology3d<real_type>&,dg::bc,dg::bc,dg::bc,std::string)
 */
template<class host_vector, class real_type>
dg::MIHMatrix_t<real_type> interpolation(
        const host_vector& x,
        const host_vector& y,
        const host_vector& z,
        const aRealMPITopology3d<real_type>& g,
        dg::bc bcx = dg::NEU, dg::bc bcy = dg::NEU, dg::bc bcz = dg::PER,
        std::string method = "dg")
{
    dg::IHMatrix_t<real_type> mat = dg::create::interpolation( x,y,z,
            g.global(), bcx, bcy, bcz, method);
    return make_mpi_matrix(  mat, g);
}



///@}

}//namespace create
}//namespace dg
