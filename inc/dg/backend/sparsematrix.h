
#pragma once

#include <numeric>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/binary_search.h>
#include "tensor_traits.h"
#include "predicate.h"
#include "exceptions.h"
#include "config.h"
#include "blas2_stencil.h"

#include "sparsematrix_cpu.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "sparsematrix_gpu.cuh"
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include "sparsematrix_omp.h"
#endif

namespace dg
{

namespace detail
{
template<class I0, class I1>
void csr2coo_inline( const I0& csr, I1& coo)
{
    using policy = dg::get_execution_policy<I0>;
    static_assert( dg::has_policy_v<I1, policy>, "Vector types must have same execution policy");
    using I0_t = dg::get_value_type<I0>;
    using I1_t = dg::get_value_type<I1>;
    dg::blas2::detail::doParallelFor_dispatch( policy(), csr.size()-1,
            []DG_DEVICE( unsigned i, const I0_t* csr_ptr, I1_t* coo_ptr)
            {
                for (int jj = csr_ptr[i]; jj < csr_ptr[i+1]; jj++)
                    coo_ptr[jj] = i;
            },
            thrust::raw_pointer_cast(csr.data()),
            thrust::raw_pointer_cast(coo.data()));
}
template<class I>
I csr2coo( const I& csr)
{
    I coo( csr.back());
    csr2coo_inline( csr, coo);
    return coo;
}
//coo must be sorted
template<class I0, class I1>
void coo2csr_inline( const I0& coo, I1& csr)
{
    using policy = dg::get_execution_policy<I0>;
    static_assert( dg::has_policy_v<I1, policy>, "Vector types must have same execution policy");
    thrust::lower_bound( coo.begin(), coo.end(),
            thrust::counting_iterator<dg::get_value_type<I1>>(0),
            thrust::counting_iterator<dg::get_value_type<I1>>( csr.size()),
            csr.begin());
    //if( (size_t)csr[num_rows] != (size_t)coo.size())
    //    throw dg::Error( dg::Message( _ping_) << "Error: Row indices contain values beyond num_rows "
    //            <<num_rows<<"\n");
}
template<class I>
I coo2csr( unsigned num_rows, const I& coo)
{
    I csr(num_rows+1,0);
    coo2csr_inline( coo, csr);
    return csr;
}
}

/*! @brief A CSR formatted sparse matrix
 *
 * This class was designed to replace our dependency on \c cusp::csr_matrix. On
 * the host arithmetic operators like + and \* are overloaded allowing for
 * expressive code to concisely assemble the matrix:
 * @snippet{trimleft} sparsematrix_t.cpp summary
 *
 * On the device like OpenMP or GPU the only currently allowed operations are \c transpose and the
 * matrix-vector multiplications through \c dg::blas2::symv (and its aliases \c
 * dg::blas2::gemv and \c dg::apply). We use the \c cusparse library to dispatch matrix-vector
 * multiplication on the GPU.
 * @sa The CSR format is nicely explained at the
 * <a href="https://docs.nvidia.com/cuda/cusparse/#compressed-sparse-row-csr">cusparse documentation</a>
 * We use the zero-base index
 *
 * @tparam Index The index type (on GPU must be either \c int or \c long)
 * @tparam Value The value type (on GPU must be either \c float or \c double)
 * @tparam Vector The vector class to store indices and values (typically \c thrust::host_vector or \c thrust::device_vector)
 * @sa \c IHMatrix, \c IDMatrix
 * @ingroup sparsematrix
 */
template<class Index = int, class Value = double, template<class> class Vector = thrust::host_vector>
struct SparseMatrix
{
    using policy = dg::get_execution_policy<Vector<Value>>; //!< Execution policy of the matrix
    using index_type = Index; //!< The index type (on GPU must be either \c int or \c long)
    using value_type = Value; //!< The value type (on GPU must be either \c float or \c double)

    /// A static guard to allow certain members only on host (for serial execution)
    template<class OtherMatrix>
    using enable_if_serial = std::enable_if_t<std::is_same_v<typename OtherMatrix::policy, SerialTag>, OtherMatrix>;

    /*! @brief Empty matrix
     *
     * Set values using either \c setFromCoo or \c set members
     */
    SparseMatrix() = default;
    // MW: Allowing unsored indices in row is important for our stencils
    /*! @brief Directly construct from given vectors
     *
     * @snippet{trimleft} sparsematrix_t.cpp csr_ctor
     * @param num_rows Number of rows in the matrix
     * @param num_cols Number of columns in the matrix
     * @param row_offsets Vector of size <tt>num_rows+1</tt> representing the
     * starting position of each row in the column_indices and values arrays.
     * <tt>row_offsets.back()</tt> equals the number of non-zero elements in
     * the matrix
     * @param column_indices Vector of size <tt>row_offsets.back()</tt> containing column indices.
     * The column indices may be unsorted within each row and should be unique.
     * If duplicates exist symv may not be correct (from cusparse).
     * @param values Vector of size <tt>row_offsets.back()</tt> containing nonzero values
     */
    SparseMatrix( size_t num_rows, size_t num_cols, const Vector<Index>&
        row_offsets, const Vector<Index>& column_indices, const Vector<Value>& values)
    : m_num_rows( num_rows), m_num_cols( num_cols), m_row_offsets(
    row_offsets), m_cols(column_indices), m_vals(values)
    {
    }

    /// Enable copy constructor from foreign types
    template< class I, class V, template<class> class Vec>
    friend struct SparseMatrix; // enable copy

    /*! @brief Copy construct from differen type
     *
     * Typically used to transfer a matrix from host to device
     * @snippet{trimleft} sparsematrix_t.cpp host2device
     * @param src Copy the matrix
     */
    template< class I, class V, template<class> class Vec>
    SparseMatrix( const SparseMatrix<I,V,Vec>& src)
    : m_num_rows( src.m_num_rows), m_num_cols( src.m_num_cols), m_row_offsets(
        src.m_row_offsets), m_cols( src.m_cols), m_vals( src.m_vals)
    {
    }


    /*! @brief Set csr values from coo formatted sparse matrix
     *
     * @snippet{trimleft} sparsematrix_t.cpp setFromCoo
     * @note May be unsorted, in which case the \c sort parameter should be set
     * to \c true; if duplicates exist symv may not be correct (from cusparse)
     *
     * @param num_rows Number of rows in the matrix
     * @param num_cols Number of columns in the matrix
     * @param row_indices Contains row indices of the corresponding elements in
     * the values vector
     * @param column_indices Vector of size <tt>row_indices.size()()</tt>
     * containing column indices.
     * @param values Vector of size <tt>row_indices.size()</tt> containing
     * nonzero values
     * @param sort If \c true the given vectors are sorted by row and column
     * before converted to the csr format. Per default we assume that the values
     * are sorted by row (not necessarily by column). In this case there is no
     * need to sort the indices, which may be somewhat expensive. If the row
     * indices are unsorted however, \c sort must be set to \c true  otherwise
     * the coo2csr conversion will fail.
    */
    void setFromCoo( size_t num_rows, size_t num_cols, const Vector<Index>& row_indices, const Vector<Index>& column_indices, const Vector<Value>& values, bool sort = false)
    {
        if( row_indices.size() != values.size())
            throw dg::Error( dg::Message( _ping_) << "Error: Row indices size "
                <<row_indices.size()<<" not equal to values size "<<values.size()<<"\n");
        if( column_indices.size() != values.size())
            throw dg::Error( dg::Message( _ping_) << "Error: Column indices size "
                <<column_indices.size()<<" not equal to values size "<<values.size()<<"\n");
        m_num_rows = num_rows;
        m_num_cols = num_cols;

        m_row_offsets.resize( num_rows+1);
        m_cols.resize( column_indices.size());
        m_vals.resize( values.size());

        if( sort)
        {
            Vector<Index> trows( row_indices);
            Vector<Index> tcols( column_indices);
            Vector<Index> p( row_indices.size()); // permutation
            thrust::sequence( p.begin(), p.end());
            // First sort columns
            thrust::sort_by_key( tcols.begin(), tcols.end(), p.begin());
            // Repeat sort on rows
            thrust::gather( p.begin(), p.end(), row_indices.begin(), trows.begin());
            // Now sort rows preserving relative ordering
            thrust::stable_sort_by_key( trows.begin(), trows.end(), p.begin());
            m_row_offsets.resize( num_rows+1);
            detail::coo2csr_inline( trows, m_row_offsets);
            // Repeat sort on cols and vals
            thrust::gather( p.begin(), p.end(), column_indices.begin(), m_cols.begin());
            thrust::gather( p.begin(), p.end(), values.begin(), m_vals.begin());
        }
        else
        {
            detail::coo2csr_inline( row_indices, m_row_offsets);
            thrust::copy( column_indices.begin(), column_indices.end(), m_cols.begin());
            thrust::copy( values.begin(), values.end(), m_vals.begin());
        }
        m_cache.forget();
    }

    /*! @brief Set csr values directly
     *
     * @snippet{trimleft} sparsematrix_t.cpp set
     *
     * @param num_rows Number of rows in the matrix
     * @param num_cols Number of columns in the matrix
     * @param row_offsets Vector of size <tt>num_rows+1</tt> representing the
     * starting position of each row in the column_indices and values arrays.
     * <tt>row_offsets.back()</tt> equals the number of non-zero elements in
     * the matrix
     * @param column_indices Vector of size <tt>row_offsets.back()</tt> containing column indices.
     * The column indices may be unsorted within each row and should be unique.
     * If duplicates exist symv may not be correct (from cusparse).
     * @param values Vector of size <tt>row_offsets.back()</tt> containing nonzero values
     * @param sort If \c true the given vectors are sorted by column_indices in each row.
     * It is allowed to have unsorted column indices in each row so sorting is not strictly
     * necessary. It may (or may not) have an effect on performance.
    */
    void set( size_t num_rows, size_t num_cols, const Vector<Index>& row_offsets, const Vector<Index> column_indices, const Vector<Value>& values, bool sort = false)
    {
        if ( sort)
        {
            Vector<Index> rows = detail::csr2coo( row_offsets);
            setFromCoo( num_rows, num_cols, rows, column_indices, values, true);
        }
        else
        {
            m_num_rows = num_rows, m_num_cols = num_cols;
            m_row_offsets = row_offsets, m_cols = column_indices, m_vals = values;
        }
        m_cache.forget();
    }

    /*! @brief Sort by row and column
     *
     * Sort is somewhat expensive and it is allowed to have unsorted column
     * indices (at least in cusparse). Sorting may or may not have an influence
     * on performance. Never sort a stencil for \c dg::blas2::stencil.
     */
    void sort_indices()
    {
        if( m_num_rows != m_row_offsets.size()-1)
            throw dg::Error( dg::Message( _ping_) << "Error: Row offsets have size "
                <<m_row_offsets.size()<<" but num_rows is "<<m_num_rows<<"\n");
        if( m_cols.size() != m_vals.size())
            throw dg::Error( dg::Message( _ping_) << "Error: Column indices size "
                <<m_cols.size()<<" not equal to values size "<<m_vals.size()<<"\n");

        // Sort each row
        Vector<Index> rows = detail::csr2coo( m_row_offsets);
        Vector<Index> cols = m_cols;
        Vector<Value> vals = m_vals;
        setFromCoo( m_num_rows, m_num_cols, rows, cols, vals, true);
    }

    /// Alias for \c num_rows
    size_t total_num_rows() const { return m_num_rows;} // for blas2_sparseblockmat dispatch
    /// Alias for \c num_cols
    size_t total_num_cols() const { return m_num_cols;} // for blas2_sparseblockmat dispatch

    /// Number of rows in matrix
    size_t num_rows() const { return m_num_rows;}
    /// Number of columns in matrix
    size_t num_cols() const { return m_num_cols;}
    /// Alias for \c num_nnz
    size_t num_vals() const { return m_vals.size();}
    /// Number of nonzero elements in matrix
    size_t num_nnz() const { return m_vals.size();}
    /// Alias for \c num_nnz
    size_t num_entries() const { return m_vals.size();}

    /*! @brief Read row_offsets vector
     * @return row_offsets
     */
    const Vector<Index> & row_offsets() const { return m_row_offsets;}
    /*! @brief Read column indices vector
     * @return column indices
     */
    const Vector<Index> & column_indices() const { return m_cols;}
    /*! @brief Read values vector
     * @return values
     */
    const Vector<Value> & values() const { return m_vals;}
    /*! @brief Change values vector
     * @note The reason why \c values can be changed directly while \c
     * row_offsets and \c column_indices cannot,  is that changing the values
     * does not influence the performance cache, while changing the sparsity
     * pattern through \c row_offsets and \c column_indices does
     * @return Values array reference
     */
    Vector<Value> & values() { return m_vals;}

    ///@cond
    template<class value_type>
    void symv(SharedVectorTag, SerialTag, value_type alpha, const value_type* RESTRICT x, value_type beta, value_type* RESTRICT y) const
    {
        const auto* row_ptr = thrust::raw_pointer_cast( &m_row_offsets[0]);
        const auto* col_ptr = thrust::raw_pointer_cast( &m_cols[0]);
        const auto* val_ptr = thrust::raw_pointer_cast( &m_vals[0]);
        detail::spmv_cpu_kernel( m_cache, m_num_rows, m_num_cols,
            m_vals.size(), row_ptr, col_ptr, val_ptr, alpha, beta, x, y);
    }
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    template<class value_type>
    void symv(SharedVectorTag, CudaTag, value_type alpha, const value_type* x, value_type beta, value_type* y) const
    {
        const auto* row_ptr = thrust::raw_pointer_cast( &m_row_offsets[0]);
        const auto* col_ptr = thrust::raw_pointer_cast( &m_cols[0]);
        const auto* val_ptr = thrust::raw_pointer_cast( &m_vals[0]);
        detail::spmv_gpu_kernel( m_cache, m_num_rows, m_num_cols,
            m_vals.size(), row_ptr, col_ptr, val_ptr, alpha, beta, x, y);
    }
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
    template<class value_type>
    void symv(SharedVectorTag, OmpTag, value_type alpha, const value_type* x, value_type beta, value_type* y) const
    {
        const auto* row_ptr = thrust::raw_pointer_cast( &m_row_offsets[0]);
        const auto* col_ptr = thrust::raw_pointer_cast( &m_cols[0]);
        const auto* val_ptr = thrust::raw_pointer_cast( &m_vals[0]);
        if( !omp_in_parallel())
        {
            #pragma omp parallel
            {
                detail::spmv_omp_kernel( m_cache, m_num_rows, m_num_cols,
                    m_vals.size(), row_ptr, col_ptr, val_ptr, alpha, beta, x, y);
            }
            return;
        }
        detail::spmv_omp_kernel( m_cache, m_num_rows, m_num_cols,
            m_vals.size(), row_ptr, col_ptr, val_ptr, alpha, beta, x, y);
    }
#endif
    ///@endcond

    /**
    * @brief Transposition
    *
    * The transpose is sorted even if the original is not
    * @return  A newly generated SparseMatrix containing the transpose.
    */
    SparseMatrix transpose() const
    {
        auto cols = detail::csr2coo(m_row_offsets);
        // cols are sorted

        // We need to sort now
        SparseMatrix<Index,Value,Vector> o;
        o.m_num_rows = m_num_cols;
        o.m_num_cols = m_num_rows;

        Vector<Index> rows( m_cols.begin(), m_cols.end());
        Vector<Index> p( rows.size()); // permutation
        thrust::sequence( p.begin(), p.end());

        thrust::stable_sort_by_key( rows.begin(), rows.end(), p.begin());
        o.m_row_offsets.resize( o.m_num_rows+1);
        detail::coo2csr_inline( rows, o.m_row_offsets);
        // Repeat sort on cols and vals
        o.m_cols.resize( cols.size());
        o.m_vals.resize( m_vals.size());
        // Since cols are sorted o.m_cols will be as well
        thrust::gather( p.begin(), p.end(),   cols.begin(), o.m_cols.begin());
        thrust::gather( p.begin(), p.end(), m_vals.begin(), o.m_vals.begin());
        o.m_cache.forget();
        return o;

    }

    //
    // We enable the following only for serial sparse matrices

    /*! @brief Two Matrices are considered equal if elements and sparsity pattern are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs does not equal this
     */
    template<class OtherMatrix = SparseMatrix>
    std::enable_if_t<dg::has_policy_v<OtherMatrix, SerialTag>, bool> operator!=( const SparseMatrix& rhs) const{
        if( rhs.m_num_rows != m_num_rows)
            return true;
        if( rhs.m_num_cols != m_num_cols)
            return true;
        for( size_t i = 0; i < m_row_offsets.size(); i++)
            if( m_row_offsets[i] != rhs.m_row_offsets[i])
                return true;
        for( size_t i = 0; i < m_cols.size(); i++)
            if( m_cols[i] != rhs.m_cols[i])
                return true;
        for( size_t i = 0; i < m_vals.size(); i++)
            if( m_vals[i] != rhs.m_vals[i])
                return true;
        return false;
    }

    /*! @brief Two Matrices are considered equal if elements and sparsity pattern are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs equals this
     */
    template<class OtherMatrix = SparseMatrix>
    std::enable_if_t<dg::has_policy_v<OtherMatrix, SerialTag>, bool> operator==( const SparseMatrix& rhs) const {return !((*this != rhs));}

    /**
     * @brief Negate
     *
     * @return
     */
    SparseMatrix operator-() const
    {
        return (*this)*(Value(-1));
    }
    /**
     * @brief Add
     *
     * @param op
     *
     * @return
     */
    template<class OtherMatrix = SparseMatrix>
    enable_if_serial<OtherMatrix>& operator+=( const SparseMatrix& op)
    {
        SparseMatrix tmp = *this + op;
        swap( tmp, *this);
    }
    /**
     * @brief subtract
     *
     * @param op
     *
     * @return
     */
    template<class OtherMatrix = SparseMatrix>
    enable_if_serial<OtherMatrix>& operator-=( const SparseMatrix& op)
    {
        SparseMatrix tmp = *this + (-op);
        swap( tmp, *this);
    }
    /**
     * @brief scalar multiply
     *
     * @param value
     *
     * @return
     */
    SparseMatrix& operator*=( const Value& value )
    {
        dg::blas2::detail::doParallelFor_dispatch( policy(), m_vals.size(),
            [value]DG_DEVICE( unsigned i, Value* val_ptr)
            {
                val_ptr[i] *= value;
            },
            thrust::raw_pointer_cast(m_vals.data()));
        return *this;
    }
    /**
     * @brief add
     *
     * @param lhs
     * @param rhs
     *
     * @return addition (sorted even if lhs and rhs are not)
     */
    template<class OtherMatrix = SparseMatrix>
    friend enable_if_serial<OtherMatrix> operator+( const SparseMatrix& lhs, const SparseMatrix& rhs)
    {
        if( lhs.m_num_cols != rhs.m_num_cols)
            throw dg::Error( dg::Message( _ping_) << "Error: cannot add matrix with "
                <<lhs.m_num_cols<<" columns to matrix with "<<rhs.m_num_cols<<" columns\n");
        if( lhs.m_num_rows != rhs.m_num_rows)
            throw dg::Error( dg::Message( _ping_) << "Error: cannot add matrix with "
                <<lhs.m_num_rows<<" rows to matrix with "<<rhs.m_num_rows<<" rows\n");
        Vector<Index> row_offsets, cols;
        Vector<Value> vals;

        detail::spadd_cpu_kernel( lhs.m_num_rows, lhs.m_num_cols,
            lhs.m_row_offsets, lhs.m_cols, lhs.m_vals,
            rhs.m_row_offsets, rhs.m_cols, rhs.m_vals,
            row_offsets, cols, vals);

        SparseMatrix temp(lhs.m_num_rows, rhs.m_num_cols, row_offsets, cols, vals);
        return temp;
    }
    /**
     * @brief subtract
     *
     * @param lhs
     * @param rhs
     *
     * @return subtraction (sorted even if lhs and rhs are not)
     */
    template<class OtherMatrix = SparseMatrix>
    friend enable_if_serial<OtherMatrix> operator-( const SparseMatrix& lhs, const SparseMatrix& rhs)
    {
        SparseMatrix temp(lhs);
        temp-=rhs;
        return temp;
    }
    /**
     * @brief scalar multiplication
     *
     * Simply multiply values array by given value
     * @param value
     * @param rhs
     *
     * @return
     */
    template<class OtherMatrix = SparseMatrix>
    friend enable_if_serial<OtherMatrix> operator*( const Value& value, const SparseMatrix& rhs )
    {
        SparseMatrix temp(rhs);
        temp*=value;
        return temp;
    }

    /**
     * @brief scalar multiplication
     *
     * Simply multiply values array by given value
     * @param lhs
     * @param value
     *
     * @return
     */
    template<class OtherMatrix = SparseMatrix>
    friend enable_if_serial<OtherMatrix> operator*( const SparseMatrix& lhs, const Value& value)
    {
        return  value*lhs;
    }

    /**
     * @brief matrix-matrix multiplication \f$ C = A*B\f$
     *
     * @param lhs
     * @param rhs
     *
     * @return multiplication (sorted even if lhs and rhs are not)
     */
    template<class OtherMatrix = SparseMatrix>
    friend enable_if_serial<OtherMatrix> operator*( const SparseMatrix& lhs, const SparseMatrix& rhs)
    {
        if( lhs.m_num_cols != rhs.m_num_rows)
            throw dg::Error( dg::Message( _ping_) << "Error: cannot multiply matrix with "
                <<lhs.m_num_cols<<" columns with matrix with "<<rhs.m_num_rows<<" rows\n");

        Vector<Index> row_offsets, cols;
        Vector<Value> vals;

        detail::spgemm_cpu_kernel( lhs.m_num_rows, lhs.m_num_cols, rhs.m_num_cols,
            lhs.m_row_offsets, lhs.m_cols, lhs.m_vals,
            rhs.m_row_offsets, rhs.m_cols, rhs.m_vals,
            row_offsets, cols, vals);

        SparseMatrix temp(lhs.m_num_rows, rhs.m_num_cols, row_offsets, cols, vals);
        return temp;
    }

    /**
     * @brief matrix-vector multiplication  \f$  y = S x\f$
     *
     * Works if ContainerType has the same execution policy as the SparseMatrix
     * @snippet{trimleft} operator_t.cpp matvec
     * @param S Matrix
     * @param x Vector
     *
     * @return Vector
     */
    template<class ContainerType, class = std::enable_if_t < dg::has_policy_v<ContainerType, policy> and dg::is_vector_v<ContainerType> >>
    friend ContainerType operator*( const SparseMatrix& S, const ContainerType& x)
    {
        if( S.m_num_cols != x.size())
            throw dg::Error( dg::Message( _ping_) << "Error: cannot multiply matrix with "
                <<S.m_num_cols<<" columns with vector with "<<x.size()<<" rows\n");

        ContainerType out(S.m_num_rows);
        const Value* RESTRICT x_ptr = thrust::raw_pointer_cast( x.data());
        Value* RESTRICT y_ptr = thrust::raw_pointer_cast( out.data());
        S.symv( SharedVectorTag(), policy(), Value(1), x_ptr, Value(0), y_ptr);
        return out;
    }

    /*! @brief puts a matrix linewise in output stream
     *
     * @tparam Ostream The stream e.g. std::cout
     * @param os the outstream
     * @param mat the matrix to output
     * @return the outstream
     */
    template< class Ostream, class OtherMatrix = SparseMatrix>
    friend std::enable_if_t<dg::has_policy_v<OtherMatrix, SerialTag>, Ostream>
        & operator<<(Ostream& os, const SparseMatrix& mat)
    {
        os << "Sparse Matrix with "<<mat.m_num_rows<<" rows and "<<mat.m_num_cols<<" columns\n";
        os << " # non-zeroes "<<mat.m_vals.size()<<"\n";
        for (int i = 0; i < (int)mat.m_num_rows; i++)
        {
            for (int pB = mat.m_row_offsets[i]; pB < mat.m_row_offsets[i+1]; pB++)
            {
                os << "("<<i<<","<<mat.m_cols[pB]<<","<<mat.m_vals[pB]<<") ";
            }
            os << "\n";
        }
        return os;
    }

    private:
    // The task of the cache is to keep performance information across multiple calls to
    // symv. The cache needs to forget said information any time the matrix changes
    mutable std::conditional_t< std::is_same_v<policy, SerialTag>, detail::CSRCache_cpu,
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
            detail::CSRCache_gpu>
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
            detail::CSRCache_omp>
#else
            detail::CSRCache_cpu>
#endif
            m_cache;

    size_t m_num_rows, m_num_cols;
    Vector<Index> m_row_offsets, m_cols;
    Vector<Value> m_vals;
};

///@addtogroup traits
///@{
template <class I, class T, template<class> class V>
struct TensorTraits<SparseMatrix<I, T, V> >
{
    using value_type  = T;
    using tensor_category = SparseMatrixTag;
    using execution_policy = typename SparseMatrix<I,T,V>::policy;
};
///@}
}//namespace dg

