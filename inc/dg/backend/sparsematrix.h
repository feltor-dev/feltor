
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

// Type of Index can be int or long
template<class Index = int, class Value = double, template<class> class Vector = thrust::host_vector>
struct SparseMatrix
{
    using policy = dg::get_execution_policy<Vector<Value>>;
    using index_type = Index;
    using value_type = Value;
    template<class OtherMatrix>
    using enable_if_serial = std::enable_if_t<std::is_same_v<typename OtherMatrix::policy, SerialTag>, OtherMatrix>;

    SparseMatrix() = default;
    // Sort is somewhat expensive and it is allowed to have unsorted column indices (at least in cusparse; for our stencil we also rely on keeping unsorted indices)
    SparseMatrix( size_t num_rows, size_t num_cols, Vector<Index> row_offsets, Vector<Index> cols, Vector<Value> vals)
    : m_num_rows( num_rows), m_num_cols( num_cols), m_row_offsets( row_offsets), m_cols(cols), m_vals(vals)
    {
    }

    template< class I, class V, template<class> class Vec>
    friend class SparseMatrix; // enable copy
    template< class I, class V, template<class> class Vec>
    SparseMatrix( const SparseMatrix<I,V,Vec>& src)
    : m_num_rows( src.m_num_rows), m_num_cols( src.m_num_cols), m_row_offsets(
        src.m_row_offsets), m_cols( src.m_cols), m_vals( src.m_vals)
    {
    }


    // May be unsorted; if duplicates exist symv may not be correct (from cusparse)
    // if sort sorts by row and column
    void setFromCoo( size_t num_rows, size_t num_cols, const Vector<Index>& rows, const Vector<Index>& cols, const Vector<Value>& vals, bool sort = false)
    {
        if( rows.size() != vals.size())
            throw dg::Error( dg::Message( _ping_) << "Error: Row indices size "
                <<rows.size()<<" not equal to values size "<<vals.size()<<"\n");
        if( cols.size() != vals.size())
            throw dg::Error( dg::Message( _ping_) << "Error: Column indices size "
                <<cols.size()<<" not equal to values size "<<vals.size()<<"\n");
        m_num_rows = num_rows;
        m_num_cols = num_cols;

        m_row_offsets.resize( num_rows+1);
        m_cols.resize( cols.size());
        m_vals.resize( vals.size());

        if( sort)
        {
            Vector<Index> trows( rows);
            Vector<Index> tcols( cols);
            Vector<Index> p( rows.size()); // permutation
            thrust::sequence( p.begin(), p.end());
            // First sort columns
            thrust::sort_by_key( tcols.begin(), tcols.end(), p.begin());
            // Repeat sort on rows
            thrust::gather( p.begin(), p.end(), rows.begin(), trows.begin());
            // Now sort rows preserving relative ordering
            thrust::stable_sort_by_key( trows.begin(), trows.end(), p.begin());
            m_row_offsets.resize( num_rows+1);
            detail::coo2csr_inline( trows, m_row_offsets);
            // Repeat sort on cols and vals
            thrust::gather( p.begin(), p.end(), cols.begin(), m_cols.begin());
            thrust::gather( p.begin(), p.end(), vals.begin(), m_vals.begin());
        }
        else
        {
            detail::coo2csr_inline( rows, m_row_offsets);
            thrust::copy( cols.begin(), cols.end(), m_cols.begin());
            thrust::copy( vals.begin(), vals.end(), m_vals.begin());
        }
        m_cache.forget();
    }

    // May be unsorted; if duplicates exist symv may not be correct (from cusparse)
    // if sort sorts by row and column
    void set( size_t num_rows, size_t num_cols, const Vector<Index>& row_offsets, const Vector<Index> cols, const Vector<Value>& vals, bool sort = false)
    {
        if ( sort)
        {
            Vector<Index> rows = detail::csr2coo( row_offsets);
            setFromCoo( num_rows, num_cols, rows, cols, vals, true);
        }
        else
        {
            m_num_rows = num_rows, m_num_cols = num_cols;
            m_row_offsets = row_offsets, m_cols = cols, m_vals = vals;
        }
        m_cache.forget();
    }

    size_t total_num_rows() const { return m_num_rows;}
    size_t total_num_cols() const { return m_num_cols;}
    size_t num_entries() const { return m_vals.size();}
    size_t num_rows() const { return m_num_rows;}
    size_t num_cols() const { return m_num_cols;}
    size_t num_vals() const { return m_vals.size();}
    const Vector<Index> & row_offsets() const { return m_row_offsets;}
    const Vector<Index> & column_indices() const { return m_cols;}
    const Vector<Value> & values() const { return m_vals;}
    /*! @brief Values can be changed without influencing the performance cache
     * @return Values array reference
     */
    Vector<Value> & values() { return m_vals;}

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
        SparseMatrix<Index,Value,thrust::host_vector> o;
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

    /*! @brief two Matrices are considered equal if elements are equal
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

    /*! @brief two Matrices are considered equal if elements are equal
     *
     * @param rhs Matrix to be compared to this
     * @return true if rhs equals this
     */
    template<class OtherMatrix = SparseMatrix>
    std::enable_if_t<dg::has_policy_v<OtherMatrix, SerialTag>, bool> operator==( const SparseMatrix& rhs) const {return !((*this != rhs));}

    /**
     * @brief subtract
     *
     * @return
     */
    template<class OtherMatrix = SparseMatrix>
    enable_if_serial<OtherMatrix> operator-() const
    {
        SparseMatrix temp(*this);
        for( unsigned i=0; i<m_vals.size(); i++)
            temp.m_vals[i] = -m_vals[i];
        return temp;
    }
    /**
     * @brief add
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
     * @brief matrix multiplication
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

    // Somewhat expensive
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

