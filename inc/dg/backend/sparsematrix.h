
#pragma once

#include <numeric>
#include <thrust/sort.h>
#include "tensor_traits.h"
#include "predicate.h"
#include "exceptions.h"
#include "config.h"

#include "sparsematrix_cpu.h"
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
#include "sparsematrix_gpu.cuh"
#elif THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_OMP
#include "sparsematrix_omp.h"
#endif

namespace dg
{

template<class I>
I csr2coo( const I& csr)
{
    static_assert( dg::has_policy_v<I, dg::SerialTag>);
    unsigned num_rows = csr.size()-1;
    I coo( csr[num_rows]);
    for( unsigned i=0; i<num_rows; i++)
        for (int jj = csr[i]; jj < csr[i+1]; jj++)
            coo[jj] = i;
    return coo;
}
//coo must be sorted
template<class I>
I coo2csr( unsigned num_rows, const I& coo)
{
    static_assert( dg::has_policy_v<I, dg::SerialTag>);
    I csr(num_rows+1,0);
    for( unsigned i=0; i<num_rows+1; i++)
    {
        csr[i] = std::lower_bound( coo.begin(), coo.end(), i) - coo.begin();
    }
    return csr;
}

template<class Index = int, class Value = double, template<class> class Vector = thrust::host_vector>
struct SparseMatrix
{
    using policy = dg::get_execution_policy<Vector<Value>>;
    template<class OtherMatrix>
    using enable_if_serial = std::enable_if_t<std::is_same_v<typename OtherMatrix::policy, SerialTag>, OtherMatrix>;

    SparseMatrix() = default;
    SparseMatrix( size_t num_rows, size_t num_cols, Vector<Index> row_offsets, Vector<Index> cols, Vector<Value> vals)
    : m_num_rows( num_rows), m_num_cols( num_cols), m_row_offsets( row_offsets), m_cols(cols), m_vals(vals)
    {
        sort();
    }

    template< class I, class V, template<class> class Vec>
    friend class SparseMatrix; // enable copy
    template< class I, class V, template<class> class Vec>
    SparseMatrix( const SparseMatrix<I,V,Vec>& src)
    : m_num_rows( src.m_num_rows), m_num_cols( src.m_num_cols), m_row_offsets(
        src.m_row_offsets), m_cols( src.m_cols), m_vals( src.m_vals)
    {
    }

    void set( size_t num_rows, size_t num_cols, const Vector<Index>& row_offsets, const Vector<Index> cols, const Vector<Value>& vals)
    {
        m_num_rows = num_rows, m_num_cols = num_cols;
        m_row_offsets = row_offsets, m_cols = cols, m_vals = vals;
        sort();
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
    public:
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
    //
    // We enable the following only for serial sparse matrices

    /**
    * @brief Transposition
    *
    * @return  A newly generated SparseMatrix containing the transpose.
    */
    template<class OtherMatrix = SparseMatrix>
    enable_if_serial<OtherMatrix> transpose() const
    {
        SparseMatrix o(*this);
        auto coo = csr2coo(o.m_row_offsets);
        o.m_cols.swap( coo);
        o.m_row_offsets = coo2csr( o.m_num_cols, coo);
        return o;
    }


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
    template<class OtherMatrix = SparseMatrix>
    enable_if_serial<OtherMatrix>& operator*=( const Value& value )
    {
        for( unsigned i=0; i<m_vals.size(); i++)
            m_vals[i] *= value;
        return *this;
    }
    /**
     * @brief add
     *
     * @param lhs
     * @param rhs
     *
     * @return
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
     * @return
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
     * @return
     */
    template<class OtherMatrix = SparseMatrix>
    friend enable_if_serial<OtherMatrix> operator*( const SparseMatrix& lhs, const SparseMatrix& rhs)
    {
        if( lhs.m_num_cols != rhs.m_num_rows)
            throw dg::Error( dg::Message( _ping_) << "Error: cannot multiply matrix with "
                <<lhs.m_num_cols<<" columns with matrix with "<<rhs.m_num_rows<<" rows\n");

        Vector<Index> row_offsets, cols;
        Vector<Value> vals;

        detail::spgemm_cpu_kernel( lhs.m_num_rows, lhs.m_num_cols,
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
        for (int i = 0; i < mat.m_num_rows; i++)
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

    void sort()
    {
        // Sort each row
        for( unsigned row=0; row<m_num_rows; row++)
        {
            if constexpr( std::is_same_v < policy, SerialTag>)
            {
                // All this just to avoid gcc vomiting warnings at me
                thrust::host_vector<Index> tcols( m_cols.begin() + m_row_offsets[row],
                    m_cols.begin() + m_row_offsets[row+1]);
                thrust::host_vector<Value> tvals( m_vals.begin() + m_row_offsets[row],
                    m_vals.begin() + m_row_offsets[row+1]);
                thrust::stable_sort_by_key( tcols.begin(), tcols.end(), tvals.begin());
                thrust::copy( tcols.begin(), tcols.end(), m_cols.begin() + m_row_offsets[row]);
                thrust::copy( tvals.begin(), tvals.end(), m_vals.begin() + m_row_offsets[row]);

            }
            else
                thrust::stable_sort_by_key( m_cols.begin()+m_row_offsets[row],
                    m_cols.begin() + m_row_offsets[row+1],
                    m_vals.begin(), thrust::less<Index>()); // this changes both row_offsets and vals
        }
    }
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
    using tensor_category = SparseBlockMatrixTag;
    using execution_policy = typename SparseMatrix<I,T,V>::policy;
};
///@}
}//namespace dg

