#pragma once
#include <vector>
#include "predicate.h"

namespace dg
{
///@cond

// The rationale for having this class is so that a std::vector of pointers
// is recognised as a matrix type rather than a recursive vector
// in blas2 functions

/**
 * @brief A lightweight dense matrix based on a vector of pointers
 *
 * The pointers reference the column vectors of the matrix
 * @tparam ContainerType
 */
template<class ContainerType>
struct DenseMatrix
{
    using container_type = ContainerType;
    DenseMatrix() = default;
    DenseMatrix( const std::vector<const ContainerType*>& columns) :
        m_matrix( columns), m_num_cols(columns.size()) {}
    DenseMatrix( const std::vector<const ContainerType*>& columns, unsigned num_cols) :
        m_matrix( columns), m_num_cols(num_cols) {}

    unsigned num_cols() const{
        return m_num_cols;
    }
    const ContainerType * operator[] ( unsigned idx) const
    {
        return m_matrix[idx];
    }
    const std::vector<const ContainerType*>& get() const{ return m_matrix;}
    // Should we delte copy and assignment ?

    private:
    const std::vector<const ContainerType*>& m_matrix;
    unsigned m_num_cols;
};
template <class Container>
struct TensorTraits<DenseMatrix<Container> >
{
    using value_type  = get_value_type<Container>;
    using execution_policy = get_execution_policy<Container>;
    using tensor_category = DenseMatrixTag;
};

///@endcond
/*! @brief Lightweight DenseMatrix for \c dg::blas2::gemv
 *
 * The philosophy is that a column matrix is represented by a
 * \c std::vector of pointers
 * and can be multiplied with a coefficient vector
 * \f[ \vec y = V \vec c = \sum_i c_i \vec v_{i} \f]
 * where \f$ v_i\f$ are the columns of \f$ V\f$
 * @code{.cpp}
    std::vector<Container> matrix( 10, x);
    std::vector<const Container*> column_ptrs = dg::asPointers(matrix);
    std::vector<value_type> coeffs( 10, 0.5);
    dg::blas2::gemv( 1., dg::asDenseMatrix(matrix_ptrs), coeffs, 0.,  x);
 * @endcode
 * @note the implemented summation algorithm is a pairwise summation algorithm
 * optimized for small sized number of columns ( <= 64)
 * @param in a collection of pointers that form the columns of the dense matrix
 * @return an opaque type that internally  adds a Tag that tells the compiler
 * that the \c std::vector is to be interpreted as a dense matrix and call the
 * correct implementation.
 * @copydoc hide_ContainerType
 * @ingroup densematrix
 */
template<class ContainerType>
auto asDenseMatrix( const std::vector<const ContainerType*>& in)
{
    // TODO could be marked deprecated because with C++17 the compiler can figure out the type of DenseMatrix
    return DenseMatrix<ContainerType>(in);
}


///@copydoc asDenseMatrix(const std::vector<const ContainerType*>&)
///@param size only the first \c size pointers are used in the matrix (i.e. the
///number of columns is \c size)
///@ingroup densematrix
template<class ContainerType>
auto asDenseMatrix( const std::vector<const ContainerType*>& in, unsigned size)
{
    // TODO could be marked deprecated because with C++17 the compiler can figure out the type of DenseMatrix
    return DenseMatrix<ContainerType>(in, size);
}

/*! @brief Convert a vector of vectors to a vector of pointers
 *
 * A convenience function that can be used in combination with \c asDenseMatrix
 * @code{.cpp}
    std::vector<Container> matrix( 10, x);
    std::vector<const Container*> column_ptrs = dg::asPointers(matrix);
    std::vector<value_type> coeffs( 10, 0.5);
    dg::blas2::gemv( 1., dg::asDenseMatrix(matrix_ptrs), coeffs, 0.,  x);
 * @endcode
 * @param in a collection of vectors that form the columns of the dense matrix
 * @return a vector of pointers with ptrs[i] = &in[i]
 * @attention DO NOT HOLD POINTERS AS PRIVATE DATA MEMBERS IN A CLASS unless
 * you also plan to overload the copy and assignment operators
 * @copydoc hide_ContainerType
 * @ingroup densematrix
 */
template<class ContainerType>
std::vector<const ContainerType*> asPointers( const std::vector<ContainerType>& in)
{
    std::vector<const ContainerType*> ptrs( in.size(), nullptr);
    for( unsigned i=0; i<ptrs.size(); i++)
        ptrs[i] = &in[i];
    return ptrs;
}

} // namespace dg
