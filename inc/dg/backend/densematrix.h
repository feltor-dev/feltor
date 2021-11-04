#pragma once
#include <vector>
#include "predicate.h"
#include "blas1_serial.h"

namespace dg
{
///@cond

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
    DenseMatrix( unsigned num_cols) : m_matrix( num_cols, nullptr){}
    DenseMatrix( const std::vector<ContainerType>& columns) : m_matrix( columns.size(), nullptr)
    {
        for( unsigned i=0; i<m_matrix.size(); i++)
            m_matrix[i] = &columns[i];
    }
    DenseMatrix( const std::vector<const ContainerType*>& columns) : m_matrix( columns.size(), nullptr)
    {
        for( unsigned i=0; i<m_matrix.size(); i++)
            m_matrix[i] = columns[i];
    }

    unsigned num_cols() const{
        return m_matrix.size();
    }

    const ContainerType* &operator[] ( unsigned idx)
    {
        return m_matrix[idx];
    }
    const ContainerType * operator[] ( unsigned idx) const
    {
        return m_matrix[idx];
    }
    const std::vector<const ContainerType*>& get() const{ return m_matrix;}

    private:
    std::vector<const ContainerType*> m_matrix;
};

///@endcond

template<class ContainerType>
DenseMatrix<ContainerType> asDenseMatrix( const std::vector<ContainerType>& in)
{
    return DenseMatrix<ContainerType>( in);
}
template<class ContainerType>
DenseMatrix<ContainerType> asDenseMatrix( const std::vector<ContainerType const*>& in)
{
    return DenseMatrix<ContainerType>( in);
}
template<class ContainerType>
DenseMatrix<ContainerType> asDenseMatrix( ContainerType const* const * begin, ContainerType const* const* end)
{
    std::vector<ContainerType const*> in( begin, end);
    return DenseMatrix<ContainerType>( in);
}
///@addtogroup dispatch
///@{
template <class Container>
struct TensorTraits<DenseMatrix<Container> >
{
    using value_type  = get_value_type<Container>;
    using execution_policy = get_execution_policy<Container>;
    using tensor_category = DenseMatrixTag;
};
///@}
//

} // namespace dg
