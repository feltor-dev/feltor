#pragma once
#include <vector>
#include "predicate.h"

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
// now have utilities that construct a vector of pointers
template<class ContainerType>
auto asDenseMatrix( const std::vector<const ContainerType*>& in)
{
    return DenseMatrix<ContainerType>(in);
}
template<class ContainerType>
auto asDenseMatrix( const std::vector<const ContainerType*>& in, unsigned size)
{
    return DenseMatrix<ContainerType>(in, size);
}

template<class ContainerType>
std::vector<const ContainerType*> asPointers( const std::vector<ContainerType>& in)
{
    std::vector<const ContainerType*> ptrs( in.size(), nullptr);
    for( unsigned i=0; i<ptrs.size(); i++)
        ptrs[i] = &in[i];
    return ptrs;
}

} // namespace dg
