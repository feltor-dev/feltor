#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include "tensor_traits.h"
#include "tensor_traits_scalar.h"
#include "tensor_traits_thrust.h"
#include "tensor_traits_cusp.h"
#include "tensor_traits_std.h"
#include "blas2_dispatch_shared.h"
#include "config.h"
#include "memory.h"

namespace dg{
//


/*! @brief Gather sparse matrix \f$ G_{ij} = \delta_{g(i) j}\f$ and its transpose
 *
 * A gather matrix is a coo sparse matrix format where all values are 1
 * the row indices are simply the row \f$ r_i = i\f$ and the column indices
 * are given by the gather map \f$ c_i = g(i)\f$.
 *
 * The **transpose** of the gather matrix is a scatter matrix where the column indices
 * are simiply the column \f$ c_i =i\f$ and the row index is the gather map
 * \f$ r_i = g(i)\f$.
 * @tparam Vector storage class for indices, must have the execution policy that
 * is later used for gather/scatter operations
 */
template<template<class >class Vector>
struct LocalGatherMatrix
{
    /// Empty
    LocalGatherMatrix() = default;

    /*! @brief Construct from column indices / gather map
     * @param index_map each index is a column index
     */
    LocalGatherMatrix( const thrust::host_vector<int>& index_map)
    : m_idx( index_map) { }

    template<template<class> class OtherVector>
    friend class LocalGatherMatrix;
    /*! @brief Copy from another storage class
     * @param src Copy from
     */
    template<template<class> class OtherVector>
    LocalGatherMatrix( const LocalGatherMatrix<OtherVector>& src)
    : m_idx ( src.m_idx) { }
    // TODO check if m_idx is identity

    /// Index map from constructor
    const Vector<int>& index_map() const{ return m_idx;}

    template<class ContainerType0, class ContainerType1>
    void gather(const ContainerType0& store, ContainerType1& buffer) const
    {
        gather( 1, store, 0, buffer);
    }

    /// \f$ w = \alpha G v + \beta w \f$
    template<class ContainerType0, class ContainerType1>
    void gather(dg::get_value_type<ContainerType0> alpha, const ContainerType0& store,
                dg::get_value_type<ContainerType1> beta, ContainerType1& buffer) const
    {
        thrust::gather( m_idx.begin(), m_idx.end(), store.begin(), buffer.begin());
        unsigned size = buffer.size();
        using value_type= dg::get_value_type<ContainerType0>;
        dg::blas2::detail::doParallelFor( SharedVectorTag(),
            [size, alpha, beta]DG_DEVICE( unsigned i, const value_type* x,
                const int* idx, value_type* y) {
                y[i] = alpha*x[idx[i]] + beta*y[i];
            }, size, store, m_idx, buffer);
    }
    private:
    Vector<int> m_idx; // this fully defines the matrix ( so it's the only non-mutable)
};
}// namespace dg
