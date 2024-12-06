#pragma once

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include "blas2_dispatch_shared.h"
#include "config.h"
#include "memory.h"

namespace dg{
///@cond
namespace detail{
template<class T>
void find_same(
        const thrust::host_vector<T>& numbers, // numbers must be sorted
        thrust::host_vector<T>& unique_numbers,
        thrust::host_vector<int>& howmany_numbers // same size as uniqe_numbers
        )
{
    // Find unique numbers and how often they appear
    thrust::host_vector<T> unique_ids( numbers.size());
    thrust::host_vector<int> ones( numbers.size(), 1),
        howmany(ones);
    auto new_end = thrust::reduce_by_key( numbers.begin(), numbers.end(),
            //numbers are the keys by which one is reduced
            ones.begin(), unique_ids.begin(), howmany.begin(), std::equal_to() );
    unique_numbers = thrust::host_vector<T>( unique_ids.begin(),
            new_end.first);
    howmany_numbers = thrust::host_vector<int>( howmany.begin(), new_end.second);
}

//given indices -> make a sorted unique indices vector + a gather map
//into the unique vector:
//@param buffer_idx -> (gather map/ new column indices) same size as indices
//( can alias indices, index into unique_indices
//@param unique_indices -> (list of unique indices)
template<class T>
void find_unique(
    const thrust::host_vector<T>& indices,   // Unsorted
    thrust::host_vector<int>& sort_map,      // Gather indices into sorted indices
    thrust::host_vector<int>& reduction_keys,// Gather unique indices into sorted indices
    thrust::host_vector<int>& buffer_idx,    // Gather unique indices into indices
    thrust::host_vector<T>& unique_indices)  // Sorted
{
    // sort_map is the gather map wrt to the sorted vector!
    // To duplicate the sort:
    // thrust::gather( sort_map.begin(), sort_map.end(), numbers.begin(), sorted.begin());
    // To undo the sort:
    // thrust::scatter( sorted.begin(), sorted.end(), sort_map.begin(), numbers.begin());
    // To get the gather map wrt to unsorted vector (i.e the inverse index map)
    // "Scatter the index"
    // auto gather_map = sort_map;
    // auto seq = sort_map;
    // thrust::sequence( seq.begin(), seq.end());
    // thrust::scatter( seq.begin(), seq.end(), sort_map.begin(), gather_map.begin());
    // Now gather_map indicates where each of the numbers went in the sorted vector
    // 1. Sort pids with indices so we get associated gather map
    thrust::host_vector<int> howmany;
    auto ids = indices;
    sort_map.resize( ids.size());
    thrust::sequence( sort_map.begin(), sort_map.end()); // 0,1,2,3,...
    thrust::stable_sort_by_key( ids.begin(), ids.end(),
            sort_map.begin(), std::less()); // this changes both ids and sort_map

    find_same<T>( ids, unique_indices,
            howmany);

    // manually make gather map from sorted into unique_indices on host
    thrust::host_vector<int> h_howmany(howmany), h_reduction_keys;
    for( unsigned i=0; i<unique_indices.size(); i++)
        for( int j=0; j<h_howmany[i]; j++)
            h_reduction_keys.push_back(i);
    assert( h_reduction_keys.size() == indices.size());
    reduction_keys = h_reduction_keys;
    // buffer idx is the new index
    buffer_idx.resize( indices.size());
    thrust::scatter( reduction_keys.begin(), reduction_keys.end(), sort_map.begin(),
            buffer_idx.begin());
}


}// namespace detail
///@endcond

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
    /// \f$ v = v + S w\f$
    template<class ContainerType0, class ContainerType1>
    void scatter_plus(const ContainerType0& buffer, ContainerType1& store) const
    {
        if( ! m_allocated) // 1st time scatter is called
            prepare_scatter();
        // This is essentially an implementation for a coo format
        // sparse matrix vector multiplication
        // Note a S w + b v is not possible efficiently in one go
        using value_type= dg::get_value_type<ContainerType0>;
        auto scatter_lambda = []DG_DEVICE( unsigned i, const value_type* x,
            const int* idx, value_type* y) { y[idx[i]] += x[i]; };
        if( !m_reduction) // this is fast
        {
            unsigned size = buffer.size();
            dg::blas2::detail::doParallelFor( SharedVectorTag(),
                scatter_lambda, size, buffer, m_idx, store);
        }
        else
        {
            // this is not efficient (needs to load/store 3x)
            // (We assume this case is more of an edge case)
            // 1. Gather into sorted indices
            m_reduction_buffer.template set<value_type>( m_reduction_buffer_size);
            thrust::gather( m_gather2reduction.begin(), m_gather2reduction.end(),
                buffer.begin(), m_reduction_buffer.template get<value_type>().begin());

            // 2. Reduce multiple sorted indices
            m_scatter_buffer.template set<value_type>( m_scatter_buffer_size);
            thrust::reduce_by_key( m_red_keys.begin(), m_red_keys.end(),
                m_reduction_buffer.template get<value_type>().begin(),
                m_unique_keys.begin(), // keys output
                m_scatter_buffer.template get<value_type>().begin()); // values out

            // 3. Scatter into target
            unsigned size = m_scatter_buffer_size;
            dg::blas2::detail::doParallelFor( SharedVectorTag(),
                scatter_lambda, size, m_scatter_buffer.template get<value_type>(),
                m_scatter2target, store);
        }
    }
    private:
    Vector<int> m_idx; // this fully defines the matrix ( so it's the only non-mutable)
    void prepare_scatter( ) const
    {
        // For the scatter operation we need to catch the case that multiple
        // values scatter to the same place
        //
        // buffer -> reduction_buf -> scatter_buf -> store
        // In that case our idea is to first sort the indices such that values
        // to reduce are next to each other
        // this allows to use thrust::reduce_by_key from reduction buffer into
        // a scatter buffer.
        // Finally we can scatter the values from there after setting explicit 0s

        thrust::host_vector<int> sort_map, reduction_keys, buffer_idx, unique_indices;
        detail::find_unique( thrust::host_vector<int>(m_idx), sort_map,
            reduction_keys, buffer_idx, unique_indices);
        if( unique_indices.size() != m_idx.size())
        {
            m_reduction = true;
            m_scatter_buffer_size = unique_indices.size();
            m_reduction_buffer_size = m_idx.size();
            m_scatter2target = unique_indices;
            m_red_keys = reduction_keys;
            m_gather2reduction = sort_map;
            m_unique_keys.resize( m_scatter_buffer_size);
        }
        m_allocated =true;
    }
    mutable bool m_reduction = false, m_allocated = false;
    mutable unsigned m_scatter_buffer_size, m_reduction_buffer_size;
    mutable Vector<int> m_scatter2target, m_red_keys, m_gather2reduction; // may be empty
    mutable Vector<int> m_unique_keys;
    mutable detail::AnyVector<Vector> m_scatter_buffer, m_reduction_buffer;
};
}// namespace dg
