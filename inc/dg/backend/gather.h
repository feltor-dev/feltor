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
template<template<class> class host_vector, class T>
void find_same(
        const host_vector<T>& numbers, // numbers must be sorted
        host_vector<T>& unique_numbers,
        host_vector<int>& howmany_numbers // same size as uniqe_numbers
        )
{
    // Find unique numbers and how often they appear
    host_vector<T> unique_ids( numbers.size());
    host_vector<int> ones( numbers.size(), 1),
        howmany(ones);
    auto new_end = thrust::reduce_by_key( numbers.begin(), numbers.end(),
            //numbers are the keys by which one is reduced
            ones.begin(), unique_ids.begin(), howmany.begin(), std::equal_to() );
    unique_numbers = host_vector<T>( unique_ids.begin(),
            new_end.first);
    howmany_numbers = host_vector<int>( howmany.begin(), new_end.second);
}

//given indices -> make a sorted unique indices vector + a gather map
//into the unique vector:
//@param buffer_idx -> (gather map/ new column indices) same size as indices
//( can alias indices, index into unique_indices
//@param unique_indices -> (list of unique indices)
template<template<class> class host_vector, class T>
void findUniqueIndices(
    const host_vector<T>& indices,   // Unsorted
    host_vector<int>& sort_map,      // Gather indices into sorted indices
    host_vector<int>& reduction_keys,// Gather unique indices into sorted indices
    host_vector<int>& buffer_idx,    // Gather unique indices into indices
    host_vector<T>& unique_indices)  // Sorted
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
    host_vector<int> howmany;
    auto ids = indices;
    sort_map.resize( ids.size());
    thrust::sequence( sort_map.begin(), sort_map.end()); // 0,1,2,3,...
    thrust::stable_sort_by_key( ids.begin(), ids.end(),
            sort_map.begin(), std::less()); // this changes both ids and sort_map

    find_same<host_vector,T>( ids, unique_indices,
            howmany);

    // manually make gather map from sorted into unique_indices
    for( unsigned i=0; i<unique_indices.size(); i++)
        for( int j=0; j<howmany[i]; j++)
            reduction_keys.push_back(i);
    assert( reduction_keys.size() == indices.size());
    // buffer idx is the new index
    buffer_idx.resize( indices.size());
    thrust::scatter( reduction_keys.begin(), reduction_keys.end(), sort_map.begin(),
            buffer_idx.begin());
}


}// namespace detail
///@endcond

// TODO document or make non-public
// Given an index map construct the associate gather matrix and its transpose
// Represents the matrix \f$ G_{ij} = \delta_{g[i]j}\f$ and its transpose
template<template<class >class Vector>
struct LocalGatherMatrix
{

    // index_map gives num_rows of G, so we need num_cols as well
    LocalGatherMatrix( const thrust::host_vector<int>& index_map)
    : m_idx( index_map)
    {
        // For the scatter operation we need to catch the case that multiple
        // values scatter to the same place
        // buffer -> reduction_buf -> scatter_buf -> store
        // In that case our idea is to first sort the indices such that values
        // to reduce are next to each other
        // this allows to use thrust::reduce_by_key from reduction buffer into
        // a scatter buffer.
        // Finally we can scatter the values from there after setting explicit 0s

        thrust::host_vector<int> sort_map, reduction_keys, buffer_idx, unique_indices;
        detail::findUniqueIndices<thrust::host_vector>(
            index_map, sort_map, reduction_keys, buffer_idx, unique_indices);
        if( unique_indices.size() == index_map.size())
        {
            m_reduction = false;
            m_scatter_buffer_size = 0;
            m_reduction_buffer_size = 0;
        }
        else // we need to first gather into sorted buffer -> reduce -> scatter
        {
            m_reduction = true;
            m_scatter_buffer_size = unique_indices.size();
            m_reduction_buffer_size = index_map.size();
            m_scatter2target = unique_indices;
            m_red_keys = reduction_keys;
            m_unique_keys = unique_indices;
            m_gather2reduction = sort_map;
        }
    }

    template<template<class> class OtherVector>
    friend class LocalGatherMatrix;
    template<template<class> class OtherVector>
    LocalGatherMatrix( const LocalGatherMatrix<OtherVector>& src)
    : m_idx ( src.m_idx)
    {}

    const Vector<int>& index_map() const{ return m_idx;}

    // w = Gv
    template<class ContainerType0, class ContainerType1>
    void gather( const ContainerType0& store, ContainerType1& buffer) const
    {
        thrust::gather( m_idx.begin(), m_idx.end(), store.begin(), buffer.begin());
    }
    // v = v + S w
    template<class ContainerType0, class ContainerType1>
    void scatter_plus( const ContainerType0& buffer, ContainerType1& store)
    {
        using value_type= dg::get_value_type<ContainerType0>;
        if( !m_reduction)
        {
            unsigned size = buffer.size();
            dg::blas2::detail::doParallelFor( SharedVectorTag(),
                [size]DG_DEVICE( unsigned i, const value_type* x,
                    const int* idx, value_type* y) {
                    y[idx[i]] += x[i];
                }, size, buffer, m_idx, store);
        }
        else
        {
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
                [size]DG_DEVICE( unsigned i, const value_type* x,
                    const int* idx, value_type* y) {
                    y[idx[i]] += x[i];
                }, size, m_scatter_buffer.template get<value_type>(), m_scatter2target, store);
        }
    }
    private:
    Vector<int> m_idx;
    Vector<int> m_scatter2target, m_red_keys, m_gather2reduction; // may be empty
    bool m_reduction = false;
    unsigned m_scatter_buffer_size = 0, m_reduction_buffer_size = 0;
    mutable Vector<int> m_unique_keys;
    mutable detail::AnyVector<Vector> m_scatter_buffer, m_reduction_buffer;
};
}// namespace dg
