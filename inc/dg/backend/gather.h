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
// For the index battle:

/*!
* gather_map is the gather map wrt to the unsorted elements!
* To duplicate the sort:
@code{.cpp}
thrust::scatter( unsorted_elements.begin(), unsorted_elements.end(),
                 gather_map.begin(), sorted_elements.begin());
@endcode
*
* To undo the sort:
@code{.cpp}
thrust::gather( gather_map.begin(), gather_map.end(),
                sorted_elements.begin(), unsorted_elements.begin());
@endcode
*
* To get the gather map wrt to sorted elements (i.e the inverse index map)
* "Scatter the index"
@code{.cpp}
auto sort_map = gather_map;
auto seq = gather_map;
thrust::sequence( seq.begin(), seq.end());
thrust::scatter( seq.begin(), seq.end(), gather_map.begin(), sort_map.begin());
@endcode
* Now sort_map indicates where each of the sorted elements went in the unsorted vector
*/
template<class T>
void find_unique_stable_sort(
        const thrust::host_vector<T>& unsorted_elements,
        thrust::host_vector<int>& gather_map,     // gather unsorted elements from sorted elements (order of equal is preserved), bijective
        thrust::host_vector<int>& reduction_keys, // gather sorted elements from unique_elements
        thrust::host_vector<T>& unique_elements,
        thrust::host_vector<int>& howmany_elements)
{
    // 1. Sort pids with elements so we get associated gather map
    auto ids = unsorted_elements;
    gather_map.resize( ids.size());
    thrust::sequence( gather_map.begin(), gather_map.end()); // 0,1,2,3,...
    thrust::stable_sort_by_key( ids.begin(), ids.end(),
            gather_map.begin(), std::less()); // this changes both ids and gather_map
    // Find unique elements and how often they appear
    thrust::host_vector<T> unique_ids( ids.size());
    thrust::host_vector<int> ones( ids.size(), 1), howmany(ones);
    auto new_end = thrust::reduce_by_key( ids.begin(), ids.end(),
            ones.begin(), unique_ids.begin(), howmany.begin(), std::equal_to() );
    unique_elements = thrust::host_vector<T>( unique_ids.begin(),
            new_end.first);
    howmany_elements = thrust::host_vector<int>( howmany.begin(), new_end.second);
    reduction_keys.clear();
    for( unsigned i=0; i<howmany_elements.size(); i++)
        reduction_keys.insert( reduction_keys.end(), howmany_elements[i], i);

    //invert the gather_map (because above is wrt to sorted vector)
    auto sort_map = gather_map;
    auto seq = gather_map;
    thrust::sequence( seq.begin(), seq.end());
    thrust::scatter( seq.begin(), seq.end(), gather_map.begin(), sort_map.begin());
    gather_map = sort_map;
}

/*!
 * Make a gather map from unique_elements to sorted elements via
@code{.cpp}
// To make a gather map from unique elements to unsorted element
thrust::gather( gather_map.begin(), gather_map.end(),
                reduction_keys.begin(), sort_map.begin());
@endcode
 */
template<class T>
void find_unique_order_preserving(
    const thrust::host_vector<T>& unsorted_elements,
    thrust::host_vector<int>& gather_map,     // Gather unsorted elements from reordered elements ( size == elements.size()), bijective
    thrust::host_vector<int>& reduction_keys, // gather sorted elements from unique_elements
    thrust::host_vector<T>& unique_elements,
    thrust::host_vector<int>& howmany_elements)
{
    // find unique elements and how many there are preserving order
    unique_elements.clear();
    howmany_elements.clear();
    std::vector<std::vector<int>> sort; // gather sorted from unsorted elements
    for( unsigned u=0; u<unsorted_elements.size(); u++)
    {
        auto it =std::find( unique_elements.begin(), unique_elements.end(), unsorted_elements[u]);
        if(  it == unique_elements.end()) // not found
        {
            unique_elements.push_back( unsorted_elements[u]);
            sort.push_back( std::vector<int>(1,u));
        }
        else
        {
            size_t idx = std::distance( unique_elements.begin(), it);
            sort[idx].push_back( u);
        }
    }
    // now flatten sort
    for( unsigned i=0; i<unique_elements.size(); i++)
    {
        howmany_elements.push_back( sort[i].size());
        reduction_keys.insert( reduction_keys.end(), howmany_elements[i], i);
        for( int k=0; k<howmany_elements[i]; k++)
            gather_map.push_back(sort[i][k]);
    }
    //invert the gather_map (because above is wrt to sorted vector)
    auto sort_map = gather_map;
    auto seq = gather_map;
    thrust::sequence( seq.begin(), seq.end());
    thrust::scatter( seq.begin(), seq.end(), gather_map.begin(), sort_map.begin());
    gather_map = sort_map;
}

// map is sorted
template<class T>
std::map<int, thrust::host_vector<T>> make_map(
    const thrust::host_vector<T>& idx_map, // sorted by pid
    const thrust::host_vector<int>& pids, // unique
    const thrust::host_vector<int>& howmany // size
    )
{
    std::map<int, thrust::host_vector<T>> map;
    unsigned start = 0;
    for( unsigned u=0; u<pids.size(); u++)
    {
        if( howmany[u] != 0)
        {
            map[pids[u]] = thrust::host_vector<T>( howmany[u]);
            for( unsigned i=0; i<(unsigned)howmany[u]; i++)
            {
                map[pids[u]][i] = idx_map[ start + i];
            }
            start += howmany[u];
        }
    }
    return map;
}

template<class T>
thrust::host_vector<T> flatten_map(
    const std::map<int,thrust::host_vector<T>>& idx_map // map is sorted automatically
    )
{
    thrust::host_vector<T> flat;
    // flatten map
    for( auto& idx : idx_map)
        for( unsigned u=0; u<idx.second.size(); u++)
            flat.push_back( idx.second[u]);
    return flat;
}

// Convert a unsorted and possible duplicate global index list to unique
// stable_sorted by pid and duplicates
// idx 0 is pid, idx 1 is localIndex on that pid
inline static void gIdx2unique_idx(
    const thrust::host_vector<std::array<int,2>>& gIdx,
    thrust::host_vector<int>& bufferIdx, // gIdx size, gather gIdx from sorted_unique_gIdx
    thrust::host_vector<int>& sorted_unique_gIdx,
    thrust::host_vector<int>& unique_pids, // sorted
    thrust::host_vector<int>& howmany_pids)
{
    thrust::host_vector<int> gather_map1, gather_map2, howmany;
    thrust::host_vector<std::array<int,2>> locally_unique_gIdx;
    detail::find_unique_order_preserving( gIdx, gather_map1,
        gather_map2, locally_unique_gIdx, howmany);
    // get pids
    thrust::host_vector<int> pids(locally_unique_gIdx.size()),
        lIdx(pids);
    for( int i=0; i<(int)pids.size(); i++)
    {
        pids[i] = locally_unique_gIdx[i][0];
        lIdx[i] = locally_unique_gIdx[i][1]; // the local index
    }
    thrust::host_vector<int> gather_map3, red_keys;
    detail::find_unique_stable_sort( pids, gather_map3, red_keys,
        unique_pids, howmany_pids);
    // duplicate the sort on lIdx
    sorted_unique_gIdx = lIdx;
    thrust::scatter( lIdx.begin(), lIdx.end(),
             gather_map3.begin(), sorted_unique_gIdx.begin());
    // buffer index
    bufferIdx.resize( gather_map1.size());
    for( unsigned u=0; u<bufferIdx.size(); u++)
        bufferIdx[u] = gather_map3[gather_map2[gather_map1[u]]];
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
            thrust::scatter( buffer.begin(), buffer.end(), m_scatter2reduction.begin(),
                m_reduction_buffer.template get<value_type>().begin());

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
        // possible Optimisation: if scatter2target is contiguous one does not need to scatter
        // For the scatter operation we need to catch the case that multiple
        // values scatter to the same place
        //
        // buffer -> reduction_buf -> scatter_buf -> store
        // In that case our idea is to first gather the indices such that values
        // to reduce are next to each other
        // this allows to use thrust::reduce_by_key from reduction buffer into
        // a scatter buffer.
        // Finally we can scatter the values from there after setting explicit 0s

        thrust::host_vector<int> sort_map, reduction_keys, unique_indices, howmany;
        detail::find_unique_order_preserving( thrust::host_vector<int>(m_idx),
            sort_map, reduction_keys, unique_indices, howmany);
        assert( reduction_keys.size() == m_idx.size());
        if( unique_indices.size() != m_idx.size())
        {
            m_reduction = true;
            m_scatter_buffer_size = unique_indices.size();
            m_reduction_buffer_size = m_idx.size();
            m_scatter2target = unique_indices;
            m_red_keys = reduction_keys;
            m_scatter2reduction = sort_map;
            m_unique_keys.resize( m_scatter_buffer_size);
        }
        m_allocated =true;
    }
    mutable bool m_reduction = false, m_allocated = false;
    mutable unsigned m_scatter_buffer_size, m_reduction_buffer_size;
    mutable Vector<int> m_scatter2target, m_red_keys, m_scatter2reduction; // may be empty
    mutable Vector<int> m_unique_keys;
    mutable detail::AnyVector<Vector> m_scatter_buffer, m_reduction_buffer;
};
}// namespace dg
