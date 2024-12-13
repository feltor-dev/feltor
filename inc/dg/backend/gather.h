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
* gather1 is the gather map wrt to the unsorted elements!
* To duplicate the sort:
@code{.cpp}
thrust::scatter( unsorted_elements.begin(), unsorted_elements.end(),
                 gather1.begin(), sorted_elements.begin());
@endcode
*
* To undo the sort:
@code{.cpp}
thrust::gather( gather1.begin(), gather1.end(),
                sorted_elements.begin(), unsorted_elements.begin());
@endcode
*
* To get the gather map wrt to sorted elements (i.e the inverse index map)
* "Scatter the index"
@code{.cpp}
auto sort_map = gather1;
auto seq = gather1;
thrust::sequence( seq.begin(), seq.end());
thrust::scatter( seq.begin(), seq.end(), gather1.begin(), sort_map.begin());
@endcode
* Now sort_map indicates where each of the sorted elements went in the unsorted vector
 * Make a gather map from unique to sorted elements via
@code{.cpp}
// To make a gather map from unique elements to unsorted element
thrust::gather( gather1.begin(), gather1.end(),
                gather2.begin(), gather_map.begin());
@endcode
*/
template<class T>
struct Unique
{
    thrust::host_vector<int> gather1;  // gather unsorted elements from sorted elements (order of equal is preserved), bijective
    thrust::host_vector<int> gather2;  // gather sorted elements from unique ( == reduction keys)
    thrust::host_vector<T> unique;     // unique elements, sorted or same order as original
    thrust::host_vector<int> howmany;  // howmany of each unique element is there?
};

thrust::host_vector<int> combine_gather(
    const thrust::host_vector<int>& g1,
    const thrust::host_vector<int>& g2)
{
    thrust::host_vector<int> gather(g1.size());
    for( unsigned u=0; u<g1.size(); u++)
        gather[u] = g2[g1[u]];
    return gather;
}
thrust::host_vector<int> invert_permutation( const thrust::host_vector<int>& p)
{
    auto sort_map = p;
    auto seq = p;
    thrust::sequence( seq.begin(), seq.end());
    thrust::scatter( seq.begin(), seq.end(), p.begin(), sort_map.begin());
    return sort_map;
}

template<class T>
Unique<T> find_unique_stable_sort(
    const thrust::host_vector<T>& unsorted_elements)
{
    Unique<T> uni;
    // 1. Sort pids with elements so we get associated gather map
    auto ids = unsorted_elements;
    uni.gather1.resize( ids.size());
    thrust::sequence( uni.gather1.begin(), uni.gather1.end()); // 0,1,2,3,...
    thrust::stable_sort_by_key( ids.begin(), ids.end(),
            uni.gather1.begin(), std::less()); // this changes both ids and gather1
    // Find unique elements and how often they appear
    thrust::host_vector<T> unique_ids( ids.size());
    thrust::host_vector<int> ones( ids.size(), 1), howmany(ones);
    auto new_end = thrust::reduce_by_key( ids.begin(), ids.end(),
            ones.begin(), unique_ids.begin(), howmany.begin(), std::equal_to() );
    uni.unique = thrust::host_vector<T>( unique_ids.begin(),
            new_end.first);
    uni.howmany = thrust::host_vector<int>( howmany.begin(), new_end.second);
    for( unsigned i=0; i<uni.howmany.size(); i++)
        uni.gather2.insert( uni.gather2.end(), uni.howmany[i], i);

    //invert the gather1 (because above is wrt to sorted vector)
    uni.gather1 = invert_permutation(  uni.gather1);
    return uni;
}

template<class T>
Unique<T> find_unique_order_preserving(
    const thrust::host_vector<T>& unsorted_elements)
{
    Unique<T> uni;
    // find unique elements and how many there are preserving order
    std::vector<std::vector<int>> sort; // gather sorted from unsorted elements
    for( unsigned u=0; u<unsorted_elements.size(); u++)
    {
        auto it =std::find( uni.unique.begin(),
                uni.unique.end(), unsorted_elements[u]);
        if(  it == uni.unique.end()) // not found
        {
            uni.unique.push_back( unsorted_elements[u]);
            sort.push_back( std::vector<int>(1,u));
        }
        else
        {
            size_t idx = std::distance( uni.unique.begin(), it);
            sort[idx].push_back( u);
        }
    }
    // now flatten sort
    for( unsigned i=0; i<uni.unique.size(); i++)
    {
        uni.howmany.push_back( sort[i].size());
        uni.gather2.insert( uni.gather2.end(), uni.howmany[i], i);
        for( int k=0; k<uni.howmany[i]; k++)
            uni.gather1.push_back(sort[i][k]);
    }
    //invert the gather_map (because above is wrt to sorted vector)
    uni.gather1 = invert_permutation( uni.gather1);
    return uni;
}

template<class T>
std::map<int,int> get_size_map( const std::map<int, thrust::host_vector<T>>& idx_map)
{
    std::map<int,int> out;
    for( auto& idx : idx_map)
        out[idx.first] = idx.second.size();
    return out;
}
static inline std::map<int,int> make_size_map( const thrust::host_vector<int>& sizes)
{
    std::map<int,int> out;
    for( unsigned u=0; u<sizes.size(); u++)
    {
        if ( sizes[u] != 0)
            out[u] = sizes[u];
    }
    return out;
}
template<class Keys, class Values>
std::map<dg::get_value_type<Keys>,dg::get_value_type<Values>> make_map(
    const Keys& keys,
    const Values& vals) // same size
{
    std::map<dg::get_value_type<Keys>,dg::get_value_type<Values>> out;
    for( unsigned u=0; u<keys.size(); u++)
        out[keys[u]] = vals[u];
    return out;
}
template<class ContainerType>
std::map<int, ContainerType> make_map(
    const ContainerType& flat_map, // sorted by key
    const std::map<int,int>& size_map // key and chunk size of idx_map
    )
{
    std::map<int, ContainerType> map;
    unsigned start = 0;
    for( auto& size : size_map)
    {
        if( size.second != 0)
        {
            map[size.first] = ContainerType(
                flat_map.begin()+start, flat_map.begin() + start + size.second);
            start += size.second;
        }
    }
    return map;
}

// keys get lost
template<class ContainerType>
thrust::host_vector<dg::get_value_type<ContainerType>> flatten_map(
    const std::map<int,ContainerType>& idx_map // map is sorted automatically
    )
{
    thrust::host_vector<dg::get_value_type<ContainerType>> flat;
    // flatten map
    for( auto& idx : idx_map)
        flat.insert(flat.end(), idx.second.begin(), idx.second.end());
    return flat;
}

static inline std::map<int,int> contiguous_range(
    const thrust::host_vector<int>& in)
{
    std::map<int,int> range;
    for( unsigned u=0; u<in.size(); u++)
    {
        int idx = in[u];
        if( range.find(idx) != range.end())
        {
            range[idx]++;
        }
        else
            range[idx] = 1;
    }
    return range;
}


}// namespace detail
///@endcond
// TODO Maybe this can be public
// Convert a unsorted and possible duplicate global index list to unique
// stable_sorted by pid and duplicates map
// bufferIdx gives
// idx 0 is pid, idx 1 is localIndex on that pid
// TODO gIdx can be unsorted and contain duplicate entries
// TODO idx 0 is pid, idx 1 is localIndex on that pid
// TODO Maybe be a static MPIGather function?
inline static std::map<int, thrust::host_vector<int>> gIdx2unique_idx(
    const thrust::host_vector<std::array<int,2>>& gIdx, // unsorted (cannot be map)
    thrust::host_vector<int>& bufferIdx) // gIdx size, gather gIdx from flatten_map
{
    detail::Unique<std::array<int,2>> uni = detail::find_unique_order_preserving( gIdx);
    // get pids
    thrust::host_vector<int> pids(uni.unique.size()),
        lIdx(pids);
    for( int i=0; i<(int)pids.size(); i++)
    {
        pids[i] = uni.unique[i][0];
        lIdx[i] = uni.unique[i][1]; // the local index
    }
    detail::Unique<int> uni_pids = detail::find_unique_stable_sort( pids);
    bufferIdx = detail::combine_gather( uni.gather1,
                   detail::combine_gather( uni.gather2, uni_pids.gather1));
    // duplicate the sort on lIdx
    auto sorted_unique_gIdx = lIdx;
    thrust::scatter( lIdx.begin(), lIdx.end(),
             uni_pids.gather1.begin(), sorted_unique_gIdx.begin());
    // return map
    auto pids_howmany = detail::make_map( uni_pids.unique, uni_pids.howmany);
    return detail::make_map( sorted_unique_gIdx, pids_howmany);
}

inline static thrust::host_vector<int> make_kron_indices(
    unsigned left_size, // (local) of SparseBlockMat
    const thrust::host_vector<int>& idx, // local 1d indices in units of n
    unsigned n, // block size
    unsigned num_cols, // (local) number of block columns (of gatherFrom)
    unsigned right_size) // (local) of SparseBlockMat
{
    thrust::host_vector<int> g2( idx.size()*n*left_size*right_size);
    for( unsigned l=0; l<idx.size(); l++)
    for( unsigned j=0; j<n; j++)
    for( unsigned i=0; i<left_size; i++)
    for( unsigned k=0; k<right_size; k++)
        g2[((l*n+j)*left_size+i)*right_size + k] =
             ((i*num_cols + idx[l])*n + j)*right_size + k;
    return g2;
}



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
