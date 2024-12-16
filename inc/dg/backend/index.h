#pragma once
#include <limits>
#include <thrust/sort.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include "tensor_traits.h"
#include "tensor_traits_scalar.h"
#include "tensor_traits_thrust.h"
#include "tensor_traits_cusp.h"
#include "tensor_traits_std.h"

namespace dg
{
///@cond
namespace detail{
// For the index battle:

// map from vector of keys and values
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

/// \f$ G = G_1 \cdot G_2\f$
template<class IntegerVec>
IntegerVec combine_gather( const IntegerVec& g1, const IntegerVec& g2)
{
    IntegerVec gather = g1;
    thrust::gather( g1.begin(), g1.end(), g2.begin(), gather.begin());
    return gather;
}

/*! @brief \f$ P^{-1} = P^T\f$

* To get the gather map wrt to sorted elements (i.e the inverse index map)
* "Scatter the index"
@code{.cpp}
auto sort_map = gather1;
auto seq = gather1;
thrust::sequence( seq.begin(), seq.end());
thrust::scatter( seq.begin(), seq.end(), gather1.begin(), sort_map.begin());
@endcode
*/
template<class IntegerVec>
IntegerVec invert_permutation( const IntegerVec& p)
{
    auto sort_map = p;
    auto seq = p;
    thrust::sequence( seq.begin(), seq.end());
    thrust::scatter( seq.begin(), seq.end(), p.begin(), sort_map.begin());
    return sort_map;
}



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

/// Utility for MPIContiguousGather
struct MsgChunk
{
    int idx; /// !< starting index of message
    int size; /// !< size of message
};
static inline thrust::host_vector<MsgChunk>
find_contiguous_chunks(
    const thrust::host_vector<int>& in) // These are unsorted
{
    thrust::host_vector<MsgChunk> range;
    int previous = std::numeric_limits<int>::max();
    for( unsigned u=0; u<in.size(); u++)
    {
        int idx = in[u];
        if( idx -1 == previous)
            range.back().size ++;
        else
            range.push_back({idx,1});
        previous = idx;
    }
    return range;
}



}// namespace detail
///@endcond
//
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
    std::map<int, thrust::host_vector<int>> out;
    std::map<int,int> pids_howmany = detail::make_map( uni_pids.unique, uni_pids.howmany);
    unsigned start = 0;
    for( auto& pids : pids_howmany)
        if( pids.second != 0)
        {
            thrust::host_vector<int> partial(
                sorted_unique_gIdx.begin()+start, sorted_unique_gIdx.begin() + start + pids.second);
            out[pids.first] = partial;
            start += pids.second;
        }
    return out;
}


/**
 * @brief Construct from global indices index map
 *
 * Uses the \c global2localIdx() member of MPITopology to generate \c
 * localIndexMap and \c pidIndexMap
 * @param globalIndexMap Each element <tt> globalIndexMap[i] </tt>
 * represents a global vector index from (or to) where to take the value
 * <tt>buffer[i]</tt>. There are <tt> local_buffer_size =
 * globalIndexMap.size() </tt> messages.
 * @param p the conversion object
 * @tparam ConversionPolicy has to have the members:
 *  - <tt> bool global2localIdx(unsigned,unsigned&,unsigned&) const;</tt>
 *  where the first parameter is the global index and the other two are the
 *  output pair (localIdx, rank). return true if successful, false if
 *  global index is not part of the grid
 *  - <tt> MPI_Comm %communicator() const;</tt>  returns the communicator
 *  to use in the gatherscatter
 *  - <tt> local_size(); </tt> return the local vector size
 * @sa basictopology the MPI %grids defined in Level 3 can all be used as a
 * ConversionPolicy
 */
template<class ConversionPolicy>
std::map<int, thrust::host_vector<int>> gIdx2unique_idx(
    const thrust::host_vector<int>& globalIndexMap,
    thrust::host_vector<int>& bufferIdx, // may alias globalIndexMap
    const ConversionPolicy& p)
{
    // TODO update docu on local_size() ( if we don't scatter we don't need it)
    thrust::host_vector<std::array<int,2>> gIdx( globalIndexMap.size());
    bool success = true;
    for(unsigned i=0; i<gIdx.size(); i++)
        if( !p.global2localIdx(globalIndexMap[i],
                    gIdx[i][1], gIdx[i][0]) )
            success = false;

    assert( success);
    return gIdx2unique_idx( gIdx, bufferIdx);
}
} // namespace dg
