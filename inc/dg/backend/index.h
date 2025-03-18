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

template<class host_vector >
Unique<typename host_vector::value_type> find_unique_stable_sort(
    const host_vector& unsorted_elements)
{
    using T = typename host_vector::value_type;
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

template<class host_vector >
Unique<typename host_vector::value_type> find_unique_order_preserving(
    const host_vector& unsorted_elements)
{
    using T = typename host_vector::value_type;
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
inline bool operator==( const dg::detail::MsgChunk& a, const dg::detail::MsgChunk& b)
{
    return a.idx == b.idx and a.size == b.size;
}

inline thrust::host_vector<MsgChunk>
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

// for every size != 0 in sizes add map[idx] = size;
inline std::map<int,int> make_size_map( const thrust::host_vector<int>& sizes)
{
    std::map<int,int> out;
    for( unsigned u=0; u<sizes.size(); u++)
    {
        if ( sizes[u] != 0)
            out[u] = sizes[u];
    }
    return out;
}

////////////////// Functionality for packaging mpi messages
// We pack every message into a vector of primitive type
//
//forward declare
template<class T>
auto flatten ( const T& t);

inline thrust::host_vector<int> flatten ( const MsgChunk& t)
{
    thrust::host_vector<int> flat(2);
    flat[0] = t.idx, flat[1] = t.size;
    return flat;
}

template<class T>
thrust::host_vector<T> flatten ( const T& t, dg::AnyScalarTag)
{
    return thrust::host_vector<T>(1, t);
}
template<class ContainerType>
auto flatten (
    const ContainerType& t, dg::AnyVectorTag)
{
    decltype( flatten(t[0])) flat;
    for( unsigned u=0; u<t.size(); u++)
    {
        auto value = flatten(t[u]);
        flat.insert(flat.end(), value.begin(), value.end());
    }
    return flat;
}

template<class T>
auto flatten ( const T& t) // return type is  thrust::host_vector<Type>
{
    using tensor_tag = dg::get_tensor_category<T>;
    return flatten( t, tensor_tag());
}

// 1. flatten values -> a map of flattened type
template<class Message>
auto flatten_values( const std::map<int,Message>& idx_map // map is sorted automatically
)
{
    using vector_type = decltype( flatten( idx_map.at(0) ));
    std::map<int,vector_type> flat;
    for( auto& idx : idx_map)
    {
        flat[idx.first]  = flatten(idx.second);
    }
    return flat;
}

// 2. flatten the map (keys get lost)
template<class vector>
vector flatten_map(
    const std::map<int,vector>& idx_map // map is sorted automatically
    )
{
    vector flat;
    for( auto& idx : idx_map)
        flat.insert(flat.end(), idx.second.begin(), idx.second.end());
    return flat;
}

////////////////// Functionality for unpacking mpi messages
// unpack a vector of primitive types into original data type

inline void make_target(
    const thrust::host_vector<int>& src, MsgChunk& target)
{
    assert( src.size() == 2);
    target = {src[0], src[1]};
}
inline void make_target(
    const thrust::host_vector<int>& src, thrust::host_vector<MsgChunk>& target)
{
    assert( src.size() % 2 == 0);
    target.clear();
    for( unsigned u=0; u<src.size()/2; u++)
        target.push_back( { src[2*u], src[2*u+1]});
}
template<class T, size_t N>
void make_target(
    const thrust::host_vector<T>& src, std::array<T,N>& target)
{
    assert( src.size() == N);
    thrust::copy( src.begin(), src.end(), target.begin());
}
template<class T, size_t N>
void make_target(
    const thrust::host_vector<T>& src, thrust::host_vector<std::array<T,N>>& target)
{
    assert( src.size() % N == 0);
    target.clear();
    for( unsigned u=0; u<src.size()/N; u++)
    {
        std::array<T,N> t;
        thrust::copy( src.begin() + N*u, src.begin() + N*(u+1), t.begin());
        target.push_back( t);
    }
}
template<class Target, class Source>
void make_target( const thrust::host_vector<Source>& src, Target& t, AnyScalarTag)
{
    assert( src.size() == 1);
    t = src[0];
}
template<class Target, class Source>
void make_target( const thrust::host_vector<Source>& src, Target& t, AnyVectorTag)
{
    t = src;
}
template<class Target, class Source>
void make_target( const thrust::host_vector<Source>& src, Target& t)
{
    using tensor_tag = dg::get_tensor_category<Target>;
    make_target(src, t, tensor_tag()  );
}

template<class Target, class T>
std::map<int, Target> make_map_t(
    const thrust::host_vector<T>& flat_map,
    const std::map<int,int>& size_map // key and chunk size of idx_map
    )
{
    // 1. unflatten vector
    std::map<int, Target> map;
    unsigned start = 0;
    for( auto& size : size_map)
    {
        if( size.second != 0)
        {
            thrust::host_vector<T> partial(
                flat_map.begin()+start, flat_map.begin() + start + size.second);
            start += size.second;
            make_target( partial, map[size.first]);
        }
    }
    // 2. Convert each message into target type
    return map;
}




///////////////////////////////////////////////Get sizes of flat message/////////////
template<class T>
std::map<int,int> get_size_map( const std::map<int, thrust::host_vector<T>>& idx_map)
{
    std::map<int,int> out;
    for( auto& idx : idx_map)
        out[idx.first] = idx.second.size();
    return out;
}




}// namespace detail
// TODO Maybe this can be public? Why would anyone call these functions directly?
// TODO Maybe be a MPIGather function?
/**
 * @brief Convert an unsorted and possible duplicate global index list to unique
 * stable_sorted by pid and duplicates map

 * bufferIdx gives
 * @note gIdx can be unsorted and contain duplicate entries
 * @param gIdx idx 0 is pid, idx 1 is localIndex on that pid
 * @param bufferIdx Same size as gIdx. Index into flattened map
*/
template<class ArrayVec = thrust::host_vector<std::array<int,2>>, class IntVec = thrust::host_vector<int>>
std::map<int, IntVec> gIdx2unique_idx(
    const ArrayVec& gIdx, // unsorted (cannot be map)
    IntVec& bufferIdx) // gIdx size, gather gIdx from flatten_map
{
    auto uni = detail::find_unique_order_preserving( gIdx);
    // get pids
    IntVec pids(uni.unique.size()), lIdx(pids);
    for( int i=0; i<(int)pids.size(); i++)
    {
        pids[i] = uni.unique[i][0];
        lIdx[i] = uni.unique[i][1]; // the local index
    }
    auto uni_pids = detail::find_unique_stable_sort( pids);
    bufferIdx = detail::combine_gather( uni.gather1,
                   detail::combine_gather( uni.gather2, uni_pids.gather1));
    // duplicate the sort on lIdx
    auto sorted_unique_gIdx = lIdx;
    thrust::scatter( lIdx.begin(), lIdx.end(),
             uni_pids.gather1.begin(), sorted_unique_gIdx.begin());
    // return map
    std::map<int,int> pids_howmany = detail::make_map( uni_pids.unique, uni_pids.howmany);
    return detail::make_map_t<IntVec>( sorted_unique_gIdx, pids_howmany);
}

template<class ConversionPolicy, class IntVec = thrust::host_vector<int>>
thrust::host_vector<std::array<int,2>> gIdx2gIdx( const IntVec& gIdx, const ConversionPolicy& p)
{
    thrust::host_vector<std::array<int,2>> arrayIdx( gIdx.size());
    for(unsigned i=0; i<gIdx.size(); i++)
        assert(p.global2localIdx(gIdx[i],
                    arrayIdx[i][1], arrayIdx[i][0]) );
    return arrayIdx;
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
template<class ConversionPolicy, class IntVec = thrust::host_vector<int>>
std::map<int, IntVec> gIdx2unique_idx(
    const IntVec& globalIndexMap,
    IntVec& bufferIdx, // may alias globalIndexMap
    const ConversionPolicy& p)
{
    // TODO update docu on local_size() ( if we don't scatter we don't need it)
    return gIdx2unique_idx( gIdx2gIdx(globalIndexMap), bufferIdx);
}
///@endcond
} // namespace dg
