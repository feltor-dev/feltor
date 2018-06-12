#pragma once

namespace dg
{

template <class Topology>
struct TopologyTraits{
    typedef typename Topology::memory_category memory_category; //!<either shared or distributed
    typedef typename Topology::dimensionality dimensionality; //!< two-dimensional or three-dimensional
    typedef typename Topology::value_type value_type; //!< value type
};

//memory categories
struct MPITag{}; //!< distributed memory system
struct SharedTag{}; //!<  shared memory system

//dimensionality
struct OneDimensionalTag{}; //!< 1d
struct TwoDimensionalTag{}; //!< 2d
struct ThreeDimensionalTag{}; //!< 3d

}//namespace dg
