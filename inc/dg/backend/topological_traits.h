#pragma once

namespace dg
{

///@cond
template <class Topology>
struct TopologyTraits{
    typedef typename Topology::memory_category memory_category; //either shared or distributed
    typedef typename Topology::dimensionality dimensionality; //either shared or distributed
};
///@endcond

//memory categories
struct MPITag{}; //!< distributed memory  system
struct SharedTag{}; //!<  shared memory system

//dimensionality 
struct OneDimensionalTag{}; //!< 1d
struct TwoDimensionalTag{}; //!< 2d
struct ThreeDimensionalTag{}; //!< 3d

}//namespace dg
