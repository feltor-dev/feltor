#pragma once

namespace dg
{

template <class Topology>
struct TopologyTraits{
    typedef typename Topology::memory_category memory_category; //!<either shared or distributed
    typedef typename Topology::value_type value_type; //!< value type
};

//memory categories
struct MPITag{}; //!< distributed memory system
struct SharedTag{}; //!<  shared memory system

}//namespace dg
