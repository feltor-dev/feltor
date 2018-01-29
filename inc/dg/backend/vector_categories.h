#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

namespace dg{

//here we introduce the concept of data access

///@addtogroup dispatch
///@{
struct AnyVectorTag{}; //!< Vector Tag base class
///@}

struct SharedVectorTag  : public AnyVectorTag {};   //!< vectors on shared memory
struct MPIVectorTag     : public AnyVectorTag {};   //!< a distributed MPI vector, contains a shared vector

struct VectorVectorTag  : public AnyVectorTag {};   //!< container of containers (either Shared or MPI Vectors
struct ArrayVectorTag   : public VectorVectorTag{}; //!< std::array of containers

/**
 * @brief Indicate that thrust - like members are available
 *
 * In detail these must be
 *  - resize()
 *  - size()
 *  - data()
 *  - begin()
 *  - end()
 */
struct ThrustVectorTag  : public SharedVectorTag {};
struct CuspVectorTag    : public ThrustVectorTag {}; //!< special tag for cusp arrays
struct StdArrayTag      : public ThrustVectorTag {}; //!< std::array< primitive_type>

}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
