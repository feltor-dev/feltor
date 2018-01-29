#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

namespace dg{

//here we introduce the concept of data access

///@addtogroup dispatch
///@{
/**
 * @brief Vector Tag base class
 *
 * The vector tag indicates how the data in the vector has to be accessed. For example
 * how do we get the pointer to the first element? Is there a contiguous chunk of memory
 * or is it a Vector of Vectors? 
 * @note in any case we assume that the class is copyable/assignable and has a \c swap member function
 */
struct AnyVectorTag{}; //!< Vector Tag base class
///@}

struct SharedVectorTag  : public AnyVectorTag {};   //!< vectors on shared memory
/**
 * @brief a distributed MPI vector, contains a shared vector
 * 
 * Must have a typedef \c container_type, which must have the SharedVectorTag
 */
struct MPIVectorTag     : public AnyVectorTag {};

/**
 * @brief A container of containers  (either Shared or MPI Vectors)
 *
 * for example a std::vector<container>
 * @note Can be used recursively, for example a std::vector<std::vector<container>>
 */
struct VectorVectorTag  : public AnyVectorTag {};
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
