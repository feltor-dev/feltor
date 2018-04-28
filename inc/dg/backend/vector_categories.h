#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

namespace dg{

//here we introduce the concept of data access

///@addtogroup dispatch
///@{
/**
 * @brief Vector Tag base class, indicates the basic Vector/container concept
 *
 * The vector tag indicates how the data in the vector has to be accessed. For example
 * how do we get the pointer to the first element? Is there a contiguous chunk of memory
 * or is it a Vector of Vectors?
 * @note in any case we assume that the class is copyable/assignable and has a \c size and a \c swap member function
 * @note \c dg::VectorTraits<Vector> has member typedefs \c value_type, \c execution_policy, \c vector_category
 */
struct AnyVectorTag{};
///@}

/**
 * @brief Indicate a contiguous chunk of shared memory
 *
 * With this tag a class promises that the data it holds lies in a contiguous chunk that
 * can be traversed knowing the pointer to its first element. Sub-Tags specify
 * how this pointer can be accessed by an algorithm, how it can be resized
 * and how information like size can be retrieved.
 */
struct SharedVectorTag  : public AnyVectorTag {};
/**
 * @brief A distributed vector contains a data container and a MPI communicator
 *
 * This tag indicates that data is distributed among one or several processes.
 * An MPI Vector is assumed to be composed of a data container together with an MPI Communicator.
 * In fact, the currently only class with this tag is the \c MPI_Vector class.
 *
 * @note This is a recursive tag in the sense that classes must provide a typedef \c container_type, for which the \c VectorTraits must be specialized
 * @see MPI_Vector, mpi_structures
 */
struct MPIVectorTag     : public AnyVectorTag {};

/**
 * @brief This tag indicates composition/recursion.
 *
 * This Tag indicates that a class is composed of an array of containers, i.e. a container of containers.
 * We assume that the bracket operator [] is defined to access the inner elements.
 * @note The class must typedef \c value_type and VectorTraits must be specialized for this type.
 * @note Examples are \c std::vector<T> and \c std::array<T,N> where T is a non-primitive data type and N is the size of the array
 */
struct VectorVectorTag  : public AnyVectorTag {};

struct ArrayVectorTag   : public VectorVectorTag{}; //!< \c std::array of containers

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
struct StdArrayTag      : public ThrustVectorTag {}; //!< \c std::array< primitive_type>

}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
