#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

#include "matrix_categories.h"

namespace dg{

//here we introduce the concept of data access

///@addtogroup dispatch
///@{
/**
 * @brief Vector Tag base class, indicates the basic Vector/container concept
 *
 * The vector tag has three functions.
First, it indicates the fundamental datatype a vector class contains (typically a double).
 Second, it describes how the data in a Vector type is layed out in memory. We distinguish between a simple, contiguous chunk of data in a shared memory system (dg::SharedVectorTag), a dataset that is
part of a larger dataset on a distributed memory system (dg::MPIVectorTag), and
a dataset that consists of a number of subsets (dg::VectorVectorTag).
Both the MPIVectorTag and the VectorVectorTag allow recursion, that is
for example a VectorVector can consist itself of many shared vectors or of many
VectorVector again. The innermost type must always be a shared vector however.
  The third function of the Vector tag is to describe how the data in the vector has to be accessed.For example
 * how do we get the pointer to the first element, the size, or how to access the MPI communicator? This is described in Derived Tags from the fundamental
Tags, e.g. the \c ThrustVectorTag.
 * @note in any case we assume that the class has a default constructor, is copyable/assignable and has a \c size and a \c swap member function
 * @note <tt> dg::TensorTraits<Vector> </tt>has member typedefs \c value_type, \c execution_policy, \c tensor_category
@attention Two instances of two different Vector types are usable in the same \c blas function if
  - their tags derive from the same VectorTag class except AnyVectorTag ( the data size
    and type, layout and communicator, is the same, but the access to it may differ)
  - their execution policy tag is exactly the same

@note any vector can serve as a diagonal matrix
 */
struct AnyVectorTag : public AnyMatrixTag{};
///@}

/**
 * @brief Indicate a contiguous chunk of shared memory
 *
 * With this tag a class promises that the data it holds lies in a contiguous chunk that
 * can be traversed knowing the pointer to its first element. Sub-Tags
 * indicate addtional functionality like data resize.
 * @note We assume a class with this Tag has the following methods
 *  - size() returns the size (in number of elements) of the contiguous data
 *  - data() returns a pointer (or pointer-like class that has a get() method to access the raw pointer) to the first element of the contiguous data
 */
struct SharedVectorTag  : public AnyVectorTag {};
/**
 * @brief A distributed vector contains a data container and a MPI communicator
 *
 * This tag indicates that data is distributed among one or several processes.
 * An MPI Vector is assumed to be composed of a data container together with an MPI Communicator.
 * In fact, the currently only class with this tag is the \c MPI_Vector class.
 *
 * @note This is a recursive tag in the sense that classes must provide a typedef \c container_type, for which the \c TensorTraits must be specialized
 * @see MPI_Vector, mpi_structures
 */
struct MPIVectorTag     : public AnyVectorTag {};

/**
 * @brief This tag indicates composition/recursion.
 *
 * This Tag indicates that a class is composed of an array of containers, i.e. a container of containers.
 * We assume that the bracket \c operator[] is defined to access the inner elements and the \c size() function returns the number of elements.
 * @note The class must typedef \c value_type and TensorTraits must be specialized for this type.
 * @note Examples are \c std::vector<T> and \c std::array<T,N> where T is a non-primitive data type and N is the size of the array
 */
struct VectorVectorTag  : public AnyVectorTag {};

struct ArrayVectorTag   : public VectorVectorTag{}; //!< \c std::array of containers

/**
 * @brief Indicate that thrust - like member functions are available
 *
 * In detail these must be
 *  - resize()
 *  - size()
 *  - data()
 *  - begin()
 *  - end()
 *  - can be constructed from iterators (begin, end)
 */
struct ThrustVectorTag  : public SharedVectorTag {};
struct CuspVectorTag    : public ThrustVectorTag {}; //!< special tag for cusp arrays
struct StdArrayTag      : public ThrustVectorTag {}; //!< <tt> std::array< primitive_type> </tt>

}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
