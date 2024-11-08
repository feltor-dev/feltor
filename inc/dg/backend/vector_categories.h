#ifndef _DG_VECTOR_CATEGORIES_
#define _DG_VECTOR_CATEGORIES_

#include "matrix_categories.h"

namespace dg{

//here we introduce the concept of data access

/**
 * @brief Vector Tag base class, indicates the basic Vector/container concept
 *
 * The vector tag has two functions.
 First, it describes how the data in a Vector type is layed out in memory. We distinguish between a simple, contiguous chunk of data in a shared memory system (\c dg::SharedVectorTag), a dataset that is
part of a larger dataset on a distributed memory system (\c dg::MPIVectorTag), and
a dataset that consists of a number of subsets (\c dg::RecursiveVectorTag).
Both the MPIVectorTag and the RecursiveVectorTag allow recursion, that is
for example a RecursiveVector can consist itself of many shared vectors or of many
RecursiveVector again. The innermost type must always be a shared vector however.
  The second function of the Vector tag is to describe how the data in the vector has to be accessed.For example
 * how do we get the pointer to the first element, the size, or how to access the MPI communicator? This is described in Derived Tags from the fundamental
Tags, e.g. the \c ThrustVectorTag.
 * @note in any case we assume that the class has a default constructor, is copyable/assignable and has a \c size and a \c swap member function
 * @note <tt> dg::TensorTraits<Vector> </tt>has member typedefs \c value_type, \c execution_policy, \c tensor_category
 * @note any vector can serve as a diagonal matrix
 * @see \ref dispatch
 */
struct AnyVectorTag : public AnyMatrixTag{};

/**
 * @brief Indicate a contiguous chunk of shared memory
 *
 * With this tag a class promises that the data it holds lies in a contiguous chunk that
 * can be traversed knowing the pointer to its first element. Sub-Tags
 * indicate addtional functionality like data resize.
 * @note We assume a class with this Tag has the following methods
 *  - \c size() returns the size (in number of elements) of the contiguous data
 *  - \c data() returns a pointer (or something for which the \c thrust::pointer_traits are specialized) to the first element of the contiguous data.
 *  The return type must be convertible to <tt> (const) value_type* </tt>
 */
struct SharedVectorTag  : public AnyVectorTag {};
/**
 * @brief A distributed vector contains a data container and a MPI communicator
 *
 * This tag indicates that data is distributed among one or several processes.
 * An MPI Vector is assumed to be composed of a data container together with an MPI Communicator.
 * In fact, the currently only class with this tag is the \c MPI_Vector class.
 *
 * @note This is a recursive tag in the sense that classes must provide a typedef \c container_type, for which the \c dg::TensorTraits must be specialized
 * @see dg::MPI_Vector, mpi_structures
 */
struct MPIVectorTag     : public AnyVectorTag {};

/**
 * @brief This tag indicates composition/recursion.
 *
 * This Tag indicates that a class is composed of an array of containers, i.e. a container of containers.
 * We assume that the bracket <tt> operator[] </tt> is defined to access the inner elements and the \c size() function returns the number of elements.
 * @note The class must typedef \c value_type (the "inner" type that is returned by the bracket operator) and <tt> dg::TensorTraits<value_type> </tt> must be specialized for this type.
 * @note Examples are \c std::vector<T> and \c std::array<T,N> where T is the inner type and N is the size of the array
 */
struct RecursiveVectorTag  : public AnyVectorTag {};
//struct RecursiveScalarTag : public RecursiveVectorTag {};
struct ArrayVectorTag  : public RecursiveVectorTag {};
struct StdMapTag : public RecursiveVectorTag{};

struct ArrayScalarTag : public SharedVectorTag {};
/**
 * @brief Indicate thrust/std - like behaviour
 *
 * There must be the typedefs
 * - \c iterator
 * - \c const_iterator
 * - \c pointer
 * - \c const_pointer
 * An instance can be constructed by
 *  - iterators \c (begin, end)
 * The member functions contan at least
 *  - \c resize()  resize the vector
 *  - \c size() returns the number of elements
 *  - \c data() returns \c pointer to the underlying array
 *  - \c begin() returns \c iterator to the beginning
 *  - \c cbegin() returns \c const_iterator to the beginning
 *  - \c end() return an \c iterator to the end
 *  - \c cend() returns a \c const_iterator to the end
 *  @note \c thrust::host_vector and \c thrust::device_vector meet these requirements
 */
struct ThrustVectorTag  : public ArrayScalarTag {};
struct CuspVectorTag    : public ThrustVectorTag {}; //!< special tag for cusp arrays

}//namespace dg

#endif //_DG_VECTOR_CATEGORIES_
