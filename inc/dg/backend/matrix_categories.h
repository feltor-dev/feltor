#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{

///@addtogroup dispatch
///@{

///tensor_category base class
struct AnyMatrixTag{};

///Indicate that a type is not a tensor
struct NotATensorTag{};
///@}


//
/**
 * @brief Indicates that the type has a member function with the same name and interface (up to the matrix itself of course)
as the
corresponding \c blas2 member function, for example
<tt> void symv( const ContainerType1&, ContainerType2& ); </tt>

These members are then implemented freely, in particular other \c blas1 and \c blas2 functions can be used
 */
struct SelfMadeMatrixTag: public AnyMatrixTag {};

/// One of cusp's matrices, for these only the \c blas2 transfer and the symv( m,x,y) are implemented
struct CuspMatrixTag: public AnyMatrixTag {};
/// indicate one of our mpi matrices
struct MPIMatrixTag: public AnyMatrixTag {};
/// indicate our sparse block matrix format
struct SparseBlockMatrixTag: public AnyMatrixTag {};
/// indicate our sparse matrix format
struct SparseMatrixTag: public SparseBlockMatrixTag {};
/// indicate our dense matrix format
struct DenseMatrixTag: public AnyMatrixTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
