#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{

///@addtogroup dispatch
///@{
struct AnyMatrixTag{};
/// indicates that the \c Matrix type has the <tt> void symv( const Vector&, Vector& ) </tt> member function.
struct SelfMadeMatrixTag {};
///@}


/// One of cusp's matrices
struct CuspMatrixTag: public AnyMatrixTag {};

/// indicate one of our mpi matrices
struct MPIMatrixTag: public AnyMatrixTag {};


///Indicate that \c TypeTraits exist for this type and it should act as a diagonal matrix
struct AnyVectorMatrixTag: public AnyMatrixTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
