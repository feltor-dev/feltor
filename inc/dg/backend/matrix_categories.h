#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{

///@addtogroup dispatch
///@{
struct AnyMatrixTag{};//!<default
/// indicates that the \c Matrix type has the \c void \c symv( const Vector&, Vector& ) member function.
struct SelfMadeMatrixTag {};
///@}

//normal matrices
struct CuspMatrixTag: public AnyMatrixTag {};


struct MPIMatrixTag: public AnyMatrixTag {};


///Indicate that a thrust vector should act as a diagonal matrix
struct ThrustMatrixTag: public AnyMatrixTag {};
///Indicate that a cusp array should act as a diagonal matrix
struct CuspPreconTag: public ThrustMatrixTag {};
///Indicate that a MPI_Vector should act as a diagonal matrix
struct MPIPreconTag: public AnyMatrixTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
