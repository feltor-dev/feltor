#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{

struct AnyMatrixTag{};

//normal matrices
struct CuspMatrixTag: public AnyMatrixTag {};

/// indicates that the \c Matrix type has the \c void \c symv( const Vector&, Vector& ) member function.
struct SelfMadeMatrixTag {};

struct MPIMatrixTag: public AnyMatrixTag {};

//preconditioners
struct ThrustMatrixTag: public AnyMatrixTag {};

struct CuspPreconTag: public ThrustMatrixTag {};
struct MPIPreconTag: public AnyMatrixTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
