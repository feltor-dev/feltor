#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{

struct AnyMatrixTag{};

//normal matrices
struct CuspMatrixTag: public AnyMatrixTag {};

struct SelfMadeMatrixTag {}; //A selfmade matrix can with any Vector

struct dx_matrixTag: public AnyMatrixTag {}; // Small banded block matrix for derivatives

struct MPIMatrixTag: public AnyMatrixTag {};

//preconditioners
struct ThrustMatrixTag: public AnyMatrixTag {};

struct CuspPreconTag: public AnyMatrixTag {};
struct MPIPreconTag: public AnyMatrixTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
