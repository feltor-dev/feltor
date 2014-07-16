#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{

struct AnyMatrixTag{};

struct CuspMatrixTag: public AnyMatrixTag {};

struct StdMatrixTag{};
//struct StdMatrixPointerTag{};

struct ThrustMatrixTag: public AnyMatrixTag {};

struct SelfMadeMatrixTag: public AnyMatrixTag {};

struct MPIMatrixTag: public AnyMatrixTag {};

struct MPIPreconTag: public AnyMatrixTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
