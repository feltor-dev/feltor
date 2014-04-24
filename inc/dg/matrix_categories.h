#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{


struct CuspMatrixTag {};

struct DiagonalPreconditionerTag {};

struct IdentityTag : public DiagonalPreconditionerTag {};

struct StdMatrixTag{};

struct OperatorMatrixTag {};

struct SelfMadeMatrixTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
