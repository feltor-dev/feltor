#ifndef _DG_MATRIX_CATEGORIES_
#define _DG_MATRIX_CATEGORIES_

namespace dg{


struct CuspMatrixTag {};

struct DiagonalPreconditionerTag {};

struct IdentityTag : public DiagonalPreconditionerTag {};

struct OperatorMatrixTag {};

struct OperatorTupleTag {};

}//namespace dg

#endif //_DG_MATRIX_CATEGORIES_
