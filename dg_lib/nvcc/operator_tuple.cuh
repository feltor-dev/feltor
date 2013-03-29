#ifndef _DG_OPERATOR_TUPLE_
#define _DG_OPERATOR_TUPLE_

#include <thrust/tuple.h>
#include "operators.cuh"
#include "matrix_traits.h"

namespace dg{

template< class Op>
struct MatrixTraits< thrust::tuple<Op, Op> > 
{
    typedef typename Op::value_type value_type;
    typedef typename Op::matrix_type operand_type;
    typedef OperatorMatrixTag matrix_category;
};



}//namespace dg

#endif //_DG_OPERATOR_TUPLE_
