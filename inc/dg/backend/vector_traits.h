#ifndef _DG_VECTOR_TRAITS_
#define _DG_VECTOR_TRAITS_

#include <vector>
#include "vector_categories.h"

namespace dg{

template< class Vector>
struct VectorTraits {
    typedef typename Vector::value_type value_type;
    typedef ThrustVectorTag vector_category; //default is a ThrustVector
};

template< class Vector>
struct VectorTraits<std::vector<Vector> >{
    typedef typename VectorTraits<Vector>::value_type value_type;
    typedef StdVectorTag vector_category;
};
//template< class Vector>
//struct VectorTraits<std::vector<Vector*> >{
//    typedef typename Vector::value_type value_type;
//    typedef StdVectorPointerTag vector_category;
//};

}//namespace dg

#endif //_DG_VECTOR_TRAITS_
