#ifndef _DG_VECTOR_TRAITS_
#define _DG_VECTOR_TRAITS_

namespace dg{

template< class Vector>
struct VectorTraits {
    typedef typename Vector::value_type value_type;
    typedef ThrustVectorTag vector_category; //default is a ThrustVector
};

}//namespace dg

#endif //_DG_VECTOR_TRAITS_
