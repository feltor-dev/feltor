#pragma once
#include "cusp/array1d.h"
#include "vector_traits.h"

namespace dg
{

template< class Vector>
struct VectorView: public cusp::array1d_view< get_pointer_type<Vector>>
{
    VectorView(){}
    VectorView(get_pointer_type<Vector> begin, get_pointer_type<Vector> end):
        Parent(begin,end){}
    VectorView( Vector& v):VectorView(thrust::raw_pointer_cast(v.data()),thrust::raw_pointer_cast(v.data()+v.size())){}
    get_pointer_type<Vector> data()const{return Parent::begin();}
    private:
    using Parent = cusp::array1d_view<get_pointer_type<Vector>>;
};

template<class Vector>
struct VectorTraits<VectorView<Vector>> {
    using value_type        = get_value_type<Vector>;
    using vector_category   = get_vector_category<Vector>;
    using execution_policy  = get_execution_policy<Vector>; 
};

}//namespace dg
