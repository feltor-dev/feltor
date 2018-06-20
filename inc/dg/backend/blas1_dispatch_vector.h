#ifndef _DG_BLAS_STD_VECTOR_
#define _DG_BLAS_STD_VECTOR_

#ifdef DG_DEBUG
#include <cassert>
#endif //DG_DEBUG

#include <vector>
#include <array>
#include "blas1_dispatch_shared.h"
#include "vector_categories.h"
#include "tensor_traits.h"
#ifdef _OPENMP
#include <omp.h>
#endif //_OPENMP

///@cond
namespace dg
{
namespace blas1
{

template<class to_ContainerType, class from_ContainerType>
inline to_ContainerType transfer( const from_ContainerType& src);

namespace detail
{

template<class To, class From>
To doTransfer( const From& src, ArrayVectorTag, AnyVectorTag)
{
    To t;
    using inner_vector = typename To::value_type;
    for (unsigned i=0; i<t.size(); i++)
        t[i] = dg::blas1::transfer<inner_vector>(src);
    return t;
}

template< class Vector, class Vector2>
inline std::vector<int64_t> doDot_superacc( const Vector& x1, const Vector2& x2, VectorVectorTag)
{
    static_assert( std::is_base_of<VectorVectorTag,
        get_tensor_category<Vector2>>::value,
        "All data layouts must derive from the same vector category (VectorVectorTag in this case)!");
#ifdef DG_DEBUG
    assert( !x1.empty());
    assert( x1.size() == x2.size() );
#endif //DG_DEBUG
    std::vector<std::vector<int64_t>> acc( x1.size());
    for( unsigned i=0; i<x1.size(); i++)
        acc[i] = doDot_superacc( x1[i], x2[i], get_tensor_category<typename Vector::value_type>());
    for( unsigned i=1; i<x1.size(); i++)
    {
        int imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[0][0]), imin, imax);
        imin = exblas::IMIN, imax = exblas::IMAX;
        exblas::cpu::Normalize( &(acc[i][0]), imin, imax);
        for( int k=exblas::IMIN; k<exblas::IMAX; k++)
            acc[0][k] += acc[i][k];
    }
    return acc[0];
}
template<class Vector, class Vector2>
get_value_type<Vector> doDot( const Vector& x, const Vector2& y, VectorVectorTag)
{
    std::vector<int64_t> acc = doDot_superacc( x,y,VectorVectorTag());
    return exblas::cpu::Round(acc.data());
}

////we need to distinguish between Scalars and Vectors
//template<class T, class Enable = void>
//struct inner_type_impl;
//
//template<class T, typename std::is_base_of< VectorVectorTag, get_tensor_category<T>::value >
//struct inner_type_impl{
//    using type = typename std::decay<T>::type::value_type;
//    using reference = typename std::conditional<std::is_const<T>::value, const type&, type& >::type;
//};
//template<class T, typename std::is_base_of< AnyScalarTag, get_tensor_category<T>::value>
//struct inner_type_impl{
//    using type = T;
//    using reference = T&;
//};
//template<class T>
//using inner_ref = typename inner_type_impl<T>::reference;

template<class T>
auto get_element( T&& v, unsigned i, AnyVectorTag)-> decltype(v[i]){
    return v[i];
}
template<class T>
T get_element( T&& v, unsigned i, AnyScalarTag){
    return v;
}
template<class T>
auto get_element( T&& v, unsigned i ) -> decltype( get_element( std::forward<T>(v), i, get_tensor_category<T>()) ) {
    return get_element( std::forward<T>(v), i, get_tensor_category<T>());
}
#ifdef _OPENMP
//omp tag implementation
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( VectorVectorTag, OmpTag, Subroutine f, container&& x, Containers&&... xs)
{
    //using inner_container = typename std::decay<container>::type::value_type;
    if( !omp_in_parallel())//to catch recursive calls
    {
        #pragma omp parallel
        {
            for( unsigned i=0; i<x.size(); i++) {
                dg::blas1::subroutine( f, get_element(std::forward<container>(x),i), get_element(std::forward<Containers>(xs),i)...);
            }
        }
    }
    else //we are already in a parallel omp region
        for( unsigned i=0; i<x.size(); i++) {
            dg::blas1::subroutine( f, get_element(std::forward<container>(x),i), get_element(std::forward<Containers>(xs),i)...);
        }
}
#endif //_OPENMP



//any tag implementation
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine_dispatch( VectorVectorTag, AnyPolicyTag, Subroutine f, container&& x, Containers&&... xs)
{
    for( unsigned i=0; i<x.size(); i++) {
        dg::blas1::subroutine( f, get_element(std::forward<container>(x),i), get_element(std::forward<Containers>(xs),i)...);
    }
}

//dispatch
template< class Subroutine, class container, class ...Containers>
inline void doSubroutine( VectorVectorTag, Subroutine f, container&& x, Containers&&... xs)
{
    static_assert( all_true<std::is_base_of<VectorVectorTag,
        get_tensor_category<Containers>>::value...>::value,
        "All data layouts must derive from the same vector category (VectorVectorTag in this case)!");
    static_assert( all_true<std::is_same<get_execution_policy<container>,
        get_execution_policy<Containers> >::value...>::value,
        "All data layouts must share the same execution policy!");
#ifdef DG_DEBUG
    //is this possible?
    //assert( !x.empty());
    //assert( x.size() == xs.size() );
#endif //DG_DEBUG
    doSubroutine_dispatch( VectorVectorTag(), get_execution_policy<container>(), f, std::forward<container>( x), std::forward<Containers>( xs)...);
}

} //namespace detail
} //namespace blas1
} //namespace dg
///@endcond

#endif //_DG_BLAS_STD_VECTOR_
