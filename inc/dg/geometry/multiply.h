#pragma once

#include "operator.h"
#include "dg/functors.h"
#include "dg/blas1.h"
#include "tensor.h"

namespace dg
{
///@brief Utility functions used in connection with the SparseTensor class
namespace tensor
{

///@addtogroup geometry
///@{

/**
 * @brief Scale tensor with a ContainerType
 *
 * Computes \f$ t^{ij} = \mu t^{ij}\f$
 * @copydoc hide_ContainerType
 * @param t input (contains result on output)
 * @param mu all elements in t are scaled with mu
 */
template<class ContainerType0, class ContainerType1>
void scal( SparseTensor<ContainerType0>& t, const ContainerType1& mu)
{
    unsigned size=t.values().size();
    for( unsigned i=0; i<size; i++)
        dg::blas1::pointwiseDot( mu, t.values()[i], t.values()[i]);
}

///@cond
namespace detail
{
template<class value_type>
struct Multiply{
    DG_DEVICE
    void operator() ( value_type t00, value_type t01,
                      value_type t10, value_type t11,
                      value_type in0, value_type in1,
                      value_type& out0, value_type& out1) const
    {
        value_type tmp0 = t00*in0 + t01*in1;
        value_type tmp1 = t10*in0 + t11*in1;
        out1 = tmp1;
        out0 = tmp0;
    }
    DG_DEVICE
    void operator() ( value_type t00, value_type t01, value_type t02,
                      value_type t10, value_type t11, value_type t12,
                      value_type t20, value_type t21, value_type t22,
                      value_type in0, value_type in1, value_type in2,
                      value_type& out0, value_type& out1, value_type& out2) const
    {
        value_type tmp0 = t00*in0 + t01*in1 + t02*in2;
        value_type tmp1 = t10*in0 + t11*in1 + t12*in2;
        value_type tmp2 = t20*in0 + t21*in1 + t22*in2;
        out2 = tmp2;
        out1 = tmp1;
        out0 = tmp0;
    }
};
template<class value_type>
struct Determinant
{
    DG_DEVICE
    value_type operator()( value_type in) const{
        return 1./sqrt(in);
    }
    DG_DEVICE
    value_type operator() ( value_type t00, value_type t01,
                            value_type t10, value_type t11) const
    {
        return t00*t11 - t10*t01;
    }
    DG_DEVICE
    value_type operator() ( value_type t00, value_type t01, value_type t02,
                            value_type t10, value_type t11, value_type t12,
                            value_type t20, value_type t21, value_type t22) const
    {
        return t00*this->operator()(t11, t12, t21, t22)
              -t01*this->operator()(t10, t12, t20, t22)
              +t02*this->operator()(t10, t11, t20, t21);
    }
};
}//namespace detail
///@endcond
/**
 * @brief Multiply a tensor with a vector in 2d
 *
 * Compute \f$ w^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the first two dimensions (ignores the 3rd dimension in t)
 * @copydoc hide_ContainerType
 * @param t input Tensor
 * @param in0 (input) first component    (may alias out0)
 * @param in1 (input) second component   (may alias out1)
 * @param out0 (output) first component  (may alias in0)
 * @param out1 (output) second component (may alias in1)
 * @note Currently required memops:
         - 6 reads + 2 writes; (-2 read if aliases are used)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with the appropriate functor
 */
template<class ContainerType>
void multiply2d( const SparseTensor<ContainerType>& t, const ContainerType& in0, const ContainerType& in1, ContainerType& out0, ContainerType& out1)
{
    dg::blas1::subroutine( detail::Multiply<get_value_type<ContainerType>>(),
                           t.value(0,0), t.value(0,1),
                           t.value(1,0), t.value(1,1),
                           in0,  in1,
                           out0, out1);
}

/**
 * @brief Multiply a tensor with a vector in 2d
 *
 * Compute \f$ w^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2,3\}\f$
 * @copydoc hide_ContainerType
 * @param t input Tensor
 * @param in0 (input)  first component  (may alias out0)
 * @param in1 (input)  second component (may alias out1)
 * @param in2 (input)  third component  (may alias out2)
 * @param out0 (output)  first component  (may alias in0)
 * @param out1 (output)  second component (may alias in1)
 * @param out2 (output)  third component  (may alias in2)
 */
template<class ContainerType>
void multiply3d( const SparseTensor<ContainerType>& t, const ContainerType& in0, const ContainerType& in1, const ContainerType& in2, ContainerType& out0, ContainerType& out1, ContainerType& out2)
{
    dg::blas1::subroutine( detail::Multiply<get_value_type<ContainerType>>(),
                           t.value(0,0), t.value(0,1), t.value(0,2),
                           t.value(1,0), t.value(1,1), t.value(1,2),
                           t.value(2,0), t.value(2,1), t.value(2,2),
                           in0, in1, in2,
                           out0, out1, out2);
}

/**
* @brief Compute the determinant of a 3d tensor
* @copydoc hide_ContainerType
* @param t the input tensor
* @return the determinant of t
*/
template<class ContainerType>
ContainerType determinant( const SparseTensor<ContainerType>& t)
{
    ContainerType det = t.value(0,0);
    dg::blas1::evaluate( det, dg::equals(), detail::Determinant<get_value_type<ContainerType>>(),
                           t.value(0,0), t.value(0,1), t.value(0,2),
                           t.value(1,0), t.value(1,1), t.value(1,2),
                           t.value(2,0), t.value(2,1), t.value(2,2));
    return det;
}
/**
* @brief Compute the minor determinant of a tensor
* @copydoc hide_ContainerType
* @param t the input tensor
* @return the upper left minor determinant of t
*/
template<class ContainerType>
ContainerType determinant2d( const SparseTensor<ContainerType>& t)
{
    ContainerType det = t.value(0,0);
    dg::blas1::evaluate( det, dg::equals(), detail::Determinant<get_value_type<ContainerType>>(),
                           t.value(0,0), t.value(0,1),
                           t.value(1,0), t.value(1,1));
    return det;
}

/**
 * @brief Compute the sqrt of the inverse minor determinant of a tensor
 *
 * This is a convenience function that is equivalent to
 * @code
    ContainerType vol=determinant2d(t);
    dg::blas1::transform(vol, vol, dg::INVERT<>());
    dg::blas1::transform(vol, vol, dg::SQRT<>());
    @endcode
 * @copydoc hide_ContainerType
 * @param t the input tensor
 * @return the inverse square root of the determinant of \c t
 */
template<class ContainerType>
ContainerType volume2d( const SparseTensor<ContainerType>& t)
{
    ContainerType vol=determinant2d(t);
    dg::blas1::transform(vol, vol, detail::Determinant<get_value_type<ContainerType>>());
    return vol;
}
/**
 * @brief Compute the sqrt of the inverse determinant of a 3d tensor
 *
 * This is a convenience function that is equivalent to
 * @code
    ContainerType vol=determinant(t);
    dg::blas1::transform(vol, vol, dg::INVERT<>());
    dg::blas1::transform(vol, vol, dg::SQRT<>());
    @endcode
 * @copydoc hide_ContainerType
 * @param t the input tensor
 * @return the inverse square root of the determinant of \c t
 */
template<class ContainerType>
ContainerType volume( const SparseTensor<ContainerType>& t)
{
    ContainerType vol=determinant(t);
    dg::blas1::transform(vol, vol, detail::Determinant<get_value_type<ContainerType>>());
    return vol;
}

///@}

}//namespace tensor
}//namespace dg
