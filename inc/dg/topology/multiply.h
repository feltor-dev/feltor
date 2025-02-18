#pragma once

#include "operator.h"
#include "dg/functors.h"
#include "dg/blas1.h"
#include "tensor.h"

/*!@file
 *
 * Basic tensor functions (functions that involve sparse tensors)
 */

namespace dg
{
///@addtogroup variadic_subroutines
///@{

/// \f$ y_i \leftarrow \lambda T_{ij} x_i + \mu y_i\f$
template<class value_type>
struct TensorMultiply2d{
    DG_DEVICE
    void operator() (
              value_type lambda,
              value_type t00, value_type t01,
              value_type t10, value_type t11,
              value_type in0, value_type in1,
              value_type mu,
              value_type& out0, value_type& out1) const
    {
        value_type tmp0 = DG_FMA(t00,in0 , t01*in1);
        value_type tmp1 = DG_FMA(t10,in0 , t11*in1);
        value_type temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0, temp);
    }
};
/// \f$ y_i \leftarrow \lambda T_{ij} x_i + \mu y_i\f$
template<class value_type>
struct TensorMultiply3d{
    DG_DEVICE
    void operator() ( value_type lambda,
                      value_type t00, value_type t01, value_type t02,
                      value_type t10, value_type t11, value_type t12,
                      value_type t20, value_type t21, value_type t22,
                      value_type in0, value_type in1, value_type in2,
                      value_type mu,
                      value_type& out0, value_type& out1, value_type& out2) const
    {
        value_type tmp0 = DG_FMA( t00,in0 , (DG_FMA( t01,in1 , t02*in2)));
        value_type tmp1 = DG_FMA( t10,in0 , (DG_FMA( t11,in1 , t12*in2)));
        value_type tmp2 = DG_FMA( t20,in0 , (DG_FMA( t21,in1 , t22*in2)));
        value_type temp = out2*mu;
        out2 = DG_FMA( lambda, tmp2, temp);
        temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0, temp);
    }
};
/// \f$ y_i \leftarrow \lambda T^{-1}_{ij} x_i + \mu y_i\f$
template<class value_type>
struct InverseTensorMultiply2d{
    DG_DEVICE
    void operator() (  value_type lambda,
                       value_type t00, value_type t01,
                       value_type t10, value_type t11,
                       value_type in0, value_type in1,
        value_type mu, value_type& out0, value_type& out1) const
    {
        value_type dett = DG_FMA( t00,t11 , (-t10*t01));
        value_type tmp0 = DG_FMA( in0,t11 , (-in1*t01));
        value_type tmp1 = DG_FMA( t00,in1 , (-t10*in0));
        value_type temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1/dett, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0/dett, temp);
    }
};
/// \f$ y_i \leftarrow \lambda T^{-1}_{ij} x_i + \mu y_i\f$
template<class value_type>
struct InverseTensorMultiply3d{
    DG_DEVICE
    void operator() ( value_type lambda,
                      value_type t00, value_type t01, value_type t02,
                      value_type t10, value_type t11, value_type t12,
                      value_type t20, value_type t21, value_type t22,
                      value_type in0, value_type in1, value_type in2,
                      value_type mu,
                      value_type& out0, value_type& out1, value_type& out2) const
    {
        value_type dett = det( t00,t01,t02, t10,t11,t12, t20,t21,t22);

        value_type tmp0 = det( in0,t01,t02, in1,t11,t12, in2,t21,t22);
        value_type tmp1 = det( t00,in0,t02, t10,in1,t12, t20,in2,t22);
        value_type tmp2 = det( t00,t01,in0, t10,t11,in1, t20,t21,in2);
        value_type temp = out2*mu;
        out2 = DG_FMA( lambda, tmp2/dett, temp);
        temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1/dett, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0/dett, temp);
    }
    private:
    DG_DEVICE
    value_type det( value_type t00, value_type t01, value_type t02,
                    value_type t10, value_type t11, value_type t12,
                    value_type t20, value_type t21, value_type t22)const
    {
        return t00*DG_FMA(t11, t22, (-t12*t21))
              -t01*DG_FMA(t10, t22, (-t20*t12))
              +t02*DG_FMA(t10, t21, (-t20*t11));
    }
};
///@}

///@addtogroup variadic_evaluates
///@{

/// \f$ y = \lambda\mu v_i T_{ij} w_j \f$
template<class value_type>
struct TensorDot2d{
    DG_DEVICE
    value_type operator() (
              value_type lambda,
              value_type v0,  value_type v1,
              value_type t00, value_type t01,
              value_type t10, value_type t11,
              value_type mu,
              value_type w0, value_type w1
              ) const
    {
        value_type tmp0 = DG_FMA(t00,w0 , t01*w1);
        value_type tmp1 = DG_FMA(t10,w0 , t11*w1);
        return lambda*mu*DG_FMA(v0,tmp0  , v1*tmp1);
    }
};
/// \f$ y = \lambda \mu v_i T_{ij} w_j \f$
template<class value_type>
struct TensorDot3d{
    DG_DEVICE
    value_type operator() (
              value_type lambda,
              value_type v0,  value_type v1,  value_type v2,
              value_type t00, value_type t01, value_type t02,
              value_type t10, value_type t11, value_type t12,
              value_type t20, value_type t21, value_type t22,
              value_type mu,
              value_type w0, value_type w1, value_type w2) const
    {
        value_type tmp0 = DG_FMA( t00,w0 , (DG_FMA( t01,w1 , t02*w2)));
        value_type tmp1 = DG_FMA( t10,w0 , (DG_FMA( t11,w1 , t12*w2)));
        value_type tmp2 = DG_FMA( t20,w0 , (DG_FMA( t21,w1 , t22*w2)));
        return lambda*mu*DG_FMA(v0,tmp0 , DG_FMA(v1,tmp1 , v2*tmp2));
    }
};

///\f$ y = t_{00} t_{11} - t_{10}t_{01} \f$
template<class value_type>
struct TensorDeterminant2d
{
    DG_DEVICE
    value_type operator() ( value_type t00, value_type t01,
                            value_type t10, value_type t11) const
    {
        return DG_FMA( t00,t11 , (-t10*t01));
    }
};
///\f$ y = t_{00} t_{11}t_{22} + t_{01}t_{12}t_{20} + t_{02}t_{10}t_{21} - t_{02}t_{11}t_{20} - t_{01}t_{10}t_{22} - t_{00}t_{12}t_{21} \f$
template<class value_type>
struct TensorDeterminant3d
{
    DG_DEVICE
    value_type operator() ( value_type t00, value_type t01, value_type t02,
                            value_type t10, value_type t11, value_type t12,
                            value_type t20, value_type t21, value_type t22) const
    {
        return t00*m_t(t11, t12, t21, t22)
              -t01*m_t(t10, t12, t20, t22)
              +t02*m_t(t10, t11, t20, t21);
    }
    private:
    TensorDeterminant2d<value_type> m_t;
};
///@}

/**
 * @namespace dg::tensor
 * @brief Utility functions used in connection with the SparseTensor class
 * @ingroup tensor
 */
namespace tensor
{
///@addtogroup tensor
///@{


/**
 * @brief \f$ t^{ij} = \mu t^{ij} \ \forall i,j \f$
 *
 * Scale tensor with a Scalar or a Vector
 * @param t input (contains result on output)
 * @param mu all elements in t are scaled with mu
 * @copydoc hide_ContainerType
 */
template<class ContainerType0, class ContainerType1>
void scal( SparseTensor<ContainerType0>& t, const ContainerType1& mu)
{
    unsigned size=t.values().size();
    for( unsigned i=0; i<size; i++)
        dg::blas1::pointwiseDot( mu, t.values()[i], t.values()[i]);
}

/**
 * @brief \f$ w^i = \sum_{i=0}^1 \lambda t^{ij}v_j + \mu w^i \text{ for } i\in \{0,1\}\f$
 *
 * Multiply a tensor with a vector in 2d.
 * Ignore the 3rd dimension in \c t.
 * @param t input Tensor
 * @param lambda (input)
 * @param in0 (input) first component  of \c v  (may alias out0)
 * @param in1 (input) second component of \c v  (may alias out1)
 * @param mu (input)
 * @param out0 (output) first component  of \c w (may alias in0)
 * @param out1 (output) second component of \c w (may alias in1)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with \c dg::TensorMultiply2d
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerTypeM, class ContainerType3, class ContainerType4>
void multiply2d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerTypeM& mu, ContainerType3& out0, ContainerType4& out1)
{
    dg::blas1::subroutine( dg::TensorMultiply2d<get_value_type<ContainerType0>>(),
            lambda,      t.value(0,0), t.value(0,1),
                         t.value(1,0), t.value(1,1),
                         in0,  in1,
            mu,          out0, out1);
}

/**
 * @brief \f$ w^i = \sum_{i=0}^2\lambda t^{ij}v_j + \mu w^i \text{ for } i\in \{0,1,2\}\f$
 *
 * Multiply a tensor with a vector in 3d.
 * @param t input Tensor
 * @param lambda (input) (may be a vector or an actual number like 0 or 1)
 * @param in0 (input)  first component of \c v  (may alias out0)
 * @param in1 (input)  second component of \c v (may alias out1)
 * @param in2 (input)  third component of \c v  (may alias out2)
 * @param mu  (input) (may be a vector or an actual number like 0 or 1)
 * @param out0 (output)  first component of \c w  (may alias in0)
 * @param out1 (output)  second component of \c w (may alias in1)
 * @param out2 (output)  third component of \c w  (may alias in2)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with \c dg::TensorMultiply3d
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6>
void multiply3d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, const ContainerTypeM& mu, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
    dg::blas1::subroutine( dg::TensorMultiply3d<get_value_type<ContainerType0>>(),
            lambda,      t.value(0,0), t.value(0,1), t.value(0,2),
                         t.value(1,0), t.value(1,1), t.value(1,2),
                         t.value(2,0), t.value(2,1), t.value(2,2),
                         in0, in1, in2,
            mu,          out0, out1, out2);
}

/**
 * @brief \f$ v_j = \sum_{i=0}^1\lambda (t^{-1})_{ji}w^i + \mu v_j \text{ for } i\in \{0,1\}\f$
 *
 * Multiply the inverse of a tensor \c t with a vector in 2d.
 * Ignore the 3rd dimension in \c t. The inverse of \c t is computed inplace.
 * @param t input Tensor
 * @param lambda (input) (may be a vector or an actual number like 0 or 1)
 * @param in0 (input) first component of \c w    (may alias out0)
 * @param in1 (input) second component of \c w   (may alias out1)
 * @param mu  (input) (may be a vector or an actual number like 0 or 1)
 * @param out0 (output) first component of \c v  (may alias in0)
 * @param out1 (output) second component of \c v (may alias in1)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with \c dg::InverseTensorMultiply2d
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerTypeM, class ContainerType3, class ContainerType4>
void inv_multiply2d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerTypeM& mu, ContainerType3& out0, ContainerType4& out1)
{
    dg::blas1::subroutine( dg::InverseTensorMultiply2d<get_value_type<ContainerType0>>(),
              lambda,    t.value(0,0), t.value(0,1),
                         t.value(1,0), t.value(1,1),
                         in0,  in1,
              mu,        out0, out1);
}

/**
 * @brief \f$ v_j = \sum_{i=0}^2\lambda(t^{-1})_{ji}w^i + \mu v_j \text{ for } i\in \{0,1,2\}\f$i
 *
 * Multiply the inverse of a tensor with a vector in 3d.
 * The inverse of \c t is computed inplace.
 * @param t input Tensor
 * @param lambda (input) (may be a vector or an actual number like 0 or 1)
 * @param in0 (input)  first component  of \c w (may alias out0)
 * @param in1 (input)  second component of \c w (may alias out1)
 * @param in2 (input)  third component  of \c w (may alias out2)
 * @param mu  (input) (may be a vector or an actual number like 0 or 1)
 * @param out0 (output)  first component  of \c v (may alias in0)
 * @param out1 (output)  second component of \c v (may alias in1)
 * @param out2 (output)  third component  of \c v (may alias in2)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with \c dg::InverseTensorMultiply3d
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6>
void inv_multiply3d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, const ContainerTypeM& mu, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
    dg::blas1::subroutine( dg::InverseTensorMultiply3d<get_value_type<ContainerType0>>(),
           lambda,       t.value(0,0), t.value(0,1), t.value(0,2),
                         t.value(1,0), t.value(1,1), t.value(1,2),
                         t.value(2,0), t.value(2,1), t.value(2,2),
                         in0, in1, in2,
           mu,           out0, out1, out2);
}

/**
* @brief \f$\det_{2d}( t)\f$
*
* Compute the minor determinant of a tensor \f$ \det_{2d}(t) := t_{00}t_{01}-t_{10}t_{11}\f$.
* @param t the input tensor
* @return the upper left minor determinant of \c t
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate with \c dg::TensorDeterminant2d
* @copydoc hide_ContainerType
*/
template<class ContainerType>
ContainerType determinant2d( const SparseTensor<ContainerType>& t)
{
    ContainerType det = t.value(0,0);
    dg::blas1::evaluate( det, dg::equals(), dg::TensorDeterminant2d<get_value_type<ContainerType>>(),
                           t.value(0,0), t.value(0,1),
                           t.value(1,0), t.value(1,1));
    return det;
}

/**
* @brief \f$\det( t)\f$
*
* Compute the determinant of a 3d tensor:
* \f$ \det(t) := t_{00}t_{11}t_{22} + t_{01}t_{12}t_{20} + \ldots - t_{22}t_{10}t_{01}\f$.
* @param t the input tensor
* @return the determinant of t
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate with \c dg::TensorDeterminant3d
* @copydoc hide_ContainerType
*/
template<class ContainerType>
ContainerType determinant( const SparseTensor<ContainerType>& t)
{
    ContainerType det = t.value(0,0);
    dg::blas1::evaluate( det, dg::equals(), dg::TensorDeterminant3d<get_value_type<ContainerType>>(),
                           t.value(0,0), t.value(0,1), t.value(0,2),
                           t.value(1,0), t.value(1,1), t.value(1,2),
                           t.value(2,0), t.value(2,1), t.value(2,2));
    return det;
}

/**
 * @brief \f$ \sqrt{\det_{2d}(t)}^{-1}\f$
 *
 * Compute the sqrt of the inverse minor determinant of a tensor.
 * This is a convenience function that is equivalent to
 * @code
    ContainerType vol=determinant2d(t);
    dg::blas1::transform(vol, vol, dg::InvSqrt<>());
    @endcode
 *  @note The function is called volume because when you apply it to the inverse metric
    tensor of our grids then you obtain the volume
    \f[ \sqrt{g} = 1 / \sqrt{ \det( g^{-1})}\f]
    @code
    ContainerType vol = volume2d( g.metric());
    @endcode
 * @param t the input tensor
 * @return the inverse square root of the determinant of \c t
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
ContainerType volume2d( const SparseTensor<ContainerType>& t)
{
    ContainerType vol=determinant2d(t);
    dg::blas1::transform(vol, vol, dg::InvSqrt<get_value_type<ContainerType>>());
    return vol;
}

/**
 * @brief \f$ \sqrt{\det(t)}^{-1}\f$
 *
 * Compute the sqrt of the inverse determinant of a 3d tensor.
 * This is a convenience function that is equivalent to
 * @code
    ContainerType vol=determinant(t);
    dg::blas1::transform(vol, vol, dg::InvSqrt<>());
    @endcode
 *  @note The function is called volume because when you apply it to the inverse metric
    tensor of our grids then you obtain the volume
    \f[ \sqrt{g} = 1 / \sqrt{ \det( g^{-1})}\f]
    @code
    ContainerType vol = dg::tensor::volume( g.metric());
    @endcode
 * @param t the input tensor
 * @return the inverse square root of the determinant of \c t
 * @copydoc hide_ContainerType
 */
template<class ContainerType>
ContainerType volume( const SparseTensor<ContainerType>& t)
{
    ContainerType vol=determinant(t);
    dg::blas1::transform(vol, vol, dg::InvSqrt<get_value_type<ContainerType>>());
    return vol;
}

//For convenience
/**
 * @brief \f$ w^i = \sum_{i=0}^1 t^{ij}v_j  \text{ for } i\in \{0,1\}\f$
 *
 * Multiply a tensor with a vector in 2d.
 * Ignore the 3rd dimension in \c t.
 * @param t input Tensor
 * @param in0 (input) first component  of \c v  (may alias out0)
 * @param in1 (input) second component of \c v  (may alias out1)
 * @param out0 (output) first component  of \c w (may alias in0)
 * @param out1 (output) second component of \c w (may alias in1)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with the appropriate functor
 * @copydoc hide_ContainerType
 */
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4>
void multiply2d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, ContainerType3& out0, ContainerType4& out1)
{
    multiply2d( 1, t, in0, in1, 0., out0, out1);
}

/**
 * @brief \f$ w^i = \sum_{i=0}^2 t^{ij}v_j \text{ for } i\in \{0,1,2\}\f$
 *
 * Multiply a tensor with a vector in 3d.
 * @param t input Tensor
 * @param in0 (input)  first component of \c v  (may alias out0)
 * @param in1 (input)  second component of \c v (may alias out1)
 * @param in2 (input)  third component of \c v  (may alias out2)
 * @param out0 (output)  first component of \c w  (may alias in0)
 * @param out1 (output)  second component of \c w (may alias in1)
 * @param out2 (output)  third component of \c w  (may alias in2)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with the appropriate functor
 * @copydoc hide_ContainerType
 */
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4, class ContainerType5, class ContainerType6>
void multiply3d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
    multiply3d( 1., t, in0, in1, in2, 0., out0, out1, out2);
}

/**
 * @brief \f$ v_j = \sum_{i=0}^1(t^{-1})_{ji}w^i \text{ for } i\in \{0,1\}\f$
 *
 * Multiply the inverse of a tensor \c t with a vector in 2d.
 * Ignore the 3rd dimension in \c t. The inverse of \c t is computed inplace.
 * @param t input Tensor
 * @param in0 (input) first component of \c w    (may alias out0)
 * @param in1 (input) second component of \c w   (may alias out1)
 * @param out0 (output) first component of \c v  (may alias in0)
 * @param out1 (output) second component of \c v (may alias in1)
 * @copydoc hide_ContainerType
 */
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4>
void inv_multiply2d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, ContainerType3& out0, ContainerType4& out1)
{
    inv_multiply2d( 1., t, in0, in1, out0, out1);
}

/**
 * @brief \f$ v_j = \sum_{i=0}^2(t^{-1})_{ji}w^i \text{ for } i\in \{0,1,2\}\f$i
 *
 * Multiply the inverse of a tensor with a vector in 3d.
 * The inverse of \c t is computed inplace.
 * @param t input Tensor
 * @param in0 (input)  first component  of \c w (may alias out0)
 * @param in1 (input)  second component of \c w (may alias out1)
 * @param in2 (input)  third component  of \c w (may alias out2)
 * @param out0 (output)  first component  of \c v (may alias in0)
 * @param out1 (output)  second component of \c v (may alias in1)
 * @param out2 (output)  third component  of \c v (may alias in2)
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine with the appropriate functor
 * @copydoc hide_ContainerType
 */
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerType4, class ContainerType5, class ContainerType6>
void inv_multiply3d( const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
    inv_multiply3d( 1., t, in0, in1, in2, 0., out0, out1, out2);
}

/**
 * @brief \f$ y = \alpha \lambda\mu \sum_{i=0}^1 v_it^{ij}w_j + \beta y \text{ for } i\in \{0,1\}\f$
 *
 * Ignore the 3rd dimension in \c t.
 * @param alpha scalar input prefactor
 * @param lambda second input prefactor
 * @param v0 (input) first component  of \c v  (may alias w0)
 * @param v1 (input) second component of \c v  (may alias w1)
 * @param t input Tensor
 * @param mu third input prefactor
 * @param w0 (input) first component  of \c w  (may alias v0)
 * @param w1 (input) second component of \c w  (may alias v1)
 * @param beta scalar output prefactor
 * @param y (output)
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate with \c dg::TensorDot2d
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5>
void scalar_product2d(
        get_value_type<ContainerType0> alpha,
        const ContainerTypeL& lambda,
        const ContainerType0& v0,
        const ContainerType1& v1,
        const SparseTensor<ContainerType2>& t,
        const ContainerTypeM& mu,
        const ContainerType3& w0,
        const ContainerType4& w1,
        get_value_type<ContainerType0> beta,
        ContainerType5& y)
{
    dg::blas1::evaluate( y,
             dg::Axpby( alpha, beta),
             dg::TensorDot2d<get_value_type<ContainerType0>>(),
             lambda,
             v0, v1,
             t.value(0,0), t.value(0,1),
             t.value(1,0), t.value(1,1),
             mu,
             w0, w1);
}

/**
 * @brief \f$ y = \alpha \lambda\mu \sum_{i=0}^2 v_it^{ij}w_j + \beta y \text{ for } i\in \{0,1,2\}\f$
 *
 * @param alpha scalar input prefactor
 * @param lambda second input prefactor
 * @param v0 (input) first component  of \c v  (may alias w0)
 * @param v1 (input) second component of \c v  (may alias w1)
 * @param v2 (input) third component of \c v  (may alias w1)
 * @param t input Tensor
 * @param mu third input prefactor
 * @param w0 (input) first component  of \c w  (may alias v0)
 * @param w1 (input) second component of \c w  (may alias v1)
 * @param w2 (input) third component of \c w  (may alias v1)
 * @param beta scalar output prefactor
 * @param y (output)
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate with \c dg::TensorDot3d
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6, class ContainerType7>
void scalar_product3d(
        get_value_type<ContainerType0> alpha,
        const ContainerTypeL& lambda,
        const ContainerType0& v0,
        const ContainerType1& v1,
        const ContainerType2& v2,
        const SparseTensor<ContainerType3>& t,
        const ContainerTypeM& mu,
        const ContainerType4& w0,
        const ContainerType5& w1,
        const ContainerType6& w2,
        get_value_type<ContainerType0> beta,
        ContainerType7& y)
{
    dg::blas1::evaluate( y,
            dg::Axpby( alpha, beta),
            dg::TensorDot3d<get_value_type<ContainerType0>>(),
            lambda,
            v0, v1, v2,
            t.value(0,0), t.value(0,1), t.value(0,2),
            t.value(1,0), t.value(1,1), t.value(1,2),
            t.value(2,0), t.value(2,1), t.value(2,2),
            mu,
            w0, w1, w2);
}

///@}
}//namespace tensor
}//namespace dg
