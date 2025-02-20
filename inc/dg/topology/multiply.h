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
///@cond
// nvcc does not like local classes so we need to define these globally:
// \f$ y_i \leftarrow \lambda T_{ij} x_i + \mu y_i\f$
struct TensorMultiply2d
{
    template<class VL, class V0, class V1, class V2, class VM, class V3, class V4>
    DG_DEVICE
    void operator() ( VL lambda, V0 t00, V0 t01, V0 t10, V0 t11,
                      V1 in0, V2 in1, VM mu, V3& out0, V4& out1) const
    {
        auto tmp0 = DG_FMA(t00,in0 , t01*in1);
        auto tmp1 = DG_FMA(t10,in0 , t11*in1);
        auto temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0, temp);
    }
};
// \f$ y_i \leftarrow \lambda T_{ij} x_i + \mu y_i\f$
struct TensorMultiply3d
{
    template<class VL, class V0, class V1, class V2, class V3, class VM, class V4, class V5, class V6>
    DG_DEVICE
    void operator() ( VL lambda,
                      V0 t00, V0 t01, V0 t02,
                      V0 t10, V0 t11, V0 t12,
                      V0 t20, V0 t21, V0 t22,
                      V1 in0, V2 in1, V3 in2,
                      VM mu,
                      V4& out0, V5& out1, V6& out2) const
    {
        auto tmp0 = DG_FMA( t00,in0 , (DG_FMA( t01,in1 , t02*in2)));
        auto tmp1 = DG_FMA( t10,in0 , (DG_FMA( t11,in1 , t12*in2)));
        auto tmp2 = DG_FMA( t20,in0 , (DG_FMA( t21,in1 , t22*in2)));
        auto temp = out2*mu;
        out2 = DG_FMA( lambda, tmp2, temp);
        temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0, temp);
    }
};
// \f$ y_i \leftarrow \lambda T^{-1}_{ij} x_i + \mu y_i\f$
struct InverseTensorMultiply2d
{
    template<class VL, class V0, class V1, class V2, class VM, class V3, class V4>
    DG_DEVICE
    void operator() ( VL lambda, V0 t00, V0 t01, V0 t10, V0 t11,
                      V1 in0, V2 in1, VM mu, V3& out0, V4& out1) const
    {
        auto dett = DG_FMA( t00,t11 , (-t10*t01));
        auto tmp0 = DG_FMA( in0,t11 , (-in1*t01));
        auto tmp1 = DG_FMA( t00,in1 , (-t10*in0));
        auto temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1/dett, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0/dett, temp);
    }
};
// \f$ y_i \leftarrow \lambda T^{-1}_{ij} x_i + \mu y_i\f$
struct InverseTensorMultiply3d
{
    template<class VL, class V0, class V1, class V2, class V3, class VM, class V4, class V5, class V6>
    DG_DEVICE
    void operator() ( VL lambda,
                  V0 t00, V0 t01, V0 t02,
                  V0 t10, V0 t11, V0 t12,
                  V0 t20, V0 t21, V0 t22,
                  V1 in0, V2 in1, V3 in2,
                  VM mu,
                  V4& out0, V5& out1, V6& out2) const
    {
        auto dett = t00*DG_FMA(t11, t22, (-t12*t21))
                   -t01*DG_FMA(t10, t22, (-t20*t12))
                   +t02*DG_FMA(t10, t21, (-t20*t11));

        auto tmp0 = in0*DG_FMA(t11, t22, (-t12*t21))
                   -t01*DG_FMA(in1, t22, (-in2*t12))
                   +t02*DG_FMA(in1, t21, (-in2*t11));
        auto tmp1 = t00*DG_FMA(in1, t22, (-t12*in2))
                   -in0*DG_FMA(t10, t22, (-t20*t12))
                   +t02*DG_FMA(t10, in2, (-t20*in1));
        auto tmp2 = t00*DG_FMA(t11, in2, (-in1*t21))
                   -t01*DG_FMA(t10, in2, (-t20*in1))
                   +in0*DG_FMA(t10, t21, (-t20*t11));
        auto temp = out2*mu;
        out2 = DG_FMA( lambda, tmp2/dett, temp);
        temp = out1*mu;
        out1 = DG_FMA( lambda, tmp1/dett, temp);
        temp = out0*mu;
        out0 = DG_FMA( lambda, tmp0/dett, temp);
    }
};
//\f$ y = t_{00} t_{11} - t_{10}t_{01} \f$
struct TensorDeterminant2d
{
    template<class value_type>
    DG_DEVICE
    value_type operator() ( value_type t00, value_type t01,
                            value_type t10, value_type t11) const
    {
        return DG_FMA( t00,t11 , (-t10*t01));
    }
};
//\f$ y = t_{00} t_{11}t_{22} + t_{01}t_{12}t_{20} + t_{02}t_{10}t_{21} - t_{02}t_{11}t_{20} - t_{01}t_{10}t_{22} - t_{00}t_{12}t_{21} \f$
struct TensorDeterminant3d
{
    template<class value_type>
    DG_DEVICE
    value_type operator() ( value_type t00, value_type t01, value_type t02,
                            value_type t10, value_type t11, value_type t12,
                            value_type t20, value_type t21, value_type t22) const
    {
        return t00* DG_FMA( t11,t22 , (-t21*t12))
              -t01* DG_FMA( t10,t22 , (-t20*t12))
              +t02* DG_FMA( t10,t21 , (-t20*t11));
    }
};

// \f$ y = \lambda\mu v_i T_{ij} w_j \f$
struct TensorDot2d
{
    template<class VL, class V0, class V1, class V2, class VM, class V3, class V4>
    DG_DEVICE
    auto operator() (
              VL lambda, V0 v0, V1 v1,
              V2 t00, V2 t01,
              V2 t10, V2 t11,
              VM mu,     V3 w0, V4 w1
              ) const
    {
        auto tmp0 = DG_FMA(t00,w0 , t01*w1);
        auto tmp1 = DG_FMA(t10,w0 , t11*w1);
        return lambda*mu*DG_FMA(v0,tmp0  , v1*tmp1);
    }
};
// \f$ y = \lambda \mu v_i T_{ij} w_j \f$
struct TensorDot3d
{
    template<class VL, class V0, class V1, class V2, class V3, class VM, class V4, class V5, class V6>
    DG_DEVICE
    auto operator() (
              VL lambda,
              V0 v0,  V1 v1,  V2 v2,
              V3 t00, V3 t01, V3 t02,
              V3 t10, V3 t11, V3 t12,
              V3 t20, V3 t21, V3 t22,
              VM mu,
              V4 w0, V5 w1, V6 w2) const
    {
        auto tmp0 = DG_FMA( t00,w0 , (DG_FMA( t01,w1 , t02*w2)));
        auto tmp1 = DG_FMA( t10,w0 , (DG_FMA( t11,w1 , t12*w2)));
        auto tmp2 = DG_FMA( t20,w0 , (DG_FMA( t21,w1 , t22*w2)));
        return lambda*mu*DG_FMA(v0,tmp0 , DG_FMA(v1,tmp1 , v2*tmp2));
    }
};
///@endcond

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
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerTypeM, class ContainerType3, class ContainerType4>
void multiply2d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerTypeM& mu, ContainerType3& out0, ContainerType4& out1)
{
    dg::blas1::subroutine( dg::TensorMultiply2d(), lambda,
                           t.value(0,0), t.value(0,1),
                           t.value(1,0), t.value(1,1),
                           in0, in1, mu, out0, out1);
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
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6>
void multiply3d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, const ContainerTypeM& mu, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
    dg::blas1::subroutine(dg::TensorMultiply3d(),
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
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerTypeM, class ContainerType3, class ContainerType4>
void inv_multiply2d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerTypeM& mu, ContainerType3& out0, ContainerType4& out1)
{
    dg::blas1::subroutine( dg::InverseTensorMultiply2d(), lambda,
                           t.value(0,0), t.value(0,1),
                           t.value(1,0), t.value(1,1),
                           in0,  in1, mu, out0, out1);
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
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6>
void inv_multiply3d( const ContainerTypeL& lambda, const SparseTensor<ContainerType0>& t, const ContainerType1& in0, const ContainerType2& in1, const ContainerType3& in2, const ContainerTypeM& mu, ContainerType4& out0, ContainerType5& out1, ContainerType6& out2)
{
    dg::blas1::subroutine( dg::InverseTensorMultiply3d(),
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
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate
* @copydoc hide_ContainerType
*/
template<class ContainerType>
ContainerType determinant2d( const SparseTensor<ContainerType>& t)
{
    ContainerType det = t.value(0,0);
    dg::blas1::evaluate( det, dg::equals(), dg::TensorDeterminant2d(),
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
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate
* @copydoc hide_ContainerType
*/
template<class ContainerType>
ContainerType determinant( const SparseTensor<ContainerType>& t)
{
    ContainerType det = t.value(0,0);
    dg::blas1::evaluate( det, dg::equals(), TensorDeterminant3d(),
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
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine
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
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine
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
 * @note This function is just a shortcut for a call to \c dg::blas1::subroutine
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
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class value_type0, class value_type1>
void scalar_product2d(
        value_type0 alpha,
        const ContainerTypeL& lambda,
        const ContainerType0& v0,
        const ContainerType1& v1,
        const SparseTensor<ContainerType2>& t,
        const ContainerTypeM& mu,
        const ContainerType3& w0,
        const ContainerType4& w1,
        value_type1 beta,
        ContainerType5& y)
{
    dg::blas1::evaluate( y,
             dg::Axpby( alpha, beta),
             dg::TensorDot2d(),
             lambda, v0, v1,
             t.value(0,0), t.value(0,1),
             t.value(1,0), t.value(1,1),
             mu, w0, w1);
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
 * @note This function is just a shortcut for a call to \c dg::blas1::evaluate
 * @copydoc hide_ContainerType
 */
template<class ContainerTypeL, class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3, class ContainerTypeM, class ContainerType4, class ContainerType5, class ContainerType6, class ContainerType7, class value_type0, class value_type1>
void scalar_product3d(
        value_type0 alpha,
        const ContainerTypeL& lambda,
        const ContainerType0& v0,
        const ContainerType1& v1,
        const ContainerType2& v2,
        const SparseTensor<ContainerType3>& t,
        const ContainerTypeM& mu,
        const ContainerType4& w0,
        const ContainerType5& w1,
        const ContainerType6& w2,
        value_type1 beta,
        ContainerType7& y)
{
    dg::blas1::evaluate( y,
            dg::Axpby( alpha, beta),
            dg::TensorDot3d(),
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
