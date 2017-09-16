#pragma once 

#include "dg/backend/operator.h"
#include "dg/functors.h"
#include "dg/blas1.h"
#include "tensor.h"

namespace dg
{
///@brief functions used in connection with the SparseElement and SparseTensor classes
namespace tensor
{

///@addtogroup geometry
///@{
/**
 * @brief calls sqrt transform function on value
 * @copydoc hide_container
 * @param mu if empty, stays empty, else contains sqrt of input
 */
template<class container>
void sqrt( SparseElement<container>& mu){ 
    if( mu.isSet()) 
        dg::blas1::transform( mu.value(), mu.value(), dg::SQRT<double>());
}

/**
 * @brief calls invert transform function on value
 * @copydoc hide_container
 * @param mu if empty, stays empty, else contains inverse of input
 */
template<class container>
void invert(SparseElement<container>& mu){ 
    if(mu.isSet()) 
        dg::blas1::transform( mu.value(), mu.value(), dg::INVERT<double>());
}

/**
 * @brief Scale tensor with a container
 *
 * Computes \f$ t^{ij} = \mu t^{ij}\f$ 
 * @copydoc hide_container
 * @param t input (contains result on output)
 * @param mu all elements in t are scaled with mu
 */
template<class container>
void scal( SparseTensor<container>& t, const container& mu)
{
    unsigned size=t.values().size();
    for( unsigned i=0; i<size; i++)
        dg::blas1::pointwiseDot( mu, t.value(i), t.value(i));
    if(!t.isSet(0,0)|| !t.isSet(1,1) || !t.isSet(2,2))
        t.value(size) = mu;
    for( unsigned i=0; i<3; i++)
    {
        if(!t.isSet(i,i) )
            t.idx(i,i)=size;
    }
}

/**
 * @brief Scale tensor with a form
 *
 * Computes \f$ t^{ij} = \mu t^{ij}\f$ 
 * @copydoc hide_container
 * @param t input (contains result on output)
 * @param mu if mu.isEmpty() then nothing happens, else all elements in t are scaled with its value
 */
template<class container>
void scal( SparseTensor<container>& t, const SparseElement<container>& mu)
{
    if(!mu.isSet()) return;
    else scal(t,mu.value());
}

/**
 * @brief Multiply container with form
 *
 * @copydoc hide_container
 * @param mu if mu.isEmpty() then out=in, else the input is pointwise multiplied with the value in mu
 * @param in input vector
 * @param out output vector (may alias in)
 */
template<class container>
void pointwiseDot( const SparseElement<container>& mu, const container& in, container& out)
{
    if(mu.isSet()) 
        dg::blas1::pointwiseDot(mu.value(), in,out);
    else
        out=in;
}
/**
 * @brief Multiply container with form
 *
 * @copydoc hide_container
 * @param in input vector
 * @param mu if mu.isEmpty() then out=in, else the input is pointwise multiplied with the value in mu
 * @param out output vector (may alias in)
 */
template<class container>
void pointwiseDot( const container& in, const SparseElement<container>& mu, container& out)
{
    pointwiseDot( mu, in, out);
}

/**
 * @brief Divide container with form
 *
 * @copydoc hide_container
 * @param in input vector
 * @param mu if mu.isEmpty() then out=in, else the input is pointwise divided with the value in mu
 * @param out output vector (may alias in)
 */
template<class container>
void pointwiseDivide( const container& in, const SparseElement<container>& mu, container& out)
{
    if(mu.isSet()) 
        dg::blas1::pointwiseDivide(in, mu.value(),out);
    else
        out=in;
}

///@cond
namespace detail
{
//i0 must be the diagonal index, out0 may alias in0 but not in1
template<class container>
void multiply2d_helper( const SparseTensor<container>& t, const container& in0, const container& in1, container& out0, int i0[2], int i1[2])
{
    if( t.isSet(i0[0],i0[1]) && t.isSet(i1[0],i1[1]) ) 
        dg::blas1::pointwiseDot( 1. , t.value(i0[0],i0[1]), in0, 1., t.value(i1[0], i1[1]), in1, 0., out0); 
    else if( t.isSet(i0[0],i0[1]) && !t.isSet(i1[0],i1[1]) ) 
        dg::blas1::pointwiseDot( t.value(i0[0], i0[1]), in0, out0);
    else 
    {
        out0=in0;
        if( t.isSet(i1[0], i1[1]))
            dg::blas1::pointwiseDot( 1.,  t.value(i1[0], i1[1]), in1, 1., out0);
    }
}
}//namespace detail
///@endcond
/**
 * @brief Multiply a tensor with a vector in 2d
 *
 * Compute \f$ w^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the first two dimensions (ignores the 3rd dimension in t)
 * @copydoc hide_container
 * @param t input Tensor
 * @param in0 (input) first component    (restricted)
 * @param in1 (input) second component   (may alias out1)
 * @param out0 (output) first component  (restricted)
 * @param out1 (output) second component (may alias in1)
 * @attention aliasing only allowed between out1 and in1
 */
template<class container>
void multiply2d( const SparseTensor<container>& t, const container& in0, const container& in1, container& out0, container& out1)
{
    int i0[2] = {0,0}, i1[2] = {0,1};
    int i3[2] = {1,1}, i2[2] = {1,0};
    //order is important because out1 may alias in1
    detail::multiply2d_helper( t, in0, in1, out0, i0, i1);
    detail::multiply2d_helper( t, in1, in0, out1, i3, i2);
    //needs to load a vector         10 times if every element is set (7 is the optimal algorithm for symmetric t)
    //(the ideal algorithm also only needs 70% of the time (tested on marconi))
}

/**
 * @brief Multiply a tensor with a vector in 2d
 *
 * Compute \f$ w^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2,3\}\f$
 * @copydoc hide_container
 * @param t input Tensor
 * @param in0 (input)  first component  (restricted)
 * @param in1 (input)  second component (restricted)
 * @param in2 (input)  third component  (may alias out2)
 * @param out0 (output)  first component  (restricted)
 * @param out1 (output)  second component  (restricted)
 * @param out2 (output)  third component (may alias in2)
 * @attention aliasing only allowed between out2 and in2
 */
template<class container>
void multiply3d( const SparseTensor<container>& t, const container& in0, const container& in1, const container& in2, container& out0, container& out1, container& out2)
{
    int i0[2] = {0,0}, i1[2] = {0,1};
    int i3[2] = {1,1}, i2[2] = {1,0};
    int i5[2] = {2,2}, i4[2] = {2,0};
    detail::multiply2d_helper( t, in0, in1, out0, i0, i1);
    if( t.isSet(0,2)) dg::blas1::pointwiseDot( 1., t.value(0,2), in2, 1., out0);
    detail::multiply2d_helper( t, in1, in0, out1, i3, i2);
    if( t.isSet(1,2)) dg::blas1::pointwiseDot( 1., t.value(1,2), in2, 1., out1);
    detail::multiply2d_helper( t, in2, in0, out2, i5, i4);
    if( t.isSet(2,1)) dg::blas1::pointwiseDot( 1., t.value(2,1), in1, 1., out2);
}

/**
* @brief Compute the determinant of a tensor
* @copydoc hide_container
* @param t the input tensor 
* @return the determinant of t as a SparseElement (unset if t is empty)
*/
template<class container>
SparseElement<container> determinant( const SparseTensor<container>& t)
{
    if(t.isEmpty())  return SparseElement<container>();
    SparseTensor<container> d = dense(t);
    container det = d.value(0,0);
    std::vector<container> sub_det(3,det);
    dg::blas1::transform( det, det, dg::CONSTANT(0));
    //first compute the det of three submatrices
    dg::blas1::pointwiseDot( d.value(0,0), d.value(1,1), sub_det[2]);
    dg::blas1::pointwiseDot( -1., d.value(1,0), d.value(0,1), 1. ,sub_det[2]);

    dg::blas1::pointwiseDot( d.value(0,0), d.value(2,1), sub_det[1]);
    dg::blas1::pointwiseDot( -1., d.value(2,0), d.value(0,1), 1. ,sub_det[1]);

    dg::blas1::pointwiseDot( d.value(1,0), d.value(2,1), sub_det[0]);
    dg::blas1::pointwiseDot( -1., d.value(2,0), d.value(1,1), 1. ,sub_det[0]);

    //now multiply according to Laplace expansion
    dg::blas1::pointwiseDot( 1., d.value(0,2), sub_det[0], 1.,  det);
    dg::blas1::pointwiseDot(-1., d.value(1,2), sub_det[1], 1.,  det);
    dg::blas1::pointwiseDot( 1., d.value(2,2), sub_det[2], 1.,  det);

    return SparseElement<container>(det);
}

/**
 * @brief Compute the sqrt of the inverse determinant of a tensor
 *
 * This is a convenience function that is the same as
 * @code
    SparseElement<container> volume=determinant(g);
    invert(volume);
    sqrt(volume);
    @endcode
 * @copydoc hide_container
 * @param t the input tensor 
 * @return the inverse square root of the determinant of t as a SparseElement (unset if t is empty)
 */
template<class container>
SparseElement<container> volume( const SparseTensor<container>& t)
{
    SparseElement<container> vol=determinant(t);
    invert(vol);
    sqrt(vol);
    return vol;

}


///@cond
//alias always allowed
template<class container>
void multiply2d( const CholeskyTensor<container>& ch, const container& in0, const container& in1, container& out0, container& out1)
{
    multiply2d(ch.upper(),     in0,  in1,  out0, out1);
    multiply2d(ch.diagonal(),  out0, out1, out0, out1);
    multiply2d(ch.lower(),     out0, out1, out0, out1);
}
template<class container>
void multiply3d( const CholeskyTensor<container>& ch, const container& in0, const container& in1, const container& in2, container& out0, container& out1, container& out2)
{
    multiply3d(ch.upper(),    in0,  in1, in2,  out0, out1, out2);
    multiply3d(ch.diagonal(), out0, out1,out2, out0, out1, out2);
    container temp(out1);
    multiply3d(ch.lower(),    out0, out1,out2, out0, temp, out2);
    temp.swap(out1);
}

template<class container>
SparseElement<container> determinant( const CholeskyTensor<container>& ch)
{
    SparseTensor<container> diag = dense(ch.diag() );
    SparseElement<container> det;
    if(diag.isEmpty()) return det;
    else det.value()=diag.value(0,0);
    dg::blas1::pointwiseDot( det.value(), diag.value(1,1), det.value());
    dg::blas1::pointwiseDot( det.value(), diag.value(2,2), det.value());
    return det;
}

template<class container>
void scal(const CholeskyTensor<container>& ch, const SparseElement<container>& e)
{
    const SparseTensor<container>& diag = ch.diag();
    if(!e.isSet()) return;
    unsigned size=diag.values().size();
    for( unsigned i=0; i<size; i++)
        dg::blas1::pointwiseDot( e.value(), diag.value(i), diag.value(i));
    if(!diag.isSet(0,0)|| !diag.isSet(1,1) || !diag.isSet(2,2))
        diag.value(size) = e.value();
    for( unsigned i=0; i<3; i++)
    {
        if(!diag.isSet(i,i) )
            diag.idx(i,i)=size;
    }
}
///@endcond

///@}

}//namespace tensor
}//namespace dg
