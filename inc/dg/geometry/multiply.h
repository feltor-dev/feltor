#pragma once 

#include "dg/backend/operator.h"
#include "dg/functors.h"
#include "dg/blas1.h"
#include "tensor.h"

namespace dg
{
namespace tensor
{

///@addtogroup geometry
///@{
/**
 * @brief calls sqrt transform function on value
 * @param mu if empty, value is assumed 1 and empty element is returned
 * @return Sparse element containing sqrt of input element
 */
SparseElement sqrt(const SparseElement& mu){ 
    SparseElement mu(*this);
    if( isSet()) dg::blas1::transform( mu.value(), mu.value(), dg::SQRT<double>());
    return mu;
}

/**
 * @brief calls invert transform function on value
 * @param mu if empty, value is assumed 1 and empty element is returned
 * @return Sparse element containing inverse of input element
 */
SparseElement invert(const SparseElement& mu){ 
    SparseElement mu(*this);
    if( isSet()) dg::blas1::transform( mu.value(), mu.value(), dg::INVERT<double>());
    return mu;
}

/**
 * @brief Scale tensor with a container
 *
 * Computes \f$ t^{ij} = \mu t^{ij}\f$ 
 * @tparam container container class 
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
 * @tparam container container class 
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
 * @tparam container container class 
 * @param mu if mu.isEmpty() then nothing happens, else the input is pointwise multiplied with the value in mu
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
 * @brief Divide container with form
 *
 * @tparam container container class 
 * @param in input vector
 * @param mu if mu.isEmpty() then nothing happens, else the input is pointwise divided with the value in mu
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

/**
 * @brief Multiply a tensor with a vector in 2d
 *
 * Compute \f$ w^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the first two dimensions (ignores the 3rd dimension in t)
 * @tparam container the container class
 * @param t input Tensor
 * @param in0 (input) covariant first component 
 * @param in1 (input) covariant second component
 * @param out0 (output) contravariant first component 
 * @param out1 (output) contravariant second component 
 * @note this version keeps the input intact 
 * @attention aliasing only allowed if tensor is either lower, or upper triangular 
 */
template<class container>
void multiply( const SparseTensor<container>& t, const container& in0, const container& in1, container& out0, container& out1)
{
    if(!t.isSet(0,1))//lower triangular
    {
        if(t.isSet(1,1))  dg::blas1::pointwiseDot( t.value(1,1), in1, out1);
        else out1=in1;
        if(t.isSet(1,0)) dg::blas1::pointwiseDot( 1., t.value(1,0), in0, 1., out1);
        if( t.isSet(0,0)) dg::blas1::pointwiseDot( t.value(0,0), in0, out0);
        else out0=in0;
        return;
    }
    //upper triangular and default
    if( t.isSet(0,0) ) 
        dg::blas1::pointwiseDot( t.value(0,0), in0, out0); 
    else 
        out0=in0;
    if(t.isSet(0,1)) //true
        dg::blas1::pointwiseDot( 1.,  t.value(0,1), in1, 1., out0);

    if( t.isSet(1,1) )
        dg::blas1::pointwiseDot( t.value(1,1), in1, out1);
    else 
        out1=in1;
    if(t.isSet(1,0)) //if aliasing happens this is wrong
        dg::blas1::pointwiseDot( 1.,  t.value(1,0), in0, 1., out1);
}

/**
 * @brief Multiply a tensor with a vector in 2d inplace
 *
 * Compute \f$ v^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the first two dimensions (ignores the 3rd dimension in t)
 * @tparam container the container class
 * @param t input Tensor
 * @param inout0 (input/output) covariant first component 
 * @param inout1 (input/output) covariant second component
 * @param workspace (write) optional workspace
 * @note this version overwrites the input and may or may not write into the workspace
 * @attention aliasing not allowed
 */
template<class container>
void multiply_inplace( const SparseTensor<container>& t, container& inout0, container& inout1, container& workspace)
{
    if( t.isSet(0,1) ) dg::blas1::pointwiseDot( t.value(0,1), inout1, workspace);
    //compute out1 inplace
    if( t.isSet(1,1)) dg::blas1::pointwiseDot( t.value(1,1), inout1, inout1);
    if( t.isSet(1,0)) dg::blas1::pointwiseDot( 1., t.value(1,0), inout0, 1., inout1);

    if(t.isSet(0,1)) //workspace is filled
    {
        if( !t.isSet(0,0)) dg::blas1::axpby( 1., inout0, 1., workspace, workspace);
        else dg::blas1::pointwiseDot( 1., t.value(0,0), inout0, 1., workspace); 
        workspace.swap( inout0);
    }
    else
        if( t.isSet(0,0)) dg::blas1::pointwiseDot( t.value(0,0), inout0, inout0); 

    //needs to load a vector         11 times if every element is set (5 is the optimal symmetric inplace algorithm)
    //if triangular                  7 (5 optimal)
    //if diagonal                    4 (optimal)
    //if unity                       0 (optimal)
}

/**
 * @brief Multiply a tensor with a vector in 2d
 *
 * Compute \f$ w^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the first two dimensions (ignores the 3rd dimension in t)
 * @tparam container the container class
 * @param t input Tensor
 * @param in0 (input) covariant first component 
 * @param in1 (input) covariant second component
 * @param in2 (input) covariant third component
 * @param out0 (output) contravariant first component 
 * @param out1 (output) contravariant second component 
 * @param out2 (output) contravariant third component 
 * @note this version keeps the input intact 
 * @attention aliasing only allowed if tensor is either lower, or upper triangular 
 */
template<class container>
void multiply( const SparseTensor<container>& t, const container& in0, const container& in1, const container& in2, container& out0, container& out1, container& out2)
{
    if( !t.isSet(0,1)&&!t.isSet(0,2)&&!t.isSet(1,2))
    {
        //lower triangular
        if(!t.isSet(2,2)) out2=in2;
        else dg::blas1::pointwiseDot( t.value(2,2), in2, out2);
        if(t.isSet(2,1))
            dg::blas1::pointwiseDot( 1., t.value(2,1), in1, 1., out2);
        if(t.isSet(2,0))
            dg::blas1::pointwiseDot( 1., t.value(2,0), in0, 1., out2);
        if( !t.isSet(1,1)) out1=in1;
        else dg::blas1::pointwiseDot( t.value(1,1), in1, out1);
        if(t.isSet(1,0))
            dg::blas1::pointwiseDot( 1., t.value(1,0), in0, 1., out1);
        if( !t.isSet(0,0)) out0=in0;
        else dg::blas1::pointwiseDot( t.value(0,0), in0, out0); 
    }
    //upper triangular and default
    if( !t.isSet(0,0)) out0=in0;
    else dg::blas1::pointwiseDot( t.value(0,0), in0, out0); 
    if(t.isSet( 0,1))
        dg::blas1::pointwiseDot( 1., t.value(0,1), in1, 1., out0);
    if(t.isSet( 0,2))
        dg::blas1::pointwiseDot( 1., t.value(0,2), in2, 1., out0);

    if( !t.isSet(1,1)) out1=in1;
    else dg::blas1::pointwiseDot( t.value(1,1), in1, out1);
    if(t.isSet(1,0))
        dg::blas1::pointwiseDot( 1., t.value(1,0), in0, 1., out1);
    if(t.isSet(1,2))
        dg::blas1::pointwiseDot( 1., t.value(1,2), in0, 1., out1);

    if(!t.isSet(2,2)) out2=in2;
    else dg::blas1::pointwiseDot( t.value(2,2), in2, out2);
    if(t.isSet(2,1))
        dg::blas1::pointwiseDot( 1., t.value(2,1), in1, 1., out2);
    if(t.isSet(2,0))
        dg::blas1::pointwiseDot( 1., t.value(2,0), in0, 1., out2);
}

/**
 * @brief Multiply a tensor with a vector in 2d inplace
 *
 * Compute \f$ v^i = t^{ij}v_j\f$ for \f$ i,j\in \{1,2\}\f$ in the first two dimensions (ignores the 3rd dimension in t)
 * @tparam container the container class
 * @param t input Tensor
 * @param inout0 (input/output) covariant first component 
 * @param inout1 (input/output) covariant second component
 * @param inout2 (input/output) covariant second component
 * @param workspace0 (write) optional workspace
 * @param workspace1 (write) optional workspace
 * @note this version overwrites the input and may or may not write into the workspace
 * @attention aliasing not allowed
 */
template<class container>
void multiply_inplace( const SparseTensor<container>& t, container& inout0, container& inout1, container& inout2, container& workspace0, container& workspace1)
{
    if( t.isSet(0,1) ) {
        dg::blas1::pointwiseDot( t.value(0,1), inout1, workspace0);
        if( t.isSet(0,2) ) dg::blas1::pointwiseDot( 1.,t.value(0,2), inout2, 1.,workspace0);
    }
    if(!t.isSet(0,1) && t.isSet(0,2))
    {
        dg::blas1::pointwiseDot( t.value(0,2), inout2, workspace0);
    }
    //else workspace0 is empty
    //
    if( t.isSet(1,0) ) {
        dg::blas1::pointwiseDot( t.value(1,0), inout0, workspace1);
        if( t.isSet(1,2) ) dg::blas1::pointwiseDot( 1.,t.value(1,2), inout2, 1.,workspace1);
    }
    if(!t.isSet(1,0) && t.isSet(1,2))
    {
        dg::blas1::pointwiseDot( t.value(1,2), inout2, workspace1);
    }
    //else workspace1 is empty
    //
    //compute out2 inplace
    if( t.isSet(2,2)) dg::blas1::pointwiseDot( t.value(2,2), inout2, inout2);
    if( t.isSet(2,1)) dg::blas1::pointwiseDot( 1., t.value(2,1), inout1, 1., inout2);
    if( t.isSet(2,0)) dg::blas1::pointwiseDot( 1., t.value(2,0), inout0, 1., inout2);

    if(t.isSet(0,1) ||t.isSet(0,2) ) //workspace0 is filled
    {
        if( !t.isSet(0,0)) dg::blas1::axpby( 1., inout0, 1., workspace0, workspace0);
        else dg::blas1::pointwiseDot( 1., t.value(0,0), inout0, 1., workspace0); 
        workspace0.swap( inout0);
    }
    else
        if( t.isSet(0,0)) dg::blas1::pointwiseDot( t.value(0,0), inout0, inout0); 
    if(t.isSet(1,0) ||t.isSet(1,2) ) //workspace1 is filled
    {
        if( !t.isSet(1,1)) dg::blas1::axpby( 1., inout1, 1., workspace1, workspace1);
        else dg::blas1::pointwiseDot( 1., t.value(1,1), inout1, 1., workspace1); 
        workspace1.swap( inout1);
    }
    else
        if( t.isSet(1,1)) dg::blas1::pointwiseDot( t.value(1,1), inout1, inout1); 
    //if everything is set: 28 loads (12 optimal, 9 optimal symmetric)
    //if effective 2d:      12 (same as 2d algorithm, 5 optimal symmetric)

}

/**
* @brief Compute the determinant of a tensor
* @tparam container the container class
* @param t the input tensor 
* @return the determinant of t as a SparseElement (unset if t is empty)
*/
template<class container>
SparseElement<container> determinant( const SparseTensor<container>& t)
{
    if(t.isEmpty())  return SparseElement<container>();
    SparseTensor<container> d = t.dense();
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

///@cond
//alias always allowed
template<class container>
void multiply( const CholeskyTensor<container>& ch, const container& in0, const container& in1, container& out0, container& out1)
{
    multiply(ch.upper(), in0, in1, out0, out1);
    //multiply( ch.diagonal(), out0, out1, out0, out1);
    //multiply(ch.lower(), out0, out1, out0, out1);
}
template<class container>
void multiply( const CholeskyTensor<container>& ch, const container& in0, const container& in1, const container& in2, container& out0, container& out1, container& out2)
{
    multiply(ch.upper(), in0, in1,in2, out0, out1,out2);
    multiply(ch.diagonal(), out0, out1,out2, out0, out1,out2);
    multiply(ch.lower(), out0, out1,out2, out0, out1,out2);
}

template<class container>
SparseElement<container> determinant( const CholeskyTensor<container>& ch)
{
    SparseTensor<container> diag = ch.diag().dense();
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
