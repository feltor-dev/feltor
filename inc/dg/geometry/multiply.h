#pragma once 

#include "dg/backend/operator.h"
#include "dg/functors.h"
#include "dg/blas1.h"

namespace tensor
{

///@addtogroup geometry
///@{

template<class container>
void scal( SparseTensor<container>& t, const SparseElement<container>& e)
{
    if(!e.isSet()) return;
    for( unsigned i=0; i<2; i++)
        for( unsigned j=0; j<2; i++)
            if(t.isSet(i,j)) dg::blas1::pointwiseDot( e.value(), t.value(i,j), t.value(i,j));
}

template<class container>
void multiply( const SparseElement<container>& e, const container& in, container& out)
{
    if(e.isSet()) 
        dg::blas1::pointwiseDot(e.value(), in,out);
    else
        out=in;
}

template<class container>
void divide( const container& in, const SparseElement<container>& e, container& out)
{
    if(e.isSet()) 
        dg::blas1::pointwiseDivide(in, e.value(),out);
    else
        out=in;
}

///this version keeps the input intact 
///aliasing allowed if tensor is either lower, upper or diagonal alias allowed
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
    //upper triangular
    if( t.isSet(0,0) ) 
        dg::blas1::pointwiseDot( t.value(0,0), in0, out0); 
    else 
        out0=in0;
    if(t.isSet(0,1))
        dg::blas1::pointwiseDot( 1.,  t.value(0,1), in1, 1., out0);

    if( t.isSet(1,1) )
        dg::blas1::pointwiseDot( t.value(1,1), in1, out1);
    else 
        out1=in1;
    if(t.isSet(1,0))
        dg::blas1::pointwiseDot( 1.,  t.value(1,0), in0, 1., out1);
}

template<class container>
void multiply( const SparseTensor<container>& t, container& inout0, container& inout1, container& workspace)
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

///aliasing allowed if t is known to be lower or upper triangular
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

template<class container>
void multiply( const SparseTensor<container>& t, container& inout0, container& inout1, container& inout2, container& workspace0, container& workspace1)
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

template<class container>
SparseElement<container> determinant( const SparseTensor<container>& t)
{
    if(t.isEmpty())  return SparseElement<container>();
    SparseTensor<container> d = t.dense();
    container det = d.value(0,0);
    std::vector<container> sub_det(3,det);
    dg::blas1::transform( det, det, dg::CONSTANT(0));
    //first compute the det of three submatrices
    dg::blas1::pointwiseDot( d(0,0), d(1,1), sub_det[2]);
    dg::blas1::pointwiseDot( -1., d(1,0), d(0,1), 1. ,sub_det[2]);

    dg::blas1::pointwiseDot( d(0,0), d(2,1), sub_det[1]);
    dg::blas1::pointwiseDot( -1., d(2,0), d(0,1), 1. ,sub_det[1]);

    dg::blas1::pointwiseDot( d(1,0), d(2,1), sub_det[0]);
    dg::blas1::pointwiseDot( -1., d(2,0), d(1,1), 1. ,sub_det[0]);

    //now multiply according to Laplace expansion
    dg::blas1::pointwiseDot( 1., d(0,2), sub_det[0], 1.,  det);
    dg::blas1::pointwiseDot(-1., d(1,2), sub_det[1], 1.,  det);
    dg::blas1::pointwiseDot( 1., d(2,2), sub_det[2], 1.,  det);

    return SparseElement<container>(det);
}

///@cond
//alias always allowed
template<class container>
void multiply( const CholeskyTensor<container>& ch, const container& in0, const container& in1, container& out0, container& out1)
{
    multiply(ch.upper(), in0, in1, out0, out1);
    multiply(ch.diag(), out0, out1, out0, out1);
    multiply(ch.lower(), out0, out1, out0, out1);
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
    if(!diag.isSet(0,0)|| !diag.isSet(1,1) || !diag.isSet(2,2))
        diag.value(size) = e.value();
    for( unsigned i=0; i<2; i++)
    {
        if(!diag.isSet(i,i))
            diag.idx(i,i)=size;
        else
            dg::blas1::pointwiseDot( e.value(), diag.value(i,i), diag.value(i,i));
    }
}
///@endcond

///@}

}//namespace tensor
