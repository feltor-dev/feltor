#pragma once 

#include <cusp/coo_matrix.h>
#include "dg/backend/operator.h"
#include "dg/blas1.h"

namespace dg
{

/**
* @brief Abstract base class for matrices and vectors sharing or implicitly assuming elements 
*
* The goal of this class is to enable shared access to stored containers 
* or not store them at all since the storage of (and computation with) a container is expensive.

* This class contains both a (dense) matrix and a (dense) vector of integers.
* If positive or zero, the integer represents a gather index into an array of containers, 
if negative the value of the container is assumed to be 1, except for the off-diagonal entries
    in the matrix where it is assumed to be 0.
* We then only need to store non-trivial and non-repetitive containers.
* @tparam container container class
* @ingroup misc
*/
template<class container>
struct SharedContainers
{
    SharedContainers( ) {}
    SharedContainers( unsigned size):mat_idx_(size,-1), vec_idx_(size,-1){}

    /**
    * @brief 
    *
    * @param mat_idx
    * @param vec_idx
    * @param values The contained containers must all have the same size
    */
    SharedContainers( const dg::Operator<int>& mat_idx, std::vector<int>& vec_idx, std::vector<container>& values ): mat_idx_(mat_idx), vec_idx(vec_idx), values_(values){}

    template<class otherContainer>
    SharedContainers( const SharedContainers<otherContainer>& src): mat_idx_(src.mat_idx()), vec_idx_(src.vec_idx()), values_(src.values().size()), idx_(src.idx_){
        dg::blas1::transfer( src.values(), values_);
    }
    void set( const dg::Operator<int>& mat_idx, std::vector<int>& vec_idx, std::vector<container>& values ){
        mat_idx_=mat_idx;
        vec_idx_=vec_idx;
        values_=values;
    }

    /**
    * @brief check if an index is set or not
    * @param i row index 0<i<2
    * @param j col index 0<j<2
    * @return true if container is non-empty, false if value is assumed implicitly
    */
    bool isSet(size_t i, size_t j)const{
        if( mat_idx_(i,j) <0) return false;
        return true;
    }


    /**
    * @brief Test the matrix for emptiness
    * @return  true if no value in the matrix is set
    */
    bool empty()const{
        bool epty = true;
        for( unsigned i=0; i<mat_idx_.size(); i++)
            for( unsigned j=0; j<mat_idx_.size(); j++)
                if(isSet(i,j)) 
                    epty=false;
        return epty;
    }
     

    /**
    * @brief check if an index is set or not
    * @param i row index 0<i<2
    * @return true if container is non-empty, false if value is assumed implicitly
    */
    bool isSet(size_t i) const{
        if( vec_idx_[i]<0) return false;
        return true;
    }
    /*!@brief Access the underlying container
     * @return if !isSet(i,j) the default constructor of container is called, otherwise values[mat_idx(i,j)] is returned. 
     * @note If the indices fall out of range of mat_idx the result is undefined
     */
    const container& getValue(size_t i, size_t j)const{ 
        int k = mat_idx(i,j);
        if(k<0) return container();
        return values_[k];
    }
    /*!@brief Access the underlying container
     * @return if !isSet(i) the default constructor of container is called, otherwise values[vec_idx(i)] is returned. 
     * @note If the index falls out of range of vec_idx the result is undefined
     */
    const container& getValue(size_t i)const{ 
        int k = vec_idx_[i];
        if(k<0) return container();
        return values_[k];
    }

    private:
    dg::Operator<int> mat_idx_;
    std::vector<int> vec_idx_;
    std::vector<container> values_;
};

///@cond
namespace detail
{
template<class container>
void multiply( const SharedContainers<container>& t, container& in1, container& in2, container& out1, container& out2)
{
    if( t.empty())
    {
        in1.swap(out1);
        in2.swap(out2);
        return;
    }
    if( !tensor.isSet(0,0)) out1=in1;
    if(tensor.isSet( 0,0))
        dg::blas1::pointwiseDot( tensor.getValue(0,0), in1, out1); 
    if(tensor.isSet( 0,1))
        dg::blas1::pointwiseDot( 1., tensor.getValue(0,1), in2, 1., out1);

    if( !tensor.isSet(1,1)) out2=in2;
    if(tensor.isSet(1,1))
        dg::blas1::pointwiseDot( tensor.getValue(1,1), in2, out2);
    if(tensor.isSet(1,0))
        dg::blas1::pointwiseDot( 1., tensor.getValue(1,0), in1, 1., out2);
}

template<class container>
void multiply( const SharedContainers<container>& t, container& in1, container& in2, container& in3, container& out1, container& out2, container& out3)
{
    if( t.empty())
    {
        in1.swap(out1);
        in2.swap(out2);
        in3.swap(out3);
        return;
    }
    if( !tensor.isSet(0,0)) out1=in1;
    if(tensor.isSet( 0,0))
        dg::blas1::pointwiseDot( tensor.getValue(0,0), in1, out1); 
    if(tensor.isSet( 0,1))
        dg::blas1::pointwiseDot( 1., tensor.getValue(0,1), in2, 1., out1);
    if(tensor.isSet( 0,2))
        dg::blas1::pointwiseDot( 1., tensor.getValue(0,2), in3, 1., out1);

    if( !tensor.isSet(1,1)) out2=in2;
    if(tensor.isSet(1,1))
        dg::blas1::pointwiseDot( tensor.getValue(1,1), in2, out2);
    if(tensor.isSet(1,0))
        dg::blas1::pointwiseDot( 1., tensor.getValue(1,0), in1, 1., out2);
    if(tensor.isSet(1,2))
        dg::blas1::pointwiseDot( 1., tensor.getValue(1,2), in1, 1., out2);

    if(!tensor.isSet(2,2)) out3=in3;
    if(tensor.isSet(2,2))
        dg::blas1::pointwiseDot( tensor.getValue(2,2), in3, out3);
    if(tensor.isSet(2,1))
        dg::blas1::pointwiseDot( 1., tensor.getValue(2,1), in2, 1., out3);
    if(tensor.isSet(2,0))
        dg::blas1::pointwiseDot( 1., tensor.getValue(2,0), in1, 1., out3);
}

template<class container>
void sandwich( const SharedContainers<container>& t, container& chiRR, container& chiRZ, container& chiZZ, container& chixx, container& chixy, container& chiyy)
{
    if(t.empty())
    {
        chiRR.swap(chixx);
        chiRZ.swap(chixy);
        chiZZ.swap(chiyy);
        return;
    }
    
    //this is a default implementation  (add cases if optimization is necessary)
    //compute the transformation matrix
    container t00(chixx), t01(t00), t02(t00), t10(t00), t11(t00), t12(t00), t20(t00), t21(t00), t22(t00);
    container xr(chixx), xz(xr), yr(xr), yz(xz);
    //fill values for easier computations
    if(t.isSet(0,0)) dg::blas1::transfer( t.getValue(0,0), xr);
    else             dg::blas1::transform( xr,xr, dg::CONSTANT(1));
    if(t.isSet(0,1)) dg::blas1::transfer( t.getValue(0,1), xz);
    else             dg::blas1::transform( xz,xz, dg::CONSTANT(0));
    if(t.isSet(1,0)) dg::blas1::transfer( t.getValue(1,0), yr);
    else             dg::blas1::transform( yr,yr, dg::CONSTANT(0));
    if(t.isSet(1,1)) dg::blas1::transfer( t.getValue(1,1), yz);
    else             dg::blas1::transform( yz,yz, dg::CONSTANT(1));

    dg::blas1::pointwiseDot( xr, xr, t00);
    dg::blas1::pointwiseDot( xr, xz, t01);
    dg::blas1::scal( t01, 2.);
    dg::blas1::pointwiseDot( xz, g.xz, t02);

    dg::blas1::pointwiseDot( xr, yr, t10);
    dg::blas1::pointwiseDot( xr, yz, t11);
    dg::blas1::pointwiseDot( 1., yr, g.xz(), 1., t11);
    dg::blas1::pointwiseDot( xz, yz, t12);

    dg::blas1::pointwiseDot( yr, yr, t20);
    dg::blas1::pointwiseDot( yr, yz, t21);
    dg::blas1::scal( t21, 2.);
    dg::blas1::pointwiseDot( yz, yz, t22);

    //now multiply
    dg::blas1::pointwiseDot(     t00, chiRR,     chixx);
    dg::blas1::pointwiseDot( 1., t01, chiRZ, 1., chixx);
    dg::blas1::pointwiseDot( 1., t02, chiZZ, 1., chixx);
    dg::blas1::pointwiseDot(     t10, chiRR,     chixy);
    dg::blas1::pointwiseDot( 1., t11, chiRZ, 1., chixy);
    dg::blas1::pointwiseDot( 1., t12, chiZZ, 1., chixy);
    dg::blas1::pointwiseDot(     t20, chiRR,     chiyy);
    dg::blas1::pointwiseDot( 1., t21, chiRZ, 1., chiyy);
    dg::blas1::pointwiseDot( 1., t22, chiZZ, 1., chiyy);
}

}//namespace detail
///@endcond

}//namespace dg
