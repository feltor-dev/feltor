#pragma once 

#include <cusp/coo_matrix.h>
#include "dg/backend/operator.h"
#include "dg/blas1.h"

namespace dg
{

/*
1. make SparseTensor (remove vec_idx in SparseTensor)
2. make SparseElement (a std::vector with 0 or 1 element)
10  SparseElement det = dg::tensor::determinant(metric) //if not empty make dense and then compute
3. dg::tensor::sqrt(metric) //inverse sqrt of determinant
4. if(vol.isSet()) dg::blas1::pointwiseDot( vol, ...)
5. dg::tensor::scal( metric, vol); //if(isSet) multiply all values elements
6. dg::tensor::multiply( metric, in1, in2, out1, out2); //sparse matrix dense vector mult
6. dg::tensor::multiply( metric, in1, in2, in3, out1, out2, out3);  
7  perp_metric = metric.perp(); //for use in arakawa when computing perp_vol
8  dense_metric = metric.dens(); //add 0 and 1 to values and rewrite indices
9  metric.isPerp(), metric.isDense(), metric.isEmpty()
11 dg::tensor::product( tensor, jac, temp) //make jac dense and then use multiply 3 times (needs write access in getValue(i,j)
12 dg::tensor::product( jac.transpose(), temp, result)  //then use three elements if symmetric
13 Geometry has  std::vector<sparseElements>  v[0], v[1], v[2] and Jacobian 
14 when computing metric believe generator and compute on host for-loop
*/
  

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
struct SparseTensor
{
    SparseTensor( ):mat_idx(3-1) {}

    /**
    * @brief 
    *
    * @param mat_idx
    * @param values The contained containers must all have the same size
    */
    SparseTensor( const std::vector<container>& values ): mat_idx_(3,-1), values_(values){}

    template<class otherContainer>
    SparseTensor( const SparseTensor<otherContainer>& src): mat_idx_(src.mat_idx()), values_(src.values().size()), idx_(src.idx_){
        dg::blas1::transfer( src.values(), values_);
    }
    ///sets values leaves indices unchanged (size must be  the same or larger as current 
    void set( const std::vector<container>& values ){
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

    int operator()(unsigned i, unsigned j)const{return mat_idx_(i,j);}
    int& operator()(unsigned i, unsigned j){return mat_idx_(i,j);}

    /**
    * @brief Test the matrix for emptiness
    * @return  true if no value in the matrix is set
    */
    bool isEmpty()const{ 
        bool empty=true;
        for(unsigned i=0; i<3; i++)
            for( unsigned j=0; j<3; j++)
                if( isSet(i,j) ) 
                    empty=false;
        return empty;
    }
    ///if all elements are set
    bool isDense()const{
        bool dense=true;
        for(unsigned i=0; i<3; i++)
            for( unsigned j=0; j<3; j++)
                if( !isSet(i,j) ) 
                    dense=false;
        return dense;
    }
    ///if all elements in the third dimension are empty
    bool isPerp() const
    {
        bool empty=true;
        for(unsigned i=0; i<3; i++)
        {
            if( isSet(i,2) || isSet(2,i)) 
                empty=false;
        }
        return empty;
    }
     
     ///return an empty Tensor
     SparseTensor empty()const{return SparseTensor();}
     ///copy and fill all unset values
     SparseTensor dense()const;
     ///copy and erase all values in the third dimension
     SparseTensor perp()const;
     ///unset an index and clears values
     void erase( unsigned i, unsigned j) {
         if(!isSet(i,j)) return;
         mat_idx_(i,j)=-1;
         clear_unused_values();
     }
     ///clear any unused valuse and reset the corresponding indices
     void clear_unused_values();

    /*!@brief Access the underlying container
     * @return if !isSet(i,j) the result is undefined, otherwise values[mat_idx(i,j)] is returned. 
     * @note If the indices fall out of range of mat_idx the result is undefined
     */
    const container& getValue(size_t i, size_t j)const{ 
        int k = mat_idx_(i,j);
        return values_[k];
    }
    container& getValue( size_t i, size_t j)
    {
        int k = mat_idx_(i,j);
        return values_[k];
    }
    ///clear all values
    void clear(){
        mat_idx_=dg::Operator<int>(3,-1);
        values_.clear()
    }
    SparseTensor transpose()const{
        SparseTensor tmp(*this);
        tmp.mat_idx_ = mat_idx_.transpose();
    }

    private:
    dg::Operator<int> mat_idx_;
    std::vector<container> values_;
    void unique_insert(std::vector<int>& indices, int& idx)
};
///@cond
template<class container>
SparseTensor<container> SparseTensor<container>::dense() const
{
    SparseTensor<container> t(*this);
    if( isEmpty()) return t;
    container tmp = t.values_[0];
    //1. diagonal
    size_t size= values_.size();
    bool diagonalIsSet=true;
    for(unsigned i=0; i<3; i++)
        if(!t.isSet(i,i)) diagonalIsSet = false;
    dg::blas1::transform( tmp, tmp, dg::CONSTANT(1));
    if (!diagonalIsSet) t.values_.push_back(tmp);
    for(unsigned i=0; i<3; i++)
        if(!t.isSet(i,i)) t.mat_idx_(i,i) = size;
    //2. off-diagonal
    size = t.values_.size();
    bool offIsSet=true;
    for(unsigned i=0; i<3; i++)
        for(unsigned j=0; j<3; j++)
            if( !t.isSet(i,j) ) offIsSet=true;
    dg::blas1::transform( tmp, tmp, dg::CONSTANT(0));
    if (!offIsSet) t.values_.push_back(tmp);
    for(unsigned i=0; i<3; i++)
        for(unsigned j=0; j<3; j++)
            if(!t.isSet(i,j) ) t.mat_idx_(i,j) = size;
    return t;
}

template<class container>
SparseTensor<container> SparseTensor<container>::unique_insert(std::vector<int>& indices, int& idx) 
{
    bool unique=true;
    unsigned size=indices.size();
    for(unsigned i=0; i<size; i++)
        if(indices[i] == idx) unique=false;
    if(unique)
    {
        indices.push_back(idx);
        idx=size;
    }
}

template<class container>
SparseTensor<container> SparseTensor<container>::perp() const
{
    SparseTensor<container> t(*this);
    if( isEmpty()) return t;
    for(unsigned i=0; i<3; i++)
    {
        if( t.isSet(2,i)) t.mat_idx_(2,i)=-1;
        if( t.isSet(i,2)) t.mat_idx_(i,2)=-1;
    }
    t.clear_unused_values();
    return t;
}

template<class container>
void SparseTensor<container>::clear_unused_values()
{
    //now erase unused elements and redefine indices
    std::vector<int> unique_idx;
    for(unsigned i=0; i<3; i++)
        for(unsigned j=0; j<3; j++)
            if(isSet(i,j))
                unique_insert( unique_idx, mat_idx_(i,j));

    std::vector<container> tmp(unique_idx.size());
    for(unsigned i=0; i<unique_idx.size(); i++)
    {
        tmp[i] = values_[unique_idx[i]];
    }
    values_.swap(tmp);
}
///@endcond


//neutral element wrt multiplication represents forms
template<class container>
struct SparseElement
{
    SparseElement(){}
    void set( const container& value){ 
        value_.clear();
        value_.push_back(value);
    }
    const container& getValue( )const { 
        return value_[0];
    }
    container& getValue() {
        return value_[0];
    }
    bool isSet()const{
        if( value_.empty()) return false;
        return true;
    }
    void clear(){value_.clear();}

    private:
    std::vector<container> value_;
};

///@cond
namespace tensor
{

template<class container>
void scal( SparseTensor<container>& t, const SparseElement<container>& e)
{
    if(!e.isSet()) return;
    for( unsigned i=0; i<2; i++)
        for( unsigned j=0; j<2; i++)
            if(t.isSet(i,j)) dg::blas1::pointwiseDot( e.getValue(), t.getValue(i,j), t.getValue(i,j));
}

template<class container>
void multiply( const SparseElement<container>& e, const container& in, container& out)
{
    if(e.isSet()) 
        dg::blas1::pointwiseDot(e.getValue(), in,out);
    else
        out=in;
}

template<class container>
void divide( const container& in, const SparseElement<container>& e, container& out)
{
    if(e.isSet()) 
        dg::blas1::pointwiseDivide(in, e.getValue(),out);
    else
        out=in;
}

///no aliasing allowed
template<class container>
void multiply( const SparseTensor<container>& t, container& in1, container& in2, container& out1, container& out2)
{
    if( !t.isSet(0,0)&& !t.isSet(0,1) && !t.isSet(1,0) &&!t.isSet(1,1)) 
    {
        in1.swap(out1);
        in2.swap(out2);
        return;
    }
    const_multiply(t,in1,in2,out1,out2);
}
///this version keeps the input intact in1 may alias in2
template<class container>
void const_multiply( const SparseTensor<container>& t, const container& in1, const container& in2, container& out1, container& out2)
{
    if( !t.isSet(0,0)&& !t.isSet(0,1) && !t.isSet(1,0) &&!t.isSet(1,1)) 
    {
        out1 = in1;
        out2 = in2;
        return;
    }
    if( !t.isSet(0,0)) out1=in1;
    if(t.isSet( 0,0))
        dg::blas1::pointwiseDot( t.getValue(0,0), in1, out1); 
    if(t.isSet( 0,1))
        dg::blas1::pointwiseDot( 1., t.getValue(0,1), in2, 1., out1);

    if( !t.isSet(1,1)) out2=in2;
    if(t.isSet(1,1))
        dg::blas1::pointwiseDot( t.getValue(1,1), in2, out2);
    if(t.isSet(1,0))
        dg::blas1::pointwiseDot( 1., t.getValue(1,0), in1, 1., out2);
}

template<class container>
void multiply( const SparseTensor<container>& t, container& in1, container& in2, container& in3, container& out1, container& out2, container& out3)
{
    if( t.isEmpty())
    {
        in1.swap(out1);
        in2.swap(out2);
        in3.swap(out3);
        return;
    }
    const_multiply(t,in1,in2,in3,out1,out2,out3);
}
template<class container>
void const_multiply( const SparseTensor<container>& t, const container& in1, const container& in2, const container& in3, container& out1, container& out2, container& out3)
{
    if( t.isEmpty())
    {
        out1 = in1;
        out2 = in2;
        out3 = in3;
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
SparseElement<container> determinant( const SparseTensor<container>& t)
{
    if(t.isEmpty())  return SparseElement<container>();
    SparseTensor<container> d = t.dense();
    container det = d.getValue(0,0);
    std::vector<container> sub_det(3,det);
    dg::blas1::transform( det, det, dg::CONSTANT(0));
    //first compute the det of three submatrices
    dg::blas1::pointwiseDot( d(0,0), d(1,1), sub_det[2]);
    dg::blas1::pointwiseDot( -1., d(1,0), d(0,1), 1. ,sub_det[2]);

    dg::blas1::pointwiseDot( d(0,0), d(2,1), sub_det[1]);
    dg::blas1::pointwiseDot( -1., d(2,0), d(0,1), 1. ,sub_det[1]);

    dg::blas1::pointwiseDot( d(1,0), d(2,1), sub_det[0]);
    dg::blas1::pointwiseDot( -1., d(2,0), d(1,1), 1. ,sub_det[0]);

    //now multiply accordint to Laplace expansion
    dg::blas1::pointwiseDot( 1., d(0,2), sub_det[0], 1.,  det);
    dg::blas1::pointwiseDot(-1., d(1,2), sub_det[1], 1.,  det);
    dg::blas1::pointwiseDot( 1., d(2,2), sub_det[2], 1.,  det);

    return SparseElement<container>(det);
}


}//namespace tensor
///@endcond

}//namespace dg
