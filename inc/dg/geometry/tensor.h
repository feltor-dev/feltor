#pragma once 

#include "dg/backend/operator.h"
#include "dg/functors.h"
#include "dg/blas1.h"

namespace dg
{
    //separate algorithms from interface!!

/**
 * @brief This is a sparse Tensor with only one element i.e. a Form
 *
 * @tparam T a T class
 * @ingroup misc
 */
template<class T>
struct SparseElement
{
    ///create empty object
    SparseElement(){}
    /**
     * @brief copy construct value
     * @param value a value
     */
    SparseElement(const T& value):value_(1,value){ }
    /**
     * @brief Type conversion from other value types
     * @tparam OtherT dg::blas1::transfer must be callable for T and OtherT
     * @param src the source matrix to convert
     */
    template<class OtherT>
    SparseElement( const SparseElement<OtherT>& src)
    {
        if(src.isSet())
        {
            value_.resize(1);
            dg::blas1::transfer(src.value(), value_[0]);
        }
    }

    ///@brief Read access
    ///@return read access to contained value
    const T& value( )const { 
        return value_[0];
    }
    /**
     * @brief write access, create a T if there isn't one already
     * @return write access, always returns a T 
     */
    T& value() {
        if(!isSet()) value_.resize(1);
        return value_[0];
    }

    /**
     * @brief check if an element is set or not
     * @return false if the value array is empty
     */
    bool isSet()const{
        if( value_.empty()) return false;
        return true;
    }
    ///@brief Clear contained value
    void clear(){value_.clear();}
    private:
    std::vector<T> value_;
};

/**
* @brief Class for 2x2 and 3x3 matrices sharing or implicitly assuming elements 
*
* This class enables shared access to stored Ts 
* or not store them at all since the storage of (and computation with) a T is expensive.

* This class contains both a (dense) matrix of integers.
* If positive or zero, the integer represents a gather index into the stored array of Ts, 
if negative the value of the T is assumed to be 1, except for the off-diagonal entries
    in the matrix where it is assumed to be 0.
* We then only need to store non-trivial and non-repetitive Ts.
* @tparam T must be default constructible and copyable.
* @ingroup misc
*/
template<class T>
struct SparseTensor
{
    ///no element is set
    SparseTensor( ):mat_idx_(3,-1.) {}

    /**
     * @brief reserve space for value_size Ts in the values array
     * @param value_size reserve space for this number of Ts (default constructor) 
     */
    SparseTensor( unsigned value_size): mat_idx_(3,-1.), values_(value_size){}

    /**
    * @brief pass array of Ts
    * @param values The contained Ts are stored in the object
    */
    SparseTensor( const std::vector<T>& values ): mat_idx_(3,-1.), values_(values){}

    /**
     * @brief Type conversion from other value types
     * @tparam OtherT dg::blas1::transfer must be callable for T and OtherT
     * @param src the source matrix to convert
     */
    template<class OtherT>
    SparseTensor( const SparseTensor<OtherT>& src): mat_idx_(3,-1.), values_(src.values().size()){
        for(unsigned i=0; i<3; i++)
            for(unsigned j=0; j<3; j++)
                mat_idx_(i,j)=src.idx(i,j);

        for( unsigned i=0; i<src.values().size(); i++)
            dg::blas1::transfer( src.values()[i], values_[i]);
    }

    /**
    * @brief check if a value is set at the given position or not
    * @param i row index 0<=i<3
    * @param j col index 0<=j<3
    * @return true if T is non-empty, false if value is assumed implicitly
    */
    bool isSet(size_t i, size_t j)const{
        if( mat_idx_(i,j) <0) return false;
        return true;
    }

    /**
    * @brief read index into the values array at the given position
    * @param i row index 0<=i<3
    * @param j col index 0<=j<3
    * @return -1 if !isSet(i,j), index into values array else
    */
    int idx(unsigned i, unsigned j)const{return mat_idx_(i,j);}
    /**
    * @brief write index into the values array at the given position
    *
    * use this and the value() member to assemble the tensor
    * @param i row index 0<=i<3
    * @param j col index 0<=j<3
    * @return write access to value index to be set
    */
    int& idx(unsigned i, unsigned j){return mat_idx_(i,j);}
     /*! @brief unset an index, does not clear the associated value
      *
      * @param i row index 0<=i<3
      * @param j col index 0<=j<3
      */
     void unset( unsigned i, unsigned j) {
         mat_idx_(i,j)=-1;
     }
     /**
      * @brief clear any unused values and reset the corresponding indices
      *
      * This function erases all values that are unreferenced by any index and appropriately redefines the remaining indices
      */
     void clear_unused_values();

    /*!@brief Read access the underlying T
     * @return if !isSet(i,j) the result is undefined, otherwise values[idx(i,j)] is returned. 
     * @param i row index 0<=i<3
     * @param j col index 0<=j<3
     * @note If the indices fall out of range of index the result is undefined
     */
    const T& value(size_t i, size_t j)const{ 
        int k = mat_idx_(i,j);
        return values_[k];
    }
    //if you're looking for this function: YOU DON'T NEED IT!!ALIASING
    //T& value(size_t i, size_t j);
    /**
     * @brief Return the T at given position, create one if there isn't one already
     * @param i index into the values array
     * @return  always returns a T 
     */
    T& value( size_t i)
    {
        if(i>=values_.size() ) values_.resize(i+1);
        return values_[i];
    }
    /**
     * @brief Return read access to the values array
     * @return read access to values array
     */
    const std::vector<T>& values()const{return values_;}
    ///clear all values, Tensor is empty after that
    void clear(){
        mat_idx_=dg::Operator<int>(3,-1);
        values_.clear();
    }

    /**
    * @brief Test the matrix for emptiness
    *
    * The matrix is empty if !isSet(i,j) for all i and j
    * @return true if no value in the matrix is set
    */
    bool isEmpty()const{ 
        bool empty=true;
        for(unsigned i=0; i<3; i++)
            for( unsigned j=0; j<3; j++)
                if( isSet(i,j) ) 
                    empty=false;
        return empty;
    }

    /**
    * @brief Test if all elements are set
    *
    * The matrix is dense if isSet(i,j) for all i and j
    * @return true if all values in the matrix are set
    */
    bool isDense()const{
        bool dense=true;
        for(unsigned i=0; i<3; i++)
            for( unsigned j=0; j<3; j++)
                if( !isSet(i,j) ) 
                    dense=false;
        return dense;
    }

    /**
    * @brief Test if the elements in the third dimension are unset
    *
    * The matrix is perpendicular if !isSet(i,j) for any i,j=2
    * @return true if perpenicular
    * */
    bool isPerp() const {
        bool empty=true;
        for(unsigned i=0; i<3; i++)
        {
            if( isSet(i,2) || isSet(2,i)) 
                empty=false;
        }
        return empty;
    }
    /**
    * @brief Test if no off-diagonals are set
    *
    * The matrix is diagonal if no off-diagonal element is set
    * @return true if no off-diagonal element is set
    */
    bool isDiagonal()const{
        bool diagonal=true;
        for(unsigned i=0; i<3; i++)
            for(unsigned j=i+1; j<3; j++)
                if( isSet(i,j) ||  isSet(j,i))
                    diagonal=false;
        return diagonal;
    }
     
     ///construct an empty Tensor
     SparseTensor empty()const{return SparseTensor();}
     ///copy and erase all values in the third dimension
     ///@note calls clear_unused_values() to get rid of the elements
     SparseTensor perp()const;

    /**
     * @brief Return the transpose of the currrent tensor
     * @return swapped rows and columns
     */
    SparseTensor transpose()const{
        SparseTensor tmp(*this);
        tmp.mat_idx_ = mat_idx_.transpose();
        return tmp;
    }

    private:
    dg::Operator<int> mat_idx_;
    std::vector<T> values_;
    void unique_insert(std::vector<int>& indices, int& idx);
};

namespace tensor
{
 /**
 * @brief Construct a tensor with all unset values filled with explicit 0 or 1
 *
 * @copydoc hide_container
 * @return a dense tensor
 * @note undefined if t.isEmpty() returns true
 */
template<class container> 
SparseTensor<container> dense(const SparseTensor<container>& tensor)
{
    SparseTensor<container> t(tensor);
    if( t.isEmpty()) throw Error(Message(_ping_)<< "Can't make an empty tensor dense! ") ;
    container tmp = t.values()[0];
    //1. diagonal
    size_t size= t.values().size();
    bool diagonalIsSet=true;
    for(unsigned i=0; i<3; i++)
        if(!t.isSet(i,i)) diagonalIsSet = false;
    if (!diagonalIsSet){
        dg::blas1::transform( tmp, tmp, dg::CONSTANT(1));
        t.value(size)=tmp;
        for(unsigned i=0; i<3; i++)
            if(!t.isSet(i,i)) t.idx(i,i) = size;
    }
    //2. off-diagonal
    size = t.values().size();
    bool offIsSet=true;
    for(unsigned i=0; i<3; i++)
        for(unsigned j=0; j<3; j++)
            if( !t.isSet(i,j) ) offIsSet=false;
    if (!offIsSet){
        dg::blas1::transform( tmp, tmp, dg::CONSTANT(0));
        t.value(size)=tmp;
        for(unsigned i=0; i<3; i++)
            for(unsigned j=0; j<3; j++)
                if(!t.isSet(i,j) ) t.idx(i,j) = size;
    }
    return t;
}
}//namespace tensor

/**
 * @brief data structure to hold the LDL^T decomposition of a symmetric positive definite matrix
 *
 * LDL^T stands for a lower triangular matrix L,  a diagonal matrix D and the transpose L^T
 * @copydoc hide_container
 * @attention the tensor in the Elliptic classes actually only need to be positive **semi-definite**
 * and unfortunately the decomposition is unstable for semi-definite matrices.
* @ingroup misc
 */
template<class container>
struct CholeskyTensor
{
    /**
     * @brief decompose given tensor
     *
     * @param in must be symmetric and positive definite
     */
    CholeskyTensor( const SparseTensor<container>& in) {
        decompose(in);
    }
    /**
     * @brief Type conversion from other value types
     * @tparam OtherContainer dg::blas1::transfer must be callable for container and OtherContainer
     * @param in the source matrix to convert
     */
    template<class OtherContainer>
    CholeskyTensor( const CholeskyTensor<OtherContainer>& in):q_(in.lower()),diag_(in.diagonal()),upper_(in.upper()) { }

    /**
     * @brief decompose given tensor
     *
     * overwrites the existing decomposition
     * @param in must be symmetric and positive definite
     */
    void decompose( const SparseTensor<container>& in)
    {
        SparseTensor<container> denseIn=dg::tensor::dense(in);
        /*
         * One nice property of positive definite is that the diagonal elements are 
         * greater than zero.
         */

        if(in.isSet(0,0))
        {
            diag_.idx(0,0)=0;
            diag_.value(0)=in.value(0,0);
        }
        if(in.isSet(1,0))
        {
            container tmp=in.value(1,0);
            if(diag_.isSet(0,0)) dg::blas1::pointwiseDivide(tmp,diag_.value(0,0),tmp);
            q_.idx(1,0)=0;
            q_.value(0)=tmp;
        }
        if(in.isSet(2,0))
        {
            container tmp=in.value(2,0);
            if(diag_.isSet(0,0))dg::blas1::pointwiseDivide(tmp,diag_.value(0,0),tmp);
            q_.idx(2,0)=1;
            q_.value(1)=tmp;
        }

        if( q_.isSet(1,0) || in.isSet(1,1))
        {
            SparseTensor<container> denseL=dg::tensor::dense(q_);
            container tmp=denseL.value(1,0);
            dg::blas1::pointwiseDot(tmp,tmp,tmp);
            if(diag_.isSet(0,0)) dg::blas1::pointwiseDot(tmp,diag_.value(0,0),tmp);
            dg::blas1::axpby( 1., denseIn.value(1,1), -1., tmp, tmp);
            diag_.idx(1,1)=1;
            diag_.value(1) = tmp;
        }

        if( in.isSet(2,1) || (q_.isSet(2,0)&&q_.isSet(1,0)))
        {
            SparseTensor<container> denseL=dg::tensor::dense(q_);
            container tmp=denseIn.value(2,1);
            dg::blas1::pointwiseDot(denseL.value(2,0), denseL.value(1,0), tmp);
            if(diag_.isSet(0,0))dg::blas1::pointwiseDot(tmp, diag_.value(0,0), tmp);
            dg::blas1::axpby(1., denseIn.value(2,1),-1.,tmp, tmp);
            if(diag_.isSet(1,1))dg::blas1::pointwiseDivide(tmp, diag_.value(1,1),tmp);
            q_.idx(2,1)=2;
            q_.value(2)=tmp;
        }
        if( in.isSet(2,2) || q_.isSet(2,0) || q_.isSet(2,1))
        {
            SparseTensor<container> denseL=dg::tensor::dense(q_);
            container tmp=denseL.value(2,0), tmp1=denseL.value(2,1);
            dg::blas1::pointwiseDot(tmp,tmp,tmp);
            if(diag_.isSet(0,0))dg::blas1::pointwiseDot(diag_.value(0,0),tmp,tmp);
            dg::blas1::pointwiseDot(tmp1,tmp1,tmp1);
            if(diag_.isSet(1,1))dg::blas1::pointwiseDot(diag_.value(1,1),tmp1,tmp1);
            dg::blas1::axpby(1., denseIn.value(2,2), -1., tmp, tmp);
            dg::blas1::axpby(1., tmp, -1., tmp1, tmp);
            diag_.idx(2,2)=2;
            diag_.value(2) = tmp;
        }
        diag_.clear_unused_values();
        q_.clear_unused_values();
        upper_=q_.transpose();
    }

    /**
     * @brief Returns L
     * @return a lower triangular matrix with 1 on diagonal
     */
    const SparseTensor<container>& lower()const{
        return q_;

    }
    /**
     * @brief Returns L^T
     * @return a upper triangular matrix with 1 on diagonal
     */
    const SparseTensor<container>& upper()const{
        return upper_;

    }

    /**
     * @brief Returns D
     * @return only diagonal elements are set if any
     */
    const SparseTensor<container>& diagonal()const{return diag_;}

    private:
    SparseTensor<container> q_, diag_, upper_;
    bool lower_;
};

///@cond

template<class container>
void SparseTensor<container>::unique_insert(std::vector<int>& indices, int& idx) 
{
    bool unique=true;
    unsigned size=indices.size();
    for(unsigned i=0; i<size; i++)
        if(indices[i] == idx) {
            unique=false;
            idx=i;
        }
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
        t.mat_idx_(2,i)=-1;
        t.mat_idx_(i,2)=-1;
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
        tmp[i] = values_[unique_idx[i]];
    values_.swap(tmp);
}
///@endcond


}//namespace dg
