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
*/
template<class container>
struct SharedContainers
{
    /**
    * @brief check if an index is set or not
    * @param i row index 0<i<2
    * @param j col index 0<j<2
    * @return true if container is non-empty, false if value is assumed implicitly
    */
    bool isSet(size_t i, size_t j)const{
        if( get_value_idx(i,j) <0) return false;
        return true;
    }
    bool isSet(size_t i) const{
        if( get_value_idx(i)<0) return false;
        return true;
    }
    /// Access the underlying container
    /// @return if !isSet(i,j) the default constructor of container is called, otherwise values[mat_idx(i,j)] is returned
    const container& getValue(size_t i, size_t j)const{ 
        int k = get_value_idx(i,j);
        if(k<0) return container();
        return values_[k];
    }
    /// Access the underlying container
    /// @return if !isSet(i) the default constructor of container is called, otherwise values[vec_idx(i)] is returned
    const container& getValue(size_t i)const{ 
        int k = get_value_idx(i);
        if(k<0) return container();
        return values_[k];
    }
    protected:
    ///by protecting the constructor we disallow changing anything through a base class reference
    SharedContainers( const dg::Operator<int>& mat_idx, std::vector<int>& vec_idx, std::vector<container>& values ): mat_idx_(mat_idx), vec_idx(vec_idx), values_(values){}
    ///we disallow changeing the size
    SharedContainers( const SharedContainers& src):  mat_idx_(src.mat_idx_), vec_idx_(src.vec_idx_), values_(src.values_){}
    ///we disallow changeing the size
    SharedContainers& operator=()(SharedContainers src)
    {
        values_.swap( src.values_);
        std::swap( mat_idx_, src.mat_idx_)
        std::swap( vec_idx_, src.vec_idx_)
        return *this;
    }
    ///rule of three
    ~SharedContainers(){}
    template<class otherContainer>
    SharedContainers( const SharedContainers<otherContainer>& src): mat_idx_(src.mat_idx()), vec_idx_(src.vec_idx()), values_(src.values().size()), idx_(src.idx_){
        dg::blas1::transfer( src.values(), values_);
    }
    const dg::Operator<int>& mat_idx() const {return mat_idx_;}
    const std::vector<int>& vec_idx() const {return vec_idx_;}
    const std::vector<container>& values() const{return values_;}
    private:
    int get_value_idx(size_t i, size_t j)const{return mat_idx_(i,j);}
    int get_value_idx(size_t i)const{return vec_idx_[i];}
    dg::Operator<int> mat_idx_;
    std::vector<int> vec_idx_;
    std::vector<container> values_;
};

///The metric tensor is a SparseTensor 
template<class container>
struct MetricTensor: public SharedContainers<container>
{
    /// default size is 3 and no values are set
    MetricTensor( ):SharedContainers( 11){ volIdx_=perpVolIdx_=-1; }
    bool volIsSet() const{ 
        if(volIdx_<0)return false;
        else return true;
    }
    template<class otherContainer>
    MatricTensor( MetricTensor<otherContainer>& src):SharedContainers<container>(src){}
    bool perpVolIsSet() const{ 
        if(perpVolIdx_<0)return false;
        else return true;
    }
    int& volIdx(){ return volIdx_;}
    int& perpVolIdx(){ return perpVolIdx_;}
    int volIdx()const{ return volIdx_;}
    int perpVolIdx()const{ return perpVolIdx_;}

    const container& getVol() const{return get(volIdx_);}
    const container& getPerpVol() const{return get(perpVolIdx_);}
    void setVol(const container& vol){ 
        volIdx_=9;
        this->get(9) = value; 
    }
    void setPerpVol(const container& vol){ 
        perpVolIdx_=10;
        this->get(10) = value; 
    }
    private:
    int volIdx_, perpVolIdx_;
};

template<class container>
struct CoordinateTransformation: public SharedContainers<container>
{
    /// the coordinates have index 9, 10 and 11
    CoordinateTransformation():SharedContainers( 12){}
};

namespace geo
{

///container sizes must match each other and the tensor container size
///may destroy input (either swap in or use as temporary storage)
///no alias allowed
template<class container>
void multiply( const SharedContainers& t, container& in1, container& in2, container& out1, container& out2)
{
    if( !tensor.isSet(0,0) && !tensor.isSet(0,1) && !tensor.isSet(1,0) && !tensor.isSet(1,1))
    {
        out1.swap(in1);
        out2.swap(in2);
        return;
    }
    if( !tensor.isSet(0,0)) out1=in1;
    else 
        dg::blas1::pointwiseDot( tensor.getValue(0,0), in1, out1); //gxx*v_x
    if(tensor.isSet( 0,1))
        dg::blas1::pointwiseDot( 1., tensor.getValue(0,1), in2, 1., out1);//gxy*v_y

    if( !tensor.isSet(1,1)) out2=in2;
    else
        dg::blas1::pointwiseDot( tensor.getValue(1,1), in2, out2); //gyy*v_y
    if(tensor.isSet(1,0))
        dg::blas1::pointwiseDot( 1., tensor.getValue(1,0), in1, 1., out2); //gyx*v_x
}

template<class container>
void multiply( const SharedContainers& t, container& in1, container& in2, container& in3, container& out1, container& out2, container& out3)
{
    bool empty = true;
    for( unsigned i=0; i<3; i++)
        for( unsigned j=0; j<3; j++)
            if(tensor.isSet(i,j)) 
                empty=false;
    if( empty)
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

}//namespace geo
}//namespace dg
