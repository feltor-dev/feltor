#pragma once 

#include <cusp/coo_matrix.h>
#include "dg/backend/operator.h"
#include "dg/blas1.h"

namespace dg
{
/**
* @brief Abstract base class for Sparse matrices of containers
*
* The size is always assumed 3x3.
* The elements are implicitly assumed either 1 (diagonal) or 0 (off-diagonal)
*/
template<class container>
struct SparseUnitTensor
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
    /**
    * @brief set an index into the values array (if <0 then value is assumed implicitly)
    * @param i row index 0<i<2
    * @param j col index 0<j<2
    * @return reference to an index     
    */
    int& valueIdx(size_t i, size_t j){ return get_value_idx(i,j); }
    /**
    * @brief get an index into the values array (if <0 then value is assumed implicitly)
    * @param i row index 0<i<2
    * @param j col index 0<j<2
    * @return reference to an index     
    */
    int valueIdx(size_t i, size_t j)const{ return get_value_idx(i,j); }
    const container& getValue(size_t i, size_t j)const{ 
        int k = get_value_idx(i,j);
        if(k<0) return container();
        return getValue(k); 
    }
    const container& getValue(size_t i)const{ return values_[i]; }
    ///set a value for the given idx
    container& getValue(int i){ return values_[i]; }
    protected:
    ///by protecting the constructor we fix the size of the values 
    ///array associated with the types of derived classes
    SparseUnitTensor( size_t values_size=9): values_(values_size),idx_(3,-1){}
    ///we disallow changeing the size
    SparseUnitTensor( const SparseUnitTensor& src): values_(src.values_), idx_(src.idx_){}
    ///we disallow changeing the size
    SparseUnitTensor& operator=()(SparseUnitTensor src)
    {
        values_.swap( src.values_);
        std::swap( idx_, src.idx_)
        return *this;
    }
    ///rule of three
    ~SparseUnitTensor(){}
    template<class otherContainer>
    SparseUnitTensor( const SparseUnitTensor<otherContainer>& src): values_(src.values_.size()), idx_(src.idx_){
        dg::blas1::transfer( src.values_, values_);
    }
    private:
    int get_value_idx(size_t i, size_t j)const{return idx_(i,j);}
    int& get_value_idx(size_t i, size_t j){return idx_(i,j);}
    std::vector<container> values_;
    dg::Operator<int>& idx_;
};

///The metric tensor is a SparseTensor 
template<class container>
struct MetricTensor: public SparseUnitTensor<container>
{
    /// default size is 3 and no values are set
    MetricTensor( ):SparseUnitTensor( 11){ volIdx_=perpVolIdx_=-1; }
    bool volIsSet() const{ 
        if(volIdx_<0)return false;
        else return true;
    }
    template<class otherContainer>
    MatricTensor( MetricTensor<otherContainer>& src):SparseUnitTensor<container>(src){}
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
struct CoordinateTransformation: public SparseUnitTensor<container>
{
    /// the coordinates have index 9, 10 and 11
    CoordinateTransformation():SparseUnitTensor( 12){}
};

namespace geo
{

///container sizes must match each other and the tensor container size
///may destroy input (either swap in or use as temporary storage)
///no alias allowed
template<class container>
void multiply( const SparseUnitTensor& t, container& in1, container& in2, container& out1, container& out2)
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
void multiply( const SparseUnitTensor& t, container& in1, container& in2, container& in3, container& out1, container& out2, container& out3)
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
