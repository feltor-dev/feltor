#pragma once 

#include <cusp/coo_matrix.h>
#include "dg/backend/operator.h"
#include "dg/blas1.h"

namespace dg
{
///sparse matrix with implicit elements either 1 (diagonal) or 0 (off-diagonal)
///the size is always 3x3 
template<class container>
struct SparseUnitTensor
{
    /// default size is 3 and no values are set
    SparseUnitTensor( ):idx_(3,-1),values_(9){}
    ///swap all data with other tensor
    void swap( SparseUnitTensor& other){
        std::swap(idx_,other.idx_);
        values_.swap( other.values_);
    }
    /**
    * @brief get an index into the values array
    * @param i row index 0<i<2
    * @param j col index 0<j<2
    * @return index (if <0 then value is assumed implicitly)
    */
    int operator()(size_t i, size_t j)const{return idx_(i,j);}
    /**
    * @brief set an index into the values array (if <0 then value is assumed implicitly)

    * @param i row index 0<i<2
    * @param j col index 0<j<2
    * @return reference to an index     
    */
    int& operator()(size_t i, size_t j){return idx_(i,j);}
    ///any value between 0 and 8
    const container& get(int i)const{return values_[i];}
    ///any value between 0 and 8
    container& get(int i){return values_[i];}
    private:
    dg::Operator<int>& idx_;
    std::vector<container> values_;

};
namespace geo
{

///container sizes must match each other and the tensor container size
///may destroy input (either swap in or use as temporary storage)
///no alias allowed
template<class container>
void multiply( const SparseUnitTensor& t, container& in1, container& in2, container& out1, container& out2)
{
    if( tensor(0,0)<0 && tensor(0,1)<0 && tensor(1,0)<0 && tensor(1,1)<0)
    {
        out1.swap(in1);
        out2.swap(in2);
    }
    if( tensor(0,0)<0) out1=in1;
    else 
        dg::blas1::pointwiseDot( tensor.get(tensor(0,0)), in1, out1); //gxx*v_x
    if(!tensor( 0,1)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(0,1)), in2, 1., out1);//gxy*v_y

    if( tensor(1,1)<0) out2=in2;
    else
        dg::blas1::pointwiseDot( tensor.get(tensor(1,1)), in2, out2); //gyy*v_y
    if(!tensor(1,0)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(1,0)), in1, 1., out2); //gyx*v_x
}

template<class container>
void multiply( const SparseUnitTensor& t, container& in1, container& in2, container& in3, container& out1, container& out2, container& out3)
{
    bool empty = true;
    for( unsigned i=0; i<3; i++)
        for( unsigned j=0; j<3; j++)
            if(!tensor(i,j)<0) 
                empty=false;

    if( empty)
    {
        in1.swap(out1);
        in2.swap(out2);
        in3.swap(out3);
    }
    if( tensor(0,0)<0) out1=in1;
    if(!tensor( 0,0)<0)
        dg::blas1::pointwiseDot( tensor.get(tensor(0,0)), in1, out1); 
    if(!tensor( 0,1)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(0,1)), in2, 1., out1);
    if(!tensor( 0,2)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(0,2)), in3, 1., out1);

    if( tensor(1,1)<0) out2=in2;
    if(!tensor(1,1)<0)
        dg::blas1::pointwiseDot( tensor.get(tensor(1,1)), in2, out2);
    if(!tensor(1,0)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(1,0)), in1, 1., out2);
    if(!tensor(1,2)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(1,2)), in1, 1., out2);

    if( tensor(2,2)<0) out3=in3;
    if(!tensor(2,2)<0)
        dg::blas1::pointwiseDot( tensor.get(tensor(2,2)), in3, out3);
    if(!tensor(2,1)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(2,1)), in2, 1., out3);
    if(!tensor(2,0)<0)
        dg::blas1::pointwiseDot( 1., tensor.get(tensor(2,0)), in1, 1., out3);
}

}//namespace geo




}//namespace dg
