#pragma once

#include "grid.h"
#include "operator.h"
#include "evaluation.h"
#include "dg/functors.h"
#include "dg/blas1.h"

namespace dg
{
    //separate algorithms from interface!!

/**
* @brief Class for 2x2 and 3x3 matrices sharing or implicitly assuming elements
*
* This class enables shared access to stored Ts
* or not store them at all since the storage of (and computation with) a T is expensive.

* This class contains a (dense) 3x3 matrix of integers.
* If positive or zero, the integer represents a gather index into the stored array of Ts,
if negative the value of the T is assumed to be 1, except for the off-diagonal entries
    in the matrix where it is assumed to be 0.
* We then only need to store non-trivial and non-repetitive Ts.
* @tparam container must be default constructible and copyable.
* @ingroup misc
*/
template<class container>
struct SparseTensor
{
    ///no value is set, Indices default to 0
    SparseTensor( ):m_mat_idx(3,0) {}

    /**
     * @brief Construct the unit tensor
     * @param grid used to create explicit zeroes and ones
     */
    template<class Topology>
    SparseTensor( const Topology& grid){
		construct(grid);
    }

    /**
    * @brief pass array of containers, Indices default to 0
    * @param values The contained containers are
    */
    SparseTensor( std::vector<container> values ): m_mat_idx(3, 0), m_values(values){}
    /**
    * @brief pass array of containers, Indices default to 0
    * @param values The contained Ts are stored in the object
    */
    void construct( const std::vector<container>& values ){
        m_mat_idx = dg::Operator<int>(3,0);
	    m_values=values;
    }
    /**
     * @brief Construct the unit tensor
     * @param grid used to create explicit zeroes and ones
     */
    template<class Topology>
    void construct( const Topology& grid){
		m_mat_idx.resize(3,0);
   		m_values.resize(2);
        dg::transfer( dg::evaluate( dg::zero, grid), m_values[0]);
        dg::transfer( dg::evaluate( dg::one, grid), m_values[1]);
        for( int i=0; i<3; i++)
            m_mat_idx( i,i) = 1;
    }

    /**
     * @brief Type conversion from other value types
     * @tparam OtherContainer \c dg::blas1::transfer must be callable for \c container and \c OtherContainer
     * @param src the source matrix to convert
     */
    template<class OtherContainer>
    SparseTensor( const SparseTensor<OtherContainer>& src): m_mat_idx(3,0), m_values(src.values().size()){
        for(unsigned i=0; i<3; i++)
            for(unsigned j=0; j<3; j++)
                m_mat_idx(i,j)=src.idx(i,j);

        for( unsigned i=0; i<src.values().size(); i++)
            dg::blas1::transfer( src.values()[i], m_values[i]);
    }

    /**
    * @brief read index into the values array at the given position
    * @param i row index 0<=i<3
    * @param j col index 0<=j<3
    * @return -1 if \c !isSet(i,j), index into values array else
    */
    int idx(unsigned i, unsigned j)const{
        return m_mat_idx(i,j);
    }
    /**
    * @brief write index into the values array at the given position
    *
    * use this and the \c values() member to assemble the tensor
    * @param i row index 0<=i<3
    * @param j col index 0<=j<3
    * @return write access to value index to be set
    */
    int& idx(unsigned i, unsigned j){
        return m_mat_idx(i,j);
    }

    /*!@brief Read access the underlying container
     * @return if !isSet(i,j) the result is undefined, otherwise values[idx(i,j)] is returned.
     * @param i row index 0<=i<3
     * @param j col index 0<=j<3
     * @note If the indices fall out of range of index the result is undefined
     */
    const container& value(size_t i, size_t j)const{
        int k = m_mat_idx(i,j);
        return m_values[k];
    }
    //if you're looking for this function: YOU DON'T NEED IT!!ALIASING
    //T& value(size_t i, size_t j);
    /**
     * @brief Return write access to the values array
     * @return write access to values array
     */
    std::vector<container>& values() {
        return m_values;
    }
    /**
     * @brief Return read access to the values array
     * @return read access to values array
     */
    const std::vector<container>& values()const{
        return m_values;
    }

    /**
     * @brief Return the transpose of the currrent tensor
     * @return swapped rows and columns
     */
    SparseTensor transpose()const{
        SparseTensor tmp(*this);
        tmp.m_mat_idx = m_mat_idx.transpose();
        return tmp;
    }

    private:
    dg::Operator<int> m_mat_idx;
    std::vector<container> m_values;
};

}//namespace dg
