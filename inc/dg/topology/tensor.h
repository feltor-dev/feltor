#pragma once

#include "grid.h"
#include "operator.h"
#include "evaluation.h"
#include "evaluationX.h"
#include "dg/functors.h"
#include "dg/blas1.h"

/*!@file
 *
 * SparseTensor
 */

namespace dg
{
    //separate algorithms from interface!!

/**
* @brief Class for 2x2 and 3x3 matrices sharing elements
*
* This class enables shared access to stored containers.
* It contains a (dense) 3x3 matrix of integers that is automatically allocated.
* The integers represent a gather index into a stored array of containers.
* In this way duplicate entries are stored only once, which helps to
* avoid unnecessary memory accesses.
* For example an orthogonal metric is represented as follows
* \f[
* \begin{pmatrix}
* g^{xx} & 0 & 0\\
* 0 & g^{yy} & 0 \\
* 0 & 0 & g^{zz}
* \end{pmatrix}
* \quad\rightarrow\quad
* \text{idx} = \begin{pmatrix}
* 1 & 0 & 0 \\
* 0 & 2 & 0 \\
* 0 & 0 & 3
* \end{pmatrix} \quad \text{values} = \begin{pmatrix}
* 0 & g^{xx} & g^{yy} & g^{zz}
* \end{pmatrix}
* \f]
* which in code can be assembled as
* @snippet{trimleft} multiply_t.cpp sparse tensor
* @tparam container must be default constructible and copyable.
* @ingroup tensor
* @sa dg::tensor
*/
template<class container>
struct SparseTensor
{
    using container_type = container;
    ///no value is set, Indices default to -1
    SparseTensor( ):m_mat_idx(3,-1) {}

    /**
     * @brief Construct the unit tensor
     * @param grid used to create explicit zeroes (Index 0) and ones (Index 1)
     */
    template<class Topology>
    SparseTensor( const Topology& grid){
		construct(grid);
    }

    /**
    * @brief Construct the unit tensor
    * @param copyable used to create explicit zeroes (Index 0) and ones (Index 1)
    */
    SparseTensor( const container& copyable ){
        construct(copyable);
    }
    /**
     * @brief Construct the unit tensor
     * @param grid used to create explicit zeroes (Index 0) and ones (Index 1)
     */
    template<class Topology>
    void construct( const Topology& grid){
		m_mat_idx.resize(3,0);
        for( int i=0; i<3; i++)
            m_mat_idx( i,i) = 1;
   		m_values.resize(2);
        dg::assign( dg::evaluate( dg::zero, grid), m_values[0]);
        dg::assign( dg::evaluate( dg::one, grid), m_values[1]);
    }
    /**
    * @brief Construct the unit tensor
    * @param copyable used to create explicit zeroes (Index 0) and ones (Index 1)
    */
    void construct( const container& copyable ){
		m_mat_idx.resize(3,0);
        for( int i=0; i<3; i++)
            m_mat_idx( i,i) = 1;
        m_values.assign(2,copyable);
        dg::blas1::copy( 0., m_values[0]);
        dg::blas1::copy( 1., m_values[1]);
    }

    /**
     * @brief Type conversion from other value types
     * @tparam OtherContainer \c dg::assign must be callable for \c container and \c OtherContainer
     * @param src the source matrix to convert
     */
    template<class OtherContainer>
    SparseTensor( const SparseTensor<OtherContainer>& src): m_mat_idx(3,-1), m_values(src.values().size()){
        for(unsigned i=0; i<3; i++)
            for(unsigned j=0; j<3; j++)
                m_mat_idx(i,j)=src.idx(i,j);

        for( unsigned i=0; i<src.values().size(); i++)
            dg::assign( src.values()[i], m_values[i]);
    }

    /**
    * @brief read index into the values array at the given position
    * @param i row index 0<=i<3
    * @param j col index 0<=j<3
    * @return index into values array
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
     * @return \c values[idx(i,j)] is returned.
     * @param i row index 0<=i<3
     * @param j col index 0<=j<3
     * @note If the indices \c (i,j) fall out of range
     * or if the corresponding value index \c idx(i,j) falls out of range of the values array, the result is undefined
     */
    const container& value(size_t i, size_t j)const{
        int k = m_mat_idx(i,j);
        return m_values[k];
    }
    //if you're looking for this function: YOU DON'T NEED IT!! (ALIASING)
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
    dg::SquareMatrix<int> m_mat_idx;
    std::vector<container> m_values;
};

}//namespace dg
