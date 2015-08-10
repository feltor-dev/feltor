#pragma once

#include "mpi_vector.h"

namespace dg
{

///@addtogroup mpi_structures
///@{

/**
* @brief Distributed memory matrix class
*
* The idea of this mpi matrix is to separate communication and computation in order to resuse existing optimized matrix formats for the computation. 
*It can be expected that this works particularly well for cases in which the communication to computation ratio is low. 
* @tparam LocalMatrix The class of the matrix for local computations. 
 symv needs to be callable on the container class of the MPI_Vector
* @tparam Collective The Communication class needs to scatter and gather values across processes. Three functions need to be callable
void gather( const container& input, container& values)
should gather all values from the input vector that are necessary for the following computation
void scatter( const container& values, container& output)
is its inverse. It scatters the result of a computation across the processes and reduces the values on double indices. Strictly speaking it's the inverse of the gather operation only if the gather map is bijectiv (no reduction necessary). 
int size()
gives the local size of the values vector.
*/
template<class LocalMatrix, class Collective >
struct RowDistMat
{
    /**
    * @brief Constructor 
    *
    * @param m The local matrix
    * @param c The communication object
    */
    RowDistMat( const LocalMatrix& m, const Collective& c):m_(m), c_(c)
    { }
    template< class OtherMatrix>
    RowDistMat( const RowDistMat<OtherMatrix, Collective>& src):m_(src.matrix()), c_(src.collective())
    { }
    const LocalMatrix& matrix() const{return m_;}
    const Collective& collective() const{return c_;}
    
    template<class container> 
    void symv( MPI_Vector<container>& x, MPI_Vector<container>& y)
    {
        assert( x.communicator() == y.communicator());
        assert( x.communicator() == c_.communicator());
        container temp( c_.size());
        c_.gather( x.data(), temp);
        symv( m_, temp, y);
    }

        
    private:
    LocalMatrix m_;
    Collective c_;
};

template<class LocalMatrix, class Collective >
struct ColDistMat
{
    /**
    * @brief Constructor 
    *
    * @param m The local matrix
    * @param c The communication object
    */
    ColDistMat( const LocalMatrix& m, const Collective& c):m_(m), c_(c)
    { }
    
    template<class container> 
    void symv( MPI_Vector<container>& x, MPI_Vector<container>& y)
    {
        container temp( c_.size());
        symv( m_, x, temp);
        c_.scatter( temp, y);
    }
    private:
    LocalMatrix m_;
    Collective c_;
};

//
///@}

} //namespace dg
