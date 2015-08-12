#pragma once

#include "mpi_vector.h"

//the corresponding blas file for the Local matrix must be included before this file
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
    RowDistMat( const LocalMatrix& m, const Collective& c):m_(m), c_(c) { }

    template< class OtherMatrix>
    RowDistMat( const RowDistMat<OtherMatrix, Collective>& src):m_(src.matrix()), c_(src.collective())
    { }
    const LocalMatrix& matrix() const{return m_;}
    const Collective& collective() const{return c_;}
    
    template<class container> 
    void symv( const MPI_Vector<container>& x, MPI_Vector<container>& y)
    {
        assert( x.communicator() == y.communicator());
        assert( x.communicator() == c_.communicator());
        //int rank;
        //MPI_Comm_rank( MPI_COMM_WORLD, &rank);
        //Timer t;
        //t.tic();
        container temp( c_.size()); //only takes time the first time it's called
        //t.toc();
        //if(rank==0)std::cout << "allocat took "<<t.diff()<<"s\n";

        //t.tic();
        c_.collect( x.data(), temp);
        //t.toc();
        //if(rank==0)std::cout << "collect took "<<t.diff()<<"s\n";
        //t.tic();
        dg::blas2::detail::doSymv( m_, temp, y.data(), 
                       typename dg::MatrixTraits<LocalMatrix>::matrix_category(), 
                       typename dg::VectorTraits<container>::vector_category(),
                       typename dg::VectorTraits<container>::vector_category() );
        //t.toc();
        //if(rank==0)std::cout << "symv    took "<<t.diff()<<"s\n";
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
    void symv( const MPI_Vector<container>& x, MPI_Vector<container>& y)
    {
        assert( x.communicator() == y.communicator());
        assert( x.communicator() == c_.communicator());
        container temp( c_.size());
        dg::blas2::detail::doSymv( m_, x.data(), temp, 
                       typename dg::MatrixTraits<LocalMatrix>::matrix_category(), 
                       typename dg::VectorTraits<container>::vector_category(),
                       typename dg::VectorTraits<container>::vector_category() );
        c_.send_and_reduce( temp, y.data());
    }
    private:
    LocalMatrix m_;
    Collective c_;
};

///@cond
template<class L, class C>
struct MatrixTraits<RowDistMat<L, C> >
{
    typedef typename L::value_type value_type;//!< value type
    typedef MPIMatrixTag matrix_category; //!< 
};
///@endcond


//
///@}

} //namespace dg
