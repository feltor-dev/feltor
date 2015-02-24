#pragma once

#include <vector>

#include "mpi_grid.h"
#include "mpi_vector.h"
#include "mpi_precon_blas.h"
#include "operator.h"


namespace dg
{
///@addtogroup mpi_structures
///@{
//
/**
 * @brief Struct holding the boundary blocks 
 */
struct BoundaryTerms
{
    std::vector<std::vector<double> > data_;//!<  blocks of data
    std::vector<int> row_; //!<row of data in 1D without block
    std::vector<int> col_; //!<col of data in 1D without block
    /**
     * @brief Apply the boundary terms in X
     *
     * @param x
     * @param y
     */
    void applyX( const MPI_Vector& x, MPI_Vector& y) const
    {
        if(data_.empty()) return;
        unsigned rows = x.Ny(), cols = x.Nx(), n = x.n();
        for( unsigned m=0; m<data_.size(); m++) //all blocks
        {
            for( unsigned s=0; s<x.Nz(); s++) //z-loop
            for( unsigned i=1; i<rows-1; i++) //y-loop
            for( unsigned k=0; k<n; k++)
            for( unsigned l=0; l<n; l++)
            for( unsigned q=0; q<n; q++) //multiplication-loop
            {
                y.data()[(((s*rows+i)*n+k)*cols + row_[m]+1)*n +l] = 0;
            }
        }
        for( unsigned m=0; m<data_.size(); m++) //all blocks
        {
            for( unsigned s=0; s<x.Nz(); s++) //z-loop
            for( unsigned i=1; i<rows-1; i++) //y-loop
            for( unsigned k=0; k<n; k++)
            for( unsigned l=0; l<n; l++)
            for( unsigned q=0; q<n; q++) //multiplication-loop
            {
                y.data()[(((s*rows+i)*n+k)*cols + row_[m]+1)*n +l] += 
                    data_[m][l*n+q]
                    *x.data()[(((s*rows+i)*n+k)*cols + col_[m]+1)*n + q ];
            }
        }
    }

    /**
     * @brief Apply the boundary terms in Y
     *
     * @param x
     * @param y
     */
    void applyY( const MPI_Vector& x, MPI_Vector& y) const
    {
        if(data_.empty()) return;
        unsigned rows = x.Ny(), cols = x.Nx(), n = x.n();
        for( unsigned m=0; m<data_.size(); m++) //all blocks
        {
            for( unsigned s=0; s<x.Nz(); s++)//z-loop
            for( unsigned k=0; k<n; k++)
            for( unsigned j=1; j<cols-1; j++) //x-loop
            for( unsigned l=0; l<n; l++)
            for( unsigned p=0; p<n; p++)
            {
                y.data()[(((s*rows+row_[m]+1)*n+k)*cols + j)*n +l] = 0;
            }
        }
        for( unsigned m=0; m<data_.size(); m++) //all blocks
        {
            for( unsigned s=0; s<x.Nz(); s++)//z-loop
            for( unsigned k=0; k<n; k++)
            for( unsigned j=1; j<cols-1; j++) //x-loop
            for( unsigned l=0; l<n; l++)
            for( unsigned p=0; p<n; p++)
            {
                y.data()[(((s*rows+row_[m]+1)*n+k)*cols + j)*n +l] += 
                     data_[m][k*n+p]
                    *x.data()[(((s*rows+col_[m]+1)*n+p)*cols + j)*n + l];
            }
        }
    }
};

/**
 * @brief Matrix class for block matrices for 2D and 3D derivatives in X and Y direction
 *
 * Stores only one line of blocks and takes care of updating
 * ghost cells before being applied to vectors. This has a huge advantage in memory consumption.
 */
struct MPI_Matrix
{
    /**
     * @brief Construct from boundary condition and number of blocks
     *
     * @param bcx boundary condition in x
     * @param comm Communicator
     * @param number # of blocks
     */
    MPI_Matrix( bc bcx, MPI_Comm comm, unsigned number): 
        dataY_(number), dataX_(number), offset_(number, 0), 
        bcx_( bcx), bcy_( dg::PER), comm_(comm){ }
    /**
     * @brief Construct from boundary condition and number of blocks
     *
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param comm Communicator
     * @param number # of blocks
     */
    MPI_Matrix( bc bcx, bc bcy, MPI_Comm comm, unsigned number): 
        dataY_(number), dataX_(number), offset_(number, 0), 
        bcx_( bcx), bcy_( bcy), comm_(comm){ }
    /**
     * @brief Set Boundary condition in x
     *
     * @return 
     */
    bc& bcx(){return bcx_;}
    /**
     * @brief Set Boundary condition in y
     *
     * @return 
     */
    bc& bcy(){return bcy_;}
    /**
     * @brief Get Boundary condition in x 
     *
     * @return 
     */
    const bc& bcx()const{return bcx_;}
    /**
     * @brief Get Boundary condition in y
     *
     * @return 
     */
    const bc& bcy()const{return bcy_;}

    /**
     * @brief Get Communicator
     *
     * @return 
     */
    MPI_Comm communicator()const{return comm_;}


    /**
     * @brief Set blocks in Y
     *
     * @return 
     */
    std::vector<std::vector<double> >& dataY()    {return dataY_;}
    /**
     * @brief Set blocks in X
     *
     * @return 
     */
    std::vector<std::vector<double> >& dataX()    {return dataX_;}
    /**
     * @brief Set offset
     *
     * @return 
     */
    std::vector<int>&                  offset()  {return offset_;}
    /**
     * @brief Set boundary terms in x
     *
     * @return 
     */
    BoundaryTerms& xterm() {return xterm_;}
    /**
     * @brief Set boundary terms in y
     *
     * @return 
     */
    BoundaryTerms& yterm() {return yterm_;}
    /**
     * @brief Set the normalisation 
     *
     * @return 
     */
    MPI_Precon& precond() {return p_;}

    /**
     * @brief Apply the matrix to a vector
     *
     * @param x
     * @param y
     */
    void symv( MPI_Vector& x, MPI_Vector& y) const;
  private:
    MPI_Precon p_;
    std::vector<std::vector<double> > dataY_;
    std::vector<std::vector<double> > dataX_;
    std::vector<int> offset_;
    BoundaryTerms xterm_;
    BoundaryTerms yterm_;
    bc bcx_, bcy_;
    MPI_Comm comm_;
};


typedef MPI_Matrix MMatrix; //!< mpi matrix type
void MPI_Matrix::symv( MPI_Vector& x, MPI_Vector& y) const
{
    int rank;
    MPI_Comm_rank(comm_, &rank);
    bool updateX = false, updateY = false;
    for( unsigned k=0; k<dataX_.size(); k++)
    {
        if( !dataY_[k].empty() )
            updateY = true;
        if( !dataX_[k].empty() )
            updateX = true;
    }
    if( updateX )
        x.x_col(comm_); 
    if( updateY) 
        x.x_row(comm_);
#ifdef DG_DEBUG
    assert( x.data().size() == y.data().size() );
#endif //DG_DEBUG
    unsigned rows = x.Ny(), cols = x.Nx(), n = x.n();
    for( unsigned i=0; i<y.data().size(); i++)
        y.data()[i] = 0;
    for( unsigned m=0; m<dataX_.size(); m++) //all blocks
    {
        if( !dataX_[m].empty())
            for( unsigned s=0; s<x.Nz(); s++) //z-loop
            for( unsigned i=1; i<rows-1; i++) //y-loop
            for( unsigned k=0; k<n; k++)
            for( unsigned j=1; j<cols-1; j++) //x-loop
            for( unsigned l=0; l<n; l++)
            for( unsigned q=0; q<n; q++) //multiplication-loop
            {
                y.data()[(((s*rows+i)*n+k)*cols + j)*n +l] += 
                    dataX_[m][l*n+q]
                    *x.data()[(((s*rows+i)*n+k)*cols + j)*n + q + offset_[m]];
            }
        if( !dataY_[m].empty())
            for( unsigned s=0; s<x.Nz(); s++)
            for( unsigned i=1; i<rows-1; i++)
            for( unsigned k=0; k<n; k++)
            for( unsigned j=1; j<cols-1; j++)
            for( unsigned l=0; l<n; l++)
            for( unsigned p=0; p<n; p++)
            {
                y.data()[(((s*rows+i)*n+k)*cols + j)*n +l] += 
                     dataY_[m][k*n+p]
                    *x.data()[(((s*rows+i)*n+p)*cols + j)*n + l + offset_[m]];
            }
    }
    xterm_.applyX( x,y);
    yterm_.applyY( x,y);
    if( !p_.data.empty())
        dg::blas2::detail::doSymv( p_, y, y, MPIPreconTag(), MPIVectorTag(), MPIVectorTag());

}

///@}

///@cond
template <>
struct MatrixTraits<MPI_Matrix>
{
    typedef double value_type;
    typedef MPIMatrixTag matrix_category;
};
template <>
struct MatrixTraits<const MPI_Matrix>
{
    typedef double value_type;
    typedef MPIMatrixTag matrix_category;
};

///@endcond

} //namespace dg
