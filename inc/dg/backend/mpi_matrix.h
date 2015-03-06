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
 * @brief Struct holding the boundary blocks either for the x- or y-direction
 *
 * Think of this as a block sparse matrix for just one dimension holding only boundary terms
 * which can be applied either to the x- or y-direction, i.e. it is of the form
 * \f[
 \begin{pmatrix}
 B_0 & B_1 & ... &  \\
 0 & 0 & ... &  \\
 & & ... & \\
 &... & & B_{n-1}
 \end{pmatrix}
 \f]
 The corresponding row/col pairs read \f$ (0,0)\f$, \f$ (0,1)\f$, ..., \f$ (N_x-1,N_x-1)\f$.
 For processes, that do not hold any boundary terms the matrix can be left entirely empty
 This format is sufficient for simple dG derivatives in x and y.
 @note not all processes need boundary terms, only those which have a boundary
 *
 */
struct BoundaryTerms
{
    /**
     * @brief blocks of data \f$ B_i \f$
     *
     * each element is a nxn block of data
     */
    std::vector<std::vector<double> > data_;
    /**
     * @brief 1D cell numbers of corresponding data_ block
     *
     * Has to be of equal size as data_.
     */
    std::vector<int> row_; 
    /**
     * @brief 1D cell numbers of corresponding data_ block
     *
     * Has to be of equal size as data_.
     */
    std::vector<int> col_; 
    /**
     * @brief Apply the boundary terms in X
     *
     * If data_ is empty the function will return immediately
     * @param x input vector
     * @param y output vector
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
     * If data_ is empty the function will return immediately
     * @param x input vector
     * @param y output vector
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
 * ghost cells before being applied to vectors. 
 * This has a huge advantage in memory consumption over other sparse matrix formats.
 * The format is that of a sparse block matrix with equal blocks in each line
 \f[
 \begin{pmatrix}
 A & B & C &  & & \\
   & A & B & C & & \\
   ...\\
   &   &   & A & B & C
   \end{pmatrix}
   \f]
 */
struct MPI_Matrix
{
    /**
     * @brief Construct from boundary condition and number of blocks
     *
     * @param bcx boundary condition in x
     * @param number # of blocks
     */
    MPI_Matrix( unsigned number): 
        dataY_(number), dataX_(number), offset_(number, 0)
       { }

    /**
     * @brief Set blocks in Y
     *
     * May be left empty.
     * If set, these will be applied in the symv() function
     * @return a reference to the internal data
     */
    std::vector<std::vector<double> >& dataY()    {return dataY_;}
    /**
     * @brief Set blocks in X
     *
     * May be left empty.
     * If set, these will be applied in the symv() function
     * @return a reference to the internal data
     */
    std::vector<std::vector<double> >& dataX()    {return dataX_;}
    /**
     * @brief Set offset
     *
     * Has to be of equal size than either dataY_ or dataX_
     * The offsets are the col numbers counting from the diagonal 
     * e.g. for a simple centered X-derivative they are -n, 0, n
     * while for the Y-derivative they are -n*n*Nx, 0, n*n*Nx
     * @return a reference to the internal data
     */
    std::vector<int>&                  offset()  {return offset_;}
    /**
     * @brief Set boundary terms in x
     *
     * May be left empty.
     * If set, these will be applied in the symv() function
     * @return a reference to the internal data
     */
    BoundaryTerms& xterm() {return xterm_;}
    /**
     * @brief Set boundary terms in y
     *
     * May be left empty.
     * If set, these will be applied in the symv() function
     * @return a reference to the internal data
     */
    BoundaryTerms& yterm() {return yterm_;}
    /**
     * @brief Set the normalisation 
     *
     * May be left empty.
     * If set, these weights will be applied in the symv() function
     * @return a reference to the internal data
     */
    MPI_Precon& precond() {return p_;}

    /**
     * @brief Apply the matrix to a vector
     *
     * @param x input vector, const except for the boundary terms
     * @param y output vector
     */
    void symv( MPI_Vector& x, MPI_Vector& y) const;
  private:
    MPI_Precon p_;
    std::vector<std::vector<double> > dataY_;
    std::vector<std::vector<double> > dataX_;
    std::vector<int> offset_;
    BoundaryTerms xterm_;
    BoundaryTerms yterm_;
};


typedef MPI_Matrix MMatrix; //!< mpi matrix type
///@}

///@cond
void MPI_Matrix::symv( MPI_Vector& x, MPI_Vector& y) const
{
#ifdef DG_DEBUG
    assert( x.data().size() == y.data().size() );
    assert( x.n() == y.n());
    assert( x.Nx() == y.Nx());
    assert( x.Ny() == y.Ny());
    assert( x.Nz() == y.Nz());
    assert( x.communicator() == y.communicator());
#endif //DG_DEBUG
    // update boundary layer if necessary
    int rank;
    MPI_Comm_rank(x.communicator(), &rank);
    bool updateX = false, updateY = false;
    for( unsigned k=0; k<dataX_.size(); k++)
    {
        if( !dataY_[k].empty() )
            updateY = true;
        if( !dataX_[k].empty() )
            updateX = true;
    }
    if( updateX )
        x.x_col(); 
    if( updateY) 
        x.x_row();
    // Apply internal data
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
    //apply boundary terms
    xterm_.applyX( x,y);
    yterm_.applyY( x,y);
    if( !p_.data.empty())
        dg::blas2::detail::doSymv( p_, y, y, MPIPreconTag(), MPIVectorTag(), MPIVectorTag());

}


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
