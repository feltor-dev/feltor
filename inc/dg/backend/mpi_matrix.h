#pragma once

#include <vector>

//#include "derivatives.cuh"
#include "mpi_grid.h"
#include "mpi_config.h"
#include "mpi_vector.h"
#include "mpi_precon_blas.h"
#include "operator.h"


namespace dg
{

/**
 * @brief Matrix class for block matrices 
 *
 * Stores only one line of blocks and takes care of updating
 * ghost cells before being applied to vectors.
 */
struct MPI_Matrix
{
    MPI_Matrix( bc bcx, bc bcy, MPI_Comm comm, unsigned number): 
        dataY_(number), dataX_(number), offset_(number), 
        bcx_( bcx), bcy_( bcy), comm_(comm){ }
    bc& bcx(){return bcx_;}
    bc& bcy(){return bcy_;}
    const bc& bcx()const{return bcx_;}
    const bc& bcy()const{return bcy_;}

    MPI_Comm communicator()const{return comm_;}

    void update_boundaryX( MPI_Vector& v) const;
    void update_boundaryY( MPI_Vector& v) const;

    std::vector<std::vector<double> >& dataY()    {return dataY_;}
    std::vector<std::vector<double> >& dataX()    {return dataX_;}
    std::vector<int>&                  offset()  {return offset_;}
    MPI_Precon& precond() {return p_;}
    const std::vector<std::vector<double> >& dataY()const {return dataY_;}
    const std::vector<std::vector<double> >& dataX()const {return dataX_;}
    const std::vector<int>& offset()const {return offset_;}
    const MPI_Precon precond()const {return p_;}

    void symv( MPI_Vector& x, MPI_Vector& y) const;
  private:
    MPI_Precon p_;
    std::vector<std::vector<double> > dataY_;
    std::vector<std::vector<double> > dataX_;
    std::vector<int> offset_;
    bc bcx_, bcy_;
    MPI_Comm comm_;
};

void MPI_Matrix::symv( MPI_Vector& x, MPI_Vector& y) const
{
    bool updateX = false, updateY = false;
    for( unsigned k=0; k<dataX_.size(); k++)
    {
        if( !dataY_[k].empty() )
            updateY = true;
        if( !dataX_[k].empty() )
            updateX = true;
    }
    if( updateX )
        update_boundaryX( x);
    if( updateY) 
        update_boundaryY( x);
#ifdef DG_DEBUG
    assert( x.data().size() == y.data().size() );
#endif //DG_DEBUG
    unsigned rows = x.Ny(), cols = x.Nx(), n = x.n();
    for( unsigned i=0; i<y.data().size(); i++)
        y.data()[i] = 0;
    for( unsigned m=0; m<dataX_.size(); m++)
    {
        if( !dataX_[m].empty())
            for( unsigned i=1; i<rows-1; i++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned j=1; j<cols-1; j++)
                        for( unsigned l=0; l<n; l++)
                            for( unsigned q=0; q<n; q++)
                            {
                                y.data()[((i*n+k)*cols + j)*n +l] += 
                                dataX_[m][l*n+q]
                                *x.data()[((i*n+k)*cols + j)*n + q + offset_[m]];
                            }
        if( !dataY_[m].empty())
            for( unsigned i=1; i<rows-1; i++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned j=1; j<cols-1; j++)
                        for( unsigned l=0; l<n; l++)
                            for( unsigned p=0; p<n; p++)
                            {
                                y.data()[((i*n+k)*cols + j)*n +l] += 
                                 dataY_[m][k*n+p]
                                *x.data()[((i*n+p)*cols + j)*n + l + offset_[m]];
                            }
    }
    if( !p_.data.empty())
        dg::blas2::detail::doSymv( p_, y, y, MPIPreconTag(), MPIVectorTag(), MPIVectorTag());

}

void MPI_Matrix::update_boundaryX( MPI_Vector& v)const
{
    v.x_col(comm_);
    if( bcx_ == PER) return;
    int low_sign, upp_sign;
    if( bcx_ == DIR)
        low_sign=upp_sign=-1;
    else if( bcx_ == NEU)
        low_sign=upp_sign=+1;
    else if( bcx_ == DIR_NEU)
        low_sign=-1, upp_sign=+1;
    else if( bcx_ == NEU_DIR)
        low_sign=+1, upp_sign=-1;
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( comm_, 2, dims, periods, coords);
    unsigned rows = v.Ny(), cols =v.Nx(), n =  v.n();
    if( coords[0] == dims[0]-1)
        for( unsigned i=1; i<rows-1; i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v.data()[(i*cols + cols-1)*n*n+k*n+l] = 
                        upp_sign*v.data()[(i*cols + cols-2)*n*n+k*n+n-l-1];
    else if( coords[0] == 0)
        for( unsigned i=1; i<rows-1; i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v.data()[i*cols*n*n+k*n+l] = 
                        low_sign*v.data()[(i*cols+1)*n*n+k*n+n-l-1];
    return;
}
void MPI_Matrix::update_boundaryY( MPI_Vector& v)const
{
    v.x_row(comm_);
    if( bcy_ == PER) return;
    int low_sign, upp_sign;
    if( bcy_ == DIR)
        low_sign=upp_sign=-1;
    else if( bcy_ == NEU)
        low_sign=upp_sign=+1;
    else if( bcy_ == DIR_NEU)
        low_sign=-1, upp_sign=+1;
    else if( bcy_ == NEU_DIR)
        low_sign=+1, upp_sign=-1;
    int dims[2], periods[2], coords[2];
    MPI_Cart_get( comm_, 2, dims, periods, coords);
    unsigned rows = v.Ny(), cols =v.Nx(), n = v.n();
    if( coords[1] == dims[1]-1)
        for( unsigned i=1; i<cols-1; i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v.data()[((rows-1)*cols+i)*n*n+k*n+l] = 
                        upp_sign*v.data()[((rows-2)*cols+i)*n*n+(n-k-1)*n+l];
    else if( coords[1] == 0)
        for( unsigned i=1; i<cols-1; i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    v.data()[i*n*n+k*n+l] = 
                        low_sign*v.data()[(cols+i)*n*n+(n-k-1)*n+l];
    return;
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


} //namespace dg
