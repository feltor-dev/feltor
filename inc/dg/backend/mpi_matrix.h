#pragma once
#include <vector>
#include "mpi_grid.h"
#include "mpi_config.h"
#include "mpi_vector.h"
#include "operator.h"


namespace dg
{

struct MPI_Matrix
{
    MPI_Matrix( bc bcx, bc bcy, MPI_Comm comm, unsigned number ): 
        data_(number), w_(number), offset_(number), state_(number),
        bcx_( bcx), bcy_( bcy), comm_(comm){}
    bc& bcx(){return bcx_;}
    bc& bcy(){return bcy_;}
    const bc& bcx()const{return bcx_;}
    const bc& bcy()const{return bcy_;}
    

    MPI_Comm communicator()const{return comm_;}



    void update_boundaryX( MPI_Vector& v) const;
    void update_boundaryY( MPI_Vector& v) const;

    std::vector<std::vector<double> >& data()    {return data_;}
    std::vector<std::vector<double> >& weights() {return w_;}
    std::vector<int>&                  offset()  {return offset_;}
    std::vector<int>&                  state()   {return state_;}
    const std::vector<std::vector<double> >& data()const {return data_;}
    const std::vector<std::vector<double> >& weights() const{return w_;}
    const std::vector<int>& offset()const {return offset_;}
    const std::vector<int>& state() const {return state_;}

    void multiplyAdd( unsigned n, const std::vector<double>& op, const std::vector<double>& w, const double* x, double* y) const;
    void multiplyAdd( const std::vector<double>& op, unsigned n, const std::vector<double>& w, const double* x, double* y) const;
    void symv( MPI_Vector& x, MPI_Vector& y) const;

  private:
    std::vector<std::vector<double> > data_;
    std::vector<std::vector<double> > w_;
    std::vector<int> offset_;
    std::vector<int> state_;
    bc bcx_, bcy_;
    MPI_Comm comm_;
};

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
    unsigned rows = v.Ny(), cols =v.Nx(), n = sqrt( v.stride());
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
    unsigned rows = v.Ny(), cols =v.Nx(), n = sqrt( v.stride());
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

void MPI_Matrix::multiplyAdd( unsigned n, const std::vector<double>& op, const std::vector<double>& w, const double* x, double* y) const
{
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<n; k++)
                y[i*n+j]+= w[i]*op[j*n+k]*x[i*n+k];
}

void MPI_Matrix::multiplyAdd( const std::vector<double>& op, unsigned n, const std::vector<double>& w,const double* x, double* y) const
{
    for( unsigned i=0; i<n; i++)
        for( unsigned j=0; j<n; j++)
            for( unsigned k=0; k<n; k++)
                y[i*n+j]+= w[j]*op[i*n+k]*x[k*n+j];
}
void MPI_Matrix::symv( MPI_Vector& x, MPI_Vector& y) const
{
    bool updateX = false, updateY = false;
    for( unsigned k=0; k<state_.size(); k++)
    {
        if( state_[k] < 0 )
            updateY = true;
        else
            updateX = true;
    }
#ifdef DG_DEBUG
    assert( x.data().size() == y.data().size() );
    assert( x.stride() == w_.data()->size()*w_.data()->size());
#endif //DG_DEBUG
    unsigned rows = x.Ny(), cols = x.Nx(), n = w_.data()->size();
    //std::cout << "n "<<n<<"\n";
    //for( unsigned i=0; i<w_.size(); i++)
    //{
    //    std::cout <<" states: "<<state_[i]<<"\n";
    //    std::cout <<" offset: "<<offset_[i]<<"\n";
    //}
    //const char* string1 = updateX?"updateX":"NOT X";
    //const char* string2 = updateY?"updateY":"NOT Y";
    //std::cout << string1 <<std::endl;
    //std::cout << string2 <<std::endl;
    //std::cout << "Before boundary update\n";
    //int rank;
    //MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    //if(rank==0) std::cout<<x<<std::endl;
    //MPI_Barrier(comm_);
    //if(rank==1) std::cout<<x<<std::endl;
    //MPI_Barrier(comm_);
    if( updateX )
        update_boundaryX( x);
    if( updateY) 
        update_boundaryY( x);
    //std::cout << "After boundary update\n";
    //if(rank==0) std::cout<<x<<std::endl;
    //MPI_Barrier(comm_);
    //if(rank==1) std::cout<<x<<std::endl;
    //MPI_Barrier(comm_);
    for( unsigned i=1; i<rows-1; i++)
        for( unsigned j=1; j<cols-1; j++)
        {
            for( unsigned k=0; k<x.stride(); k++)
                y.data()[(i*cols+j)*x.stride() +k] = 0;
            for( unsigned k=0; k<data_.size(); k++)
            {
                if( state_[k]>0)
                    multiplyAdd( n, data_[k], w_[k], &x.data()[(i*cols+j+offset_[k])*n*n], &y.data()[(i*cols+j)*n*n]);
                else
                    multiplyAdd( data_[k], n, w_[k], &x.data()[((i+offset_[k])*cols+j)*n*n], &y.data()[(i*cols+j)*n*n]);
            }
        }
    //if(rank==0) std::cout<<y<<std::endl;

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
