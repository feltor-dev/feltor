#pragma once

#include <vector>

//#include "derivatives.cuh"
#include "mpi_grid.h"
#include "mpi_config.h"
#include "mpi_vector.h"
#include "operator.h"


namespace dg
{

struct MPI_Matrix
{
    MPI_Matrix( bc bcx, bc bcy, MPI_Comm comm, unsigned number ): 
        dataY_(number), dataX_(number), offset_(number), 
        bcx_( bcx), bcy_( bcy), comm_(comm){}
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
    const std::vector<std::vector<double> >& dataY()const {return dataY_;}
    const std::vector<std::vector<double> >& dataX()const {return dataX_;}
    const std::vector<int>& offset()const {return offset_;}

    void multiplyAdd( unsigned n, const double* op, const double* w, const double* x, double* y) const;
    void multiplyAdd( const double* op, unsigned n, const double* w, const double* x, double* y) const;
    void multiplyAdd(unsigned n, const std::vector<double>& op1, const std::vector<double>& op2, const double* x, double* y) const;
    void symv( MPI_Vector& x, MPI_Vector& y) const;
    //cusp::csr_matrix<int, double, cusp::host_memory>& cusp_matrix(){ return cmatrix_;}
  private:
    //cusp::csr_matrix<int, double, cusp::host_memory> cmatrix_; //!< CSR host Matrix
    std::vector<std::vector<double> > dataY_;
    std::vector<std::vector<double> > dataX_;
    std::vector<int> offset_;
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

//inline void MPI_Matrix::multiplyAdd( unsigned n, const std::vector<double>& op, const std::vector<double>& w, const double* x, double* y) const
//{
//    for( unsigned i=0; i<n; i++)
//        for( unsigned j=0; j<n; j++)
//            for( unsigned k=0; k<n; k++)
//                y[i*n+j]+= w[i]*op[j*n+k]*x[i*n+k];
//}
//
//inline void MPI_Matrix::multiplyAdd( const std::vector<double>& op, unsigned n, const std::vector<double>& w,const double* x, double* y) const
//{
//    for( unsigned i=0; i<n; i++)
//        for( unsigned j=0; j<n; j++)
//            for( unsigned k=0; k<n; k++)
//                y[i*n+j]+= w[j]*op[i*n+k]*x[k*n+j];
//}
inline void MPI_Matrix::multiplyAdd( unsigned n, const double* op, const double* w, const double* x, double* y) const
{
    unsigned l, m;
    for( unsigned i=0; i<n; i++)
    {
        l=i*n;
        for( unsigned j=0; j<n; j++)
        {
            m = j*n;
            for( unsigned k=0; k<n; k++)
                y[l+j]+= w[i]*op[m+k]*x[l+k];
        }
    }
}

inline void MPI_Matrix::multiplyAdd( const double* op, unsigned n, const double* w,const double* x, double* y) const
{
    unsigned l;
    for( unsigned i=0; i<n; i++)
    {
        l=i*n;
        for( unsigned j=0; j<n; j++)
        {
            for( unsigned k=0; k<n; k++)
                y[l+j]+= w[j]*op[l+k]*x[k*n+j];
        }
    }
}
inline void MPI_Matrix::multiplyAdd( unsigned n, const std::vector<double>& op1, const std::vector<double>& op2, const double* x, double* y) const
{
    //unsigned l, m;
    for( unsigned i=0; i<n; i++)
    {
        for( unsigned j=0; j<n; j++)
        {
            for( unsigned k=0; k<n; k++)
                for( unsigned l=0; l<n; l++)
                    y[i*n+j]+= op1[i*n+k]*op2[j*n+l]*x[k*n+l];
        }
    }
}
/*
void MPI_Matrix::symv( MPI_Vector& x, MPI_Vector& y) const
{
    //dg::Timer t;
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
    //t.tic();
    if( updateX )
        update_boundaryX( x);
    if( updateY) 
        update_boundaryY( x);
    //t.toc();
    //std::cout << "boundary update took "<<t.diff()<<"s\n";
    //if(rank==0) std::cout<<x<<std::endl;
    //MPI_Barrier(comm_);
    //if(rank==1) std::cout<<x<<std::endl;
    //MPI_Barrier(comm_);
    //t.tic();
    for( unsigned i=1; i<rows-1; i++)
        for( unsigned j=1; j<cols-1; j++)
        {
            for( unsigned k=0; k<x.stride(); k++)
                y.data()[(i*cols+j)*x.stride() +k] = 0;
            for( unsigned k=0; k<data_.size(); k++)
            {
                if( state_[k]>0)
                    multiplyAdd( n, (data_[k]).data(), (w_[k]).data(), &x.data()[(i*cols+j+offset_[k])*n*n], &y.data()[(i*cols+j)*n*n]);
                else
                    multiplyAdd( (data_[k]).data(), n, (w_[k]).data(), &x.data()[((i+offset_[k])*cols+j)*n*n], &y.data()[(i*cols+j)*n*n]);
            }
        }
    //dg::blas2::detail::doSymv( cmatrix_, x.data(), y.data(), CuspMatrixTag(), ThrustVectorTag(), ThrustVectorTag());
    

    unsigned num, stride=n*n;
    for( unsigned k=0; k<x.size(); k++)
        y.data()[k] = 0; //x.data()[k]*w_[0][k%3];
    for( unsigned k=0; k<data_.size(); k++)
    {
        if( state_[k]>0)
            for( unsigned i=1; i<rows-1; i++)
            {
                num = i*cols;
                for( unsigned j=1; j<cols-1; j++)
                    //multiplyAdd( n, (data_[k]).data(), (w_[k]).data(), &x.data()[(num+j+offset_[k])*stride], &y.data()[(num+j)*stride]);
                    multiplyAdd(n, (data_[k]).data(), (data_[k]).data(), &x.data()[(num+j+offset_[k])*stride], &y.data()[(num+j)*stride]);
            }
        else
            for( unsigned i=1; i<rows-1; i++)
            {
                num = i*cols;
                for( unsigned j=1; j<cols-1; j++)
                    //multiplyAdd( (data_[k]).data(), n, (w_[k]).data(), &x.data()[((i+offset_[k])*cols+j)*stride], &y.data()[(num+j)*stride]);
                    multiplyAdd(n, (data_[k]).data(), (data_[k]).data(), &x.data()[((i+offset_[k])*cols+j)*stride], &y.data()[(num+j)*stride]);
            }
    }
    //t.toc();
    //std::cout << "Multiplication  took "<<t.diff()<<"s\n";
    //if(rank==0) std::cout<<y<<std::endl;


}
*/
void MPI_Matrix::symv( MPI_Vector& x, MPI_Vector& y) const
{
    //bool updateX = false, updateY = false;
    //for( unsigned k=0; k<state_.size(); k++)
    //{
    //    if( state_[k] < 0 )
    //        updateY = true;
    //    else
    //        updateX = true;
    //}
    //if( updateX )
        update_boundaryX( x);
    //if( updateY) 
        update_boundaryY( x);
#ifdef DG_DEBUG
    assert( x.data().size() == y.data().size() );
#endif //DG_DEBUG
    //std::cout << "ping0\n";
    /*
    unsigned rows = x.Ny(), cols = x.Nx(), n = x.n();
    for( unsigned i=1; i<rows-1; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=1; j<cols-1; j++)
                for( unsigned l=0; l<n; l++)
                {
                    y.data()[((i*n+k)*cols + j)*n +l] = 0;
                    for( unsigned m=0; m<dataY_.size(); m++)
                        for( unsigned p=0; p<n; p++)
                            for( unsigned q=0; q<n; q++)
                            {
                                y.data()[((i*n+k)*cols + j)*n +l] += 
                                 dataY_[m][k*n+p]
                                *dataX_[m][l*n+q]
                                *x.data()[((i*n+p)*cols + j)*n + q + offset_[m]];
                            }
                }
    */
    unsigned rows = x.Ny(), cols = x.Nx(), n = x.n();
    for( unsigned i=0; i<y.data().size(); i++)
        y.data()[i] = 0;
    dg::MPI_Vector temp(y);
    for( unsigned m=0; m<dataX_.size(); m++)
    {
    if( !dataX_[m].empty())
    for( unsigned i=1; i<rows-1; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=1; j<cols-1; j++)
                for( unsigned l=0; l<n; l++)
                {
                    temp.data()[((i*n+k)*cols + j)*n +l] = 0;
                        for( unsigned q=0; q<n; q++)
                        {
                            temp.data()[((i*n+k)*cols + j)*n +l] += 
                            dataX_[m][l*n+q]
                            *x.data()[((i*n+k)*cols + j)*n + q + offset_[m]];
                        }
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
                            *temp.data()[((i*n+p)*cols + j)*n + l];
                        }
    else
    for( unsigned i=1; i<rows-1; i++)
        for( unsigned k=0; k<n; k++)
            for( unsigned j=1; j<cols-1; j++)
                for( unsigned l=0; l<n; l++)
                    {
                        y.data()[((i*n+k)*cols + j)*n +l] += 
                            temp.data()[((i*n+k)*cols + j)*n + l];
                    }

                        //for( unsigned p=0; p<n; p++)
    }

    //std::cout << "ping1\n";

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
