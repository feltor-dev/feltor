#pragma once

#include "mpi_config.h"
#include "mpi_vector.h"
#include "operator_dynamic.h"
#include "enums.h"


namespace dg
{

struct MPI_Matrix
{
    //cusp::csr_matrix<int, double, cusp::host_memory> data; //!< CSR host Matrix
    std::vector<double> data[3];
    unsigned state; //0 is 1xM and 1 is Mx1 and 2 is laplace
    MPI_Comm comm;
    void update_boundaryX( MPI_Vector& v)
    {
        x_col(v);
        int low_sign, upp_sign;
        if( bcx == DIR)
            low_sign=upp_sign=-1;
        else if( bcx == NEU)
            low_sign=upp_sign=+1;
        else if( bcx == DIR_NEU)
            low_sign=-1, upp_sign=+1;
        else if( bcx == NEU_DIR)
            low_sign=+1, upp_sign=-1;
        int dims[2], periods[2], coords[2];
        MPI_Cart_get( comm, 2, dims, periods, coords);
        if( coords[0] == dims[0]-1)
            for( unsigned i=1; i<rows-1; i++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned l=0; l<n; l++)
                        v.data[(i*cols + cols-1)*n*n+k*n+l] = 
                            upp_sign*v.data[(i*cols + cols-2)*n*n+k*n+n-l-1];
        else if( coords[0] == 0)
            for( unsigned i=1; i<rows-1; i++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned l=0; l<n; l++)
                        v.data[i*cols*n*n+k*n+l] = 
                            low_sign*v.data[(i*cols+1)*n*n+k*n+n-l-1];
        return;
    }
    void update_boundaryY( MPI_Vector& v)
    {
        x_row(v);
        int low_sign, upp_sign;
        if( bcy == DIR)
            low_sign=upp_sign=-1;
        else if( bcy == NEU)
            low_sign=upp_sign=+1;
        else if( bcy == DIR_NEU)
            low_sign=-1, upp_sign=+1;
        else if( bcy == NEU_DIR)
            low_sign=+1, upp_sign=-1;
        if( coords[1] == dims[1]-1)
            for( unsigned i=1; i<cols-1; i++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned l=0; l<n; l++)
                        v.data[((rows-1)*cols+i)*n*n+k*n+l] = 
                            low_sign*v.data[((rows-2)*cols+i)*n*n+(n-k-1)*n+l];
        else if( coords[1] == 0)
            for( unsigned i=1; i<cols-1; i++)
                for( unsigned k=0; k<n; k++)
                    for( unsigned l=0; l<n; l++)
                        v.data[i*n*n+k*n+l] = upp_sign*v.data[i*n*n+(n-k-1)*n+l];
        return;
    }

    void multiplyAdd( unsigned n, const std::vector<double>& op, const double* x, double* y)
    {
        if( op.empty() )
            return;
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
            { 
                //y[i*n+j] = 0;
                for( unsigned k=0; k<n; k++)
                    y[i*n+j]+= w[i]*op[j*n+k]*x[i*n+k];
            }
    }
    void multiplyAdd( const std::vector<double>& op, unsigned n, const double* x, double* y)
    {
        if( op.empty() )
            return;
        for( unsigned i=0; i<n; i++)
            for( unsigned j=0; j<n; j++)
            { 
                //y[i*n+j] = 0;
                for( unsigned k=0; k<n; k++)
                    y[i*n+j]+= w[j]*op[i*n+k]*x[k*n+j];
            }
    }
  private:
    std::vector<double> w;
};

template <>
struct MatrixTraits<MPI_Matrix>
{
    typedef double value_type;
    typedef MPIMatrixTag matrix_category;
};

namespace create
{
namespace detail
{

MPI_Matrix dx( const Grid2d<double>& g, bc bcx, direction dir)
{
    MPI_Matrix m;
    unsigned n=g.n();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator a(n), b(n), bt(n);

    if( dir == symmetric)
    {
        m.data[0] = -0.5*lr; 
        m.data[1] = 0.5*(d-d.transpose());
        m.data[2] = 0.5*rl;
        return m;
    }
    if( dir == forward)
    {
        m.data[1] = -d.transpose()-l; 
        m.data[2] = rl;
        return m;
    }
    if( dir == backward)
    {
        m.data[0] = -lr, m.data[1] = d+l;
        return m;
    }
}

MPI_Matrix dxx( const MPIGrid2d<double>& g, bc bcx, bc bcy, direction dir)
{
    MPI_Matrix m;
    unsigned n=g.n();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> t = create::pipj_inv(n); 
    t *= 2./g.ny();
    Operator a(n), b(n), bt(n);

    a = lr*t*rl + (d+l)*t*(d+l).transpose + (l + r);
    b = -rl*t*(d+l) - rl;
    if( dir == backward)
    {
        a = rl*t*lr + (d+l).transpose()*t*(d+l) + (l + r);
        b = -(d+l)*t*rl - rl;
    }
    bt = b.transpose();
    m.data[0] = bt, m.data[1] = a, m.data[2] = b;
    return m;

}

void sandwich( const Operator<double>& left, MPI_Matrix& m, const Operator<double>& right)
{
    for( unsigned i=0; i<3; i++)
        if( !m.data[i].empty())
            m.data[i] = left*(Operator<double>)(m.data[i])*right;
}


} //namespace detail

MPI_Matrix dx( const MPIGrid2d<double>& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    MPI_Matrix m = dx( g, bcx, dir);
    unsigned n = g.n();
    double h = g.h
    if( no == not_normed)
    {
        Operator<double> forward = g.dlt().forward();
        sandwich( forward.transpose(), m, forward);
        m.weights = g.dlt().weights();
        for(unsigned i=0; i<n;i++)
            m.weights[i] *= g.hy()/2.; 
    }
    if( no == normed)
    {
        Operator<double> forward = g.dlt().forward();
        Operator<double> backward = g.dlt().backward();
        sandwich( backward, m, forward);
        m.weights.resize( n);
        for(unsigned i=0; i<n;i++)
            m.weights[i] = 1.; 
        
    }

}

} //namespace create

} //namespace dg
