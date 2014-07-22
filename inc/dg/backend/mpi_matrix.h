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
    bc bcx()const{return bcx_;}
    bc bcy()const{return bcy_;}
    

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
                        low_sign*v.data()[i*n*n+(n-k-1)*n+l];
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
    if( updateX )
        update_boundaryX( x);
    if( updateY) 
        update_boundaryY( x);
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

namespace create
{

MPI_Matrix dx( const MPI_Grid2d& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    unsigned n = g.n();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> normx(g.n(), 0.), normy(g.n(), 0.);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./g.hx();
    //create norm and weights
    if( no == not_normed)
    {
        for( unsigned i=0; i<g.n(); i++)
            normx( i,i) = normy( i,i) = g.dlt().weights()[i];
        normx *= g.hx()/2.;
        normy *= g.hy()/2.; // normalisation because F is invariant
    }
    else
    {
        normx = normy = create::delta(g.n());
    }
    std::vector<double> weigh(g.n());
    for( unsigned i=0; i<weigh.size(); i++)
        weigh[i] = normy(i,i);

    Operator<double> a(n), b(n), bt(n);
    if( dir == dg::symmetric)
    {
        //for( unsigned i=0; i<weigh.size(); i++)
            //std::cout << weigh[i] << " \n";

        MPI_Matrix m(bcx, g.bcy(), g.communicator(),  3);
        m.offset()[0] = -1, m.offset()[1] = 0, m.offset()[2] = 1;
        m.state()[0] = +1, m.state()[1] = +1, m.state()[2] = +1;
        m.weights()[0] = weigh, m.weights()[1] = weigh, m.weights()[2] = weigh;

        bt = normx*backward*t*(-0.5*lr )*forward; 
        a  = normx*backward*t*( 0.5*(d-d.transpose()) )*forward;
        b  = normx*backward*t*( 0.5*rl )*forward;

        m.data()[0] = bt.data(), m.data()[1] = a.data(), m.data()[2] = b.data();
        
        return m;
    }
    if( dir == dg::forward)
    {
        MPI_Matrix m(bcx, g.bcy(), g.communicator(),  2);
        m.offset()[0] = 0, m.offset()[1] = 1;
        m.state()[0] = +1, m.state()[1] = +1;
        m.weights()[0] = weigh, m.weights()[1] = weigh;

        a = normx*backward*t*(-d.transpose()-l)*forward; 
        b = normx*backward*t*(rl)*forward;
        m.data()[0] = a.data(), m.data()[1] = b.data();
        return m;
    }
    //if dir == dg::backward
    MPI_Matrix m(bcx, g.bcy(), g.communicator(),  2);
    m.offset()[0] = -1, m.offset()[1] = 0;
    m.state()[0] = +1, m.state()[1] = +1;
    m.weights()[0] = weigh, m.weights()[1] = weigh;
    bt = normx*backward*t*(-lr)*forward; 
    a  = normx*backward*t*(d+l)*forward;
    m.data()[0] = bt.data(), m.data()[1] = a.data();
    return m;
}
MPI_Matrix dx( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return dx( g, g.bcx(), no, dir);
}
MPI_Matrix dy( const MPI_Grid2d& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    MPI_Grid2d swapped_g( g.global().y0(), g.global().y1(), g.global().x0(), g.global().x1(), g.global().n(), g.global().Ny(), g.global().Nx(), g.global().bcy(), g.global().bcx(), g.communicator());
    MPI_Matrix m = dx( swapped_g, bcy, no, dir );
    for( unsigned i=0; i<m.state().size(); i++)
        m.state()[i] = -1;
    return m;
}
MPI_Matrix dy( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return dy( g, g.bcy(), no, dir);
}
MPI_Matrix dxx( const MPI_Grid2d& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    unsigned n = g.n();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> normx(g.n(), 0.), normy(g.n(), 0.);
    //create norm and weights
    if( no == not_normed)
    {
        for( unsigned i=0; i<g.n(); i++)
            normx( i,i) = normy( i,i) = g.dlt().weights()[i];
        normx *= g.hx()/2.;
        normy *= g.hy()/2.; // normalisation because F is invariant
    }
    else
    {
        normx = normy = create::delta(g.n());
    }
    std::vector<double> weigh(normx.size());
    for( unsigned i=0; i<weigh.size(); i++)
        weigh[i] = normy(i,i);

    Operator<double> a(n), b(n), bt(n);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./g.hx();

    a = (lr*t*rl + (d+l)*t*(d+l).transpose() + (l + r));
    b = -(d+l)*t*rl-rl;
    if( dir == dg::backward)
    {
        a = (rl*t*lr + (d+l).transpose()*t*(d+l) + (l + r));
        b = (-rl*t*(d+l) - rl);
    }
    bt = b.transpose();
    a = normx*backward*t*a*forward, bt = normx*backward*t*bt*forward, b = normx*backward*t*b*forward;

    MPI_Matrix m(bcx, g.bcy(), g.communicator(),  3);
    m.offset()[0] = -1, m.offset()[1] = 0, m.offset()[2] = 1;
    m.state()[0]  = +1, m.state()[1]  = +1, m.state()[2] = +1;
    m.weights()[0] = weigh, m.weights()[1] = weigh, m.weights()[2] = weigh;

    m.data()[0] = bt.data(), m.data()[1] = a.data(), m.data()[2] = b.data();
    return m;
}

MPI_Matrix laplacianM( const MPI_Grid2d& g, bc bcx, bc bcy, norm no = normed, direction dir = symmetric)
{
    MPI_Matrix lapx = dxx( g, bcx, no, dir );
    MPI_Grid2d swapped_g( g.global().y0(), g.global().y1(), g.global().x0(), g.global().x1(), g.global().n(), g.global().Ny(), g.global().Nx(), g.global().bcy(), g.global().bcx(), g.communicator());
    MPI_Matrix lapy = dxx( swapped_g, bcy, no, dir );
    for( unsigned i=0; i<lapy.state().size(); i++)
        lapy.state()[i] = -1;
    //append vectors
    lapx.bcy() = bcy;
    lapx.data().insert( lapx.data().end(), lapy.data().begin(), lapy.data().end());
    lapx.weights().insert( lapx.weights().end(), lapy.weights().begin(), lapy.weights().end());
    lapx.offset().insert( lapx.offset().end(), lapy.offset().begin(), lapy.offset().end());
    lapx.state().insert( lapx.state().end(), lapy.state().begin(), lapy.state().end());
    return lapx;
}

MPI_Matrix laplacianM( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return laplacianM( g, g.bcx(), g.bcy(), no, dir);
}

MPI_Matrix forward_transform( const MPI_Grid2d& g)
{
    unsigned n = g.n();
    Operator<double> forward = g.dlt().forward();
    std::vector<double> weigh(g.n());
    for( unsigned i=0; i<weigh.size(); i++)
        weigh[i] = 1.;

    MPI_Matrix m(g.bcx(), g.bcy(), g.communicator(),  1);
    m.offset()[0] = 0;
    m.state()[0] = +1;
    m.weights()[0] = weigh;

    m.data()[0] = forward.data();
        
    return m;
}
MPI_Matrix backward_transform( const MPI_Grid2d& g)
{
    unsigned n = g.n();
    Operator<double> op = g.dlt().backward();
    std::vector<double> weigh(g.n());
    for( unsigned i=0; i<weigh.size(); i++)
        weigh[i] = 1.;

    MPI_Matrix m(g.bcx(), g.bcy(), g.communicator(),  1);
    m.offset()[0] = 0;
    m.state()[0] = +1;
    m.weights()[0] = weigh;

    m.data()[0] = op.data();
        
    return m;
}

} //namespace create

} //namespace dg
