#pragma once

#include "mpi_matrix.h"

namespace dg{
namespace create
{

namespace detail
{
//these are to prevent metric coefficients in the normal weight functions
MPI_Precon pure_weights( const MPI_Grid3d& g)
{
    MPI_Precon p;
    p.data = g.dlt().weights();
    p.norm = g.hz()*g.hx()*g.hy()/4.;
    return p;
}
MPI_Precon pure_weights( const MPI_Grid2d& g)
{
    MPI_Precon p;
    p.data = g.dlt().weights();
    p.norm = g.hx()*g.hy()/4.;
    return p;
}
//create a normed 2d X-derivative
MPI_Matrix dx( const Grid1d<double>& g, bc bcx, direction dir, MPI_Comm comm)
{
    unsigned n=g.n();
    double hx = g.h();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> t = create::pipj_inv(n);
    t *= 2./hx;

    Operator<double> a(n), b(n), bt(n);
    if( dir == dg::symmetric)
    {
        MPI_Matrix m(bcx, comm,  3);
        m.offset()[0] = -n, m.offset()[1] = 0, m.offset()[2] = n;
        
        bt = backward*t*(-0.5*lr )*forward; 
        a  = backward*t*( 0.5*(d-d.transpose()) )*forward;
        b  = backward*t*( 0.5*rl )*forward;

        m.dataX()[0] = bt.data(), m.dataX()[1] = a.data(), m.dataX()[2] = b.data();
        return m;
    }
    if( dir == dg::forward)
    {
        MPI_Matrix m(bcx, comm,  2);
        m.offset()[0] = 0, m.offset()[1] = n;

        a = backward*t*(-d.transpose()-l)*forward; 
        b = backward*t*(rl)*forward;
        m.dataX()[0] = a.data(), m.dataX()[1] = b.data();
        return m;
    }
    //if dir == dg::backward
    MPI_Matrix m(bcx, comm,  2);
    m.offset()[0] = -n, m.offset()[1] = 0;
    bt = backward*t*(-lr)*forward; 
    a  = backward*t*(d+l)*forward;
    m.dataX()[0] = bt.data(), m.dataX()[1] = a.data();
    return m;
}

BoundaryTerms boundaryDX( const Grid1d<double>& g, bc bcx, direction dir)
{
    unsigned n=g.n(), N = g.N()-2;
    double hx = g.h();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> t = create::pipj_inv(n);
    t *= 2./hx;
    BoundaryTerms xterm;
    std::vector<int> row_, col_;
    std::vector<std::vector<double> > data;
    Operator<double> data_[4];
    if( bcx != dg::PER)
    {
        if( dir == dg::symmetric)
        {
            row_.resize(4), col_.resize(4), data.resize(4);
            row_[0] = 0, col_[0] = 0; 
            row_[1] = 0, col_[1] = 1;
            row_[2] = N-1, col_[2] = N-1; 
            row_[3] = N-1, col_[3] = N-2;
            data_[1] = 0.5*rl;
            data_[3] = -0.5*lr;
            switch( bcx)
            {
                case( dg::DIR): data_[0] = 0.5*(d-d.transpose()+l); 
                                data_[2] = 0.5*(d-d.transpose()-r); 
                                break;
                case( dg::NEU): data_[0] = 0.5*(d-d.transpose()-l); 
                                data_[2] = 0.5*(d-d.transpose()+r);
                                break;
                case( dg::DIR_NEU): data_[0] = 0.5*(d-d.transpose()+l); 
                                    data_[2] = 0.5*(d-d.transpose()+r);
                                    break;
                case( dg::NEU_DIR): data_[0] = 0.5*(d-d.transpose()-l); 
                                    data_[2] = 0.5*(d-d.transpose()-r);
                                    break;
            }
        }
        else if( dir == dg::forward)
        {
            row_.resize(3), col_.resize(3), data.resize(3);
            row_[0] = col_[0] = 0, row_[1] = 0, col_[1] = 1;
            row_[2] = col_[2] = N-1;
            data_[1] = rl;
            switch( bcx)
            {
                case( dg::DIR): data_[0] = -d.transpose(); 
                                data_[2] = -(d+l).transpose(); 
                                break;
                case( dg::NEU): data_[0] = -(d+l).transpose(); 
                                data_[2] = d;
                                break;
                case( dg::DIR_NEU): data_[0] = -d.transpose();
                                    data_[2] = d;
                                    break;
                case( dg::NEU_DIR): data_[0] = -(d+l).transpose();
                                    data_[2] = -(d+l).transpose();
                                    break;
            }
        }
        else //dir == dg::backward
        {
            row_.resize(3), col_.resize(3), data.resize(3);
            row_[0] = col_[0] = 0;
            row_[2] = col_[2] = N-1, row_[1] = N-1, col_[1] = N-2;
            data_[1] = -lr;
            switch( bcx)
            {
                case( dg::DIR): data_[2] = -d.transpose(); 
                                data_[0] = (d+l); 
                                break;
                case( dg::NEU): data_[2] = (d+l); 
                                data_[0] = d;
                                break;
                case( dg::DIR_NEU): data_[2] = (d+l);
                                    data_[0] = (d+l);
                                    break;
                case( dg::NEU_DIR): data_[2] = -d.transpose();
                                    data_[0] = d;
                                    break;
            }
        }
        for( unsigned i=0; i<row_.size(); i++)
        {
            data_[i] = backward*t*data_[i]*forward;
            data[i] = data_[i].data();
        }
        xterm.row_ = row_, xterm.col_ = col_, xterm.data_ = data;
    }
    return xterm;
}

} //namespace detail

MPI_Matrix dx( const MPI_Grid2d& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix dx = detail::dx( g1d, bcx, dir, g.communicator() );
    if( no == not_normed) dx.precond() = detail::pure_weights(g);
    dx.xterm() = detail::boundaryDX( g1d, bcx, dir);
    return dx;
}

MPI_Matrix dx( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return dx( g, g.bcx(), no, dir);
}

MPI_Matrix dy( const MPI_Grid2d& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.y0(), g.y1(), g.n(), g.Ny());
    MPI_Matrix m = detail::dx( g1d, bcy, dir, g.communicator() );
    m.dataX().swap( m.dataY());
    for( unsigned i=0; i<m.offset().size(); i++)
        m.offset()[i] *= g.Nx()*g.n();
    if( no == not_normed) m.precond() = detail::pure_weights(g);
    m.yterm() = detail::boundaryDX( g1d, bcy, dir);
    return m;
}
MPI_Matrix dy( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return dy( g, g.bcy(), no, dir);
}
MPI_Matrix dx( const MPI_Grid3d& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix dx = detail::dx( g1d, bcx, dir, g.communicator() );
    if( no == not_normed) dx.precond() = detail::pure_weights(g);
    dx.xterm() = detail::boundaryDX( g1d, bcx, dir);
    return dx;
}

MPI_Matrix dx( const MPI_Grid3d& g, norm no = normed, direction dir = symmetric)
{
    return dx( g, g.bcx(), no, dir);
}

MPI_Matrix dy( const MPI_Grid3d& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.y0(), g.y1(), g.n(), g.Ny());
    MPI_Matrix m = detail::dx( g1d, bcy, dir, g.communicator() );
    m.dataX().swap( m.dataY());
    for( unsigned i=0; i<m.offset().size(); i++)
        m.offset()[i] *= g.Nx()*g.n();
    if( no == not_normed) m.precond() = detail::pure_weights(g);
    m.yterm() = detail::boundaryDX( g1d, bcy, dir);
    return m;
}
MPI_Matrix dy( const MPI_Grid3d& g, norm no = normed, direction dir = symmetric)
{
    return dy( g, g.bcy(), no, dir);
}

namespace detail
{
MPI_Matrix jump( const Grid1d<double>& g, bc bcx, MPI_Comm comm)
{
    unsigned n = g.n();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> a(n), b(n), bt(n);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./g.h();

    a = (l + r);
    b = -rl;
    bt = -lr;
    a = backward*t*a*forward, bt = backward*t*bt*forward, b = backward*t*b*forward;

    MPI_Matrix m(bcx, comm,  3);
    m.offset()[0] = -n, m.offset()[1] = 0, m.offset()[2] = n;
    m.dataX()[0] = bt.data(), m.dataX()[1] = a.data(), m.dataX()[2] = b.data();
    return m;
}
BoundaryTerms boundaryJump( const Grid1d<double>& g, bc bcx)
{
    //only implement symmetric laplacian
    unsigned n=g.n(), N = g.N()-2;
    double hx = g.h();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> t = create::pipj_inv(n);
    t *= 2./hx;
    std::vector<std::vector<double> > data;
    Operator<double> data_[10];
    BoundaryTerms xterm;
    if( bcx != dg::PER)
    {
        std::vector<int> row_, col_;
        row_.resize(4), col_.resize(4), data.resize(4);
        row_[1] = 0, col_[1] = 1, data_[1] = -rl;
        row_[2] = N-1, col_[2] = N-2, data_[2] = -lr;
        switch( bcx)
        {
            case( dg::DIR): 
                row_[0] = 0, col_[0] = 0, data_[0] = l+r;
                row_[3] = N-1, col_[3] = N-1, data_[3] = l+r;
                break;
            case( dg::NEU): 
                row_[0] = 0, col_[0] = 0, data_[0] = r;
                row_[3] = N-1, col_[3] = N-1, data_[3] = l;
                break;
            case( dg::DIR_NEU): 
                row_[0] = 0, col_[0] = 0, data_[0] = l+r;
                row_[3] = N-1, col_[3] = N-1, data_[3] = l;
                break;
            case( dg::NEU_DIR): 
                row_[0] = 0, col_[0] = 0, data_[0] = r;
                row_[3] = N-1, col_[3] = N-1, data_[3] = l+r;
                break;
        }
        for( unsigned i=0; i<row_.size(); i++)
        {
            data_[i] = backward*t*data_[i]*forward;
            data[i] = data_[i].data();
        }
        xterm.row_ = row_, xterm.col_ = col_, xterm.data_ = data;
    }
    return xterm;
}
}//namespace detail
MPI_Matrix jump2d( const MPI_Grid2d& g, bc bcx, bc bcy, norm no)
{
    Grid1d<double> g1dX( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    Grid1d<double> g1dY( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    MPI_Matrix lapx = detail::jump( g1dX, bcx, g.communicator() );
    MPI_Matrix lapy = detail::jump( g1dY, bcy, g.communicator() );
    lapy.dataX().swap( lapy.dataY());
    for( unsigned i=0; i<lapy.offset().size(); i++)
        lapy.offset()[i] *= g.Nx()*g.n();
    //append elements
    lapx.bcy() = bcy;
    lapx.dataX().insert( lapx.dataX().end(), lapy.dataX().begin(), lapy.dataX().end());
    lapx.dataY().insert( lapx.dataY().end(), lapy.dataY().begin(), lapy.dataY().end());
    lapx.offset().insert( lapx.offset().end(), lapy.offset().begin(), lapy.offset().end());
    if( no == not_normed)
        lapx.precond()= detail::pure_weights(g);
    lapx.xterm() = detail::boundaryJump( g1dX, bcx);
    lapx.yterm() = detail::boundaryJump( g1dY, bcy);
    return lapx;
}

MPI_Matrix jump2d( const MPI_Grid2d& g, norm no)
{
    return jump2d( g, g.bcx(), g.bcy(), no);
}

MPI_Matrix jump2d( const MPI_Grid3d& g, bc bcx, bc bcy, norm no)
{
    Grid1d<double> g1dX( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    Grid1d<double> g1dY( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    MPI_Matrix lapx = detail::jump( g1dX, bcx, g.communicator() );
    MPI_Matrix lapy = detail::jump( g1dY, bcy, g.communicator() );
    lapy.dataX().swap( lapy.dataY());
    for( unsigned i=0; i<lapy.offset().size(); i++)
        lapy.offset()[i] *= g.Nx()*g.n();
    //append elements
    lapx.bcy() = bcy;
    lapx.dataX().insert( lapx.dataX().end(), lapy.dataX().begin(), lapy.dataX().end());
    lapx.dataY().insert( lapx.dataY().end(), lapy.dataY().begin(), lapy.dataY().end());
    lapx.offset().insert( lapx.offset().end(), lapy.offset().begin(), lapy.offset().end());
    if( no == not_normed)
        lapx.precond()= detail::pure_weights(g);
    lapx.xterm() = detail::boundaryJump( g1dX, bcx);
    lapx.yterm() = detail::boundaryJump( g1dY, bcy);
    return lapx;
}

MPI_Matrix jump2d( const MPI_Grid3d& g, norm no)
{
    return jump2d( g, g.bcx(), g.bcy(), no);
}

} //namespace create
} //namespace dg
