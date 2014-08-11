#pragma once

#include "mpi_matrix.h"

namespace dg{
namespace create
{

namespace detail
{
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
    BoundaryTerms xterm;
    std::vector<int> row_, col_;
    std::vector<std::vector<double> > data;
    Operator<double> data_[4];
    if( bcx != dg::PER)
    {
        if( dir == dg::symmetric)
        {
            row_.resize(4), col_.resize(4), data.resize(4);
            row_[0] = col_[0] = 0, row_[1] = 0, col_[1] = 1;
            row_[2] = col_[2] = g.N()-1, row_[3] = g.N()-1, col_[3] = g.N()-2;
            data_[1] = rl;
            data_[3] = -lr;
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
            row_[2] = col_[2] = g.N()-1;
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
        else
        {
            row_.resize(3), col_.resize(3), data.resize(3);
            row_[0] = col_[0] = 0;
            row_[2] = col_[2] = g.N()-1, row_[1] = g.N()-1, col_[1] = g.N()-2;
            data_[1] = -lr;
            switch( bcx)
            {
                case( dg::DIR): data_[2] = -d.transpose(); 
                                data_[0] = (d+l); 
                                break;
                case( dg::NEU): data_[2] = (d+l); 
                                data_[0] = d;
                                break;
                case( dg::DIR_NEU): data_[2] = -d.transpose();
                                    data_[0] = d;
                                    break;
                case( dg::NEU_DIR): data_[2] = (d+l);
                                    data_[0] = (d+l);
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
    if( no == not_normed) dx.precond() = dg::create::weights(g);
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
    if( no == not_normed) m.precond() = dg::create::weights(g);
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
    if( no == not_normed) dx.precond() = dg::create::weights(g);
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
    if( no == not_normed) m.precond() = dg::create::weights(g);
    m.yterm() = detail::boundaryDX( g1d, bcy, dir);
    return m;
}
MPI_Matrix dy( const MPI_Grid3d& g, norm no = normed, direction dir = symmetric)
{
    return dy( g, g.bcy(), no, dir);
}

namespace detail
{
MPI_Matrix dxx( const Grid1d<double>& g, bc bcx, direction dir , MPI_Comm comm)
{
    //only implement symmetric version
    unsigned n = g.n();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> a(n), b(n), bt(n), ap(a), bp(a), btp(bt);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./g.h();

    a = (lr*t*rl + (d+l)*t*(d+l).transpose() + (l + r));
    b = -(d+l)*t*rl-rl;
    ap = (rl*t*lr + (d+l).transpose()*t*(d+l) + (l + r));
    bp  = (-rl*t*(d+l) - rl);
    
    bt = b.transpose();
    btp = bp.transpose();
    a = 0.5*backward*t*(a+ap)*forward, 
    bt = 0.5*backward*t*(bt+btp)*forward, 
    b = 0.5*backward*t*(b+bp)*forward;

    MPI_Matrix m(bcx, comm,  3);
    m.offset()[0] = -n, m.offset()[1] = 0, m.offset()[2] = n;


    m.dataX()[0] = bt.data(), m.dataX()[1] = a.data(), m.dataX()[2] = b.data();
    return m;
}
BoundaryTerms boundaryDXX( const Grid1d<double>& g, bc bcx, direction dir)
{
    //only implement symmetric laplacian
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
    BoundaryTerms xterm;
    Operator<double> a(n), b(a), bt(a), bp(b), bpt(bp), ap(a), app(a), appp(a); 
    a = 0.5*( (lr*t*rl + (d+l)*t*(d+l).transpose() + l + r)
            + (rl*t*lr + (d+l).transpose()*t*(d+l) + l + r) );
    b = 0.5*(-(d+l)*t*rl-rl + -rl*t*(d+l) - rl);
    bt = b.transpose();
    bp = 0.5*(-d*t*rl-rl - rl*t*d -rl);
    bpt = bp.transpose();
    ap = 0.5*(d*t*d.transpose()+l+r + d.transpose()*t*d+l+r);
    app = 0.5*((d+l)*t*(d+l).transpose()+r + (d+l).transpose()*t*(d+l)+r);
    appp = 0.5*(lr*t*rl+d.transpose()*t*d + l + rl*t*lr + d*t*d.transpose() + l);
    std::vector<int> row_, col_;
    std::vector<std::vector<double> > data;
    Operator<double> data_[5];
    if( bcx != dg::PER)
    {
        std::vector<int> row_, col_;
        switch( bcx)
        {
            case( dg::DIR): 
                row_.resize(5), col_.resize(5), data.resize(5);
                row_[0] = 0, col_[0] = 0, data_[0] = ap;
                row_[1] = 0, col_[1] = 1, data_[1] = bp;
                row_[2] = 1, col_[2] = 0, data_[2] = bpt;
                row_[3] = 1, col_[3] = 1, data_[3] = a;
                row_[4] = 1, col_[4] = 2, data_[4] = b;
                break;
            case( dg::NEU): 
                row_.resize(4), col_.resize(4), data.resize(4);
                row_[0] = 0, col_[0] = 0, data_[0] = app;
                row_[1] = 0, col_[1] = 1, data_[1] = b;
                row_[2] = g.N()-1, col_[2] = g.N()-2, data_[2] = bt;
                row_[3] = g.N()-1, col_[3] = g.N()-1, data_[3] = appp;
                break;
            case( dg::DIR_NEU): 
                row_.resize(5), col_.resize(5), data.resize(5);
                row_[0] = 0, col_[0] = 0, data_[0] = ap;
                row_[1] = 0, col_[1] = 1, data_[1] = bp;
                row_[2] = 1, col_[2] = 0, data_[2] = bpt;
                row_[3] = g.N()-1, col_[3] = g.N()-2, data_[3] = bt;
                row_[4] = g.N()-1, col_[4] = g.N()-1, data_[4] = appp;
                break;
            case( dg::NEU_DIR): 
                row_.resize(2), col_.resize(2), data.resize(2);
                row_[0] = 0, col_[0] = 0, data_[0] = app;
                row_[1] = 0, col_[1] = 1, data_[1] = b;
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

MPI_Matrix laplacianM( const MPI_Grid2d& g, bc bcx, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1dX( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix lapx = detail::dxx( g1dX, bcx, dir, g.communicator() );
    Grid1d<double> g1dY( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    MPI_Matrix lapy = detail::dxx( g1dY, bcy, dir, g.communicator() );
    lapy.dataX().swap( lapy.dataY());
    for( unsigned i=0; i<lapy.offset().size(); i++)
        lapy.offset()[i] *= g.Nx()*g.n();
    //append elements
    lapx.bcy() = bcy;
    lapx.dataX().insert( lapx.dataX().end(), lapy.dataX().begin(), lapy.dataX().end());
    lapx.dataY().insert( lapx.dataY().end(), lapy.dataY().begin(), lapy.dataY().end());
    lapx.offset().insert( lapx.offset().end(), lapy.offset().begin(), lapy.offset().end());
    lapx.xterm() = detail::boundaryDXX( g1dX, bcx, dir);
    lapx.yterm() = detail::boundaryDXX( g1dY, bcy, dir);
    if( no == not_normed)
        lapx.precond()= dg::create::weights(g);
    return lapx;
}

MPI_Matrix laplacianM( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return laplacianM( g, g.bcx(), g.bcy(), no, dir);
}

MPI_Matrix laplacianM_perp( const MPI_Grid3d& g, bc bcx, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1dX( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix lapx = detail::dxx( g1dX, bcx, dir, g.communicator() );
    Grid1d<double> g1dY( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    MPI_Matrix lapy = detail::dxx( g1dY, bcy, dir, g.communicator() );
    lapy.dataX().swap( lapy.dataY());
    for( unsigned i=0; i<lapy.offset().size(); i++)
        lapy.offset()[i] *= g.Nx()*g.n();
    //append elements
    lapx.bcy() = bcy;
    lapx.dataX().insert( lapx.dataX().end(), lapy.dataX().begin(), lapy.dataX().end());
    lapx.dataY().insert( lapx.dataY().end(), lapy.dataY().begin(), lapy.dataY().end());
    lapx.offset().insert( lapx.offset().end(), lapy.offset().begin(), lapy.offset().end());
    if( no == not_normed)
        lapx.precond()= dg::create::weights(g);
    lapx.xterm() = detail::boundaryDXX( g1dX, bcx, dir);
    lapx.yterm() = detail::boundaryDXX( g1dY, bcy, dir);
    return lapx;
}

MPI_Matrix laplacianM_perp( const MPI_Grid3d& g, norm no = normed, direction dir = symmetric)
{
    return laplacianM_perp( g, g.bcx(), g.bcy(), no, dir);
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
}//namespace detail
MPI_Matrix jump2d( const MPI_Grid2d& g, bc bcx, bc bcy)
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
    //jump is never normed 
    lapx.precond()= dg::create::weights(g);
    return lapx;
}

MPI_Matrix jump2d( const MPI_Grid2d& g)
{
    return jump2d( g, g.bcx(), g.bcy());
}

MPI_Matrix jump2d( const MPI_Grid3d& g, bc bcx, bc bcy)
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
    //jump is never normed
    lapx.precond()= dg::create::weights(g);
    return lapx;
}

MPI_Matrix jump2d( const MPI_Grid3d& g)
{
    return jump2d( g, g.bcx(), g.bcy());
}

} //namespace create
} //namespace dg
