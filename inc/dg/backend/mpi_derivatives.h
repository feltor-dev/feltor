#pragma once

#include "mpi_matrix.h"

namespace dg{
namespace create
{

namespace detail
{
//create a normed 2d X-derivative
MPI_Matrix dx( const Grid1d<double>& g, bc bcx, norm no, direction dir, MPI_Comm comm)
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

} //namespace detail

MPI_Matrix dx( const MPI_Grid2d& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix dx = detail::dx( g1d, bcx, no, dir, g.communicator() );
    if( no == not_normed) dx.precond() = dg::create::weights(g);
    return dx;
}

MPI_Matrix dx( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return dx( g, g.bcx(), no, dir);
}

MPI_Matrix dy( const MPI_Grid2d& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.y0(), g.y1(), g.n(), g.Ny());
    MPI_Matrix m = detail::dx( g1d, bcy, no, dir, g.communicator() );
    m.dataX().swap( m.dataY());
    for( unsigned i=0; i<m.offset().size(); i++)
        m.offset()[i] *= g.Nx()*g.n();
    if( no == not_normed) m.precond() = dg::create::weights(g);
    return m;
}
MPI_Matrix dy( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return dy( g, g.bcy(), no, dir);
}
MPI_Matrix dx( const MPI_Grid3d& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix dx = detail::dx( g1d, bcx, no, dir, g.communicator() );
    if( no == not_normed) dx.precond() = dg::create::weights(g);
    return dx;
}

MPI_Matrix dx( const MPI_Grid3d& g, norm no = normed, direction dir = symmetric)
{
    return dx( g, g.bcx(), no, dir);
}

MPI_Matrix dy( const MPI_Grid3d& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1d( g.y0(), g.y1(), g.n(), g.Ny());
    MPI_Matrix m = detail::dx( g1d, bcy, no, dir, g.communicator() );
    m.dataX().swap( m.dataY());
    for( unsigned i=0; i<m.offset().size(); i++)
        m.offset()[i] *= g.Nx()*g.n();
    if( no == not_normed) m.precond() = dg::create::weights(g);
    return m;
}
MPI_Matrix dy( const MPI_Grid3d& g, norm no = normed, direction dir = symmetric)
{
    return dy( g, g.bcy(), no, dir);
}

namespace detail
{
MPI_Matrix dxx( const Grid1d<double>& g, bc bcx, norm no , direction dir , MPI_Comm comm)
{
    unsigned n = g.n();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    //create norm and weights
    Operator<double> a(n), b(n), bt(n);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./g.h();

    a = (lr*t*rl + (d+l)*t*(d+l).transpose() + (l + r));
    b = -(d+l)*t*rl-rl;
    if( dir == dg::backward)
    {
        a = (rl*t*lr + (d+l).transpose()*t*(d+l) + (l + r));
        b = (-rl*t*(d+l) - rl);
    }
    bt = b.transpose();
    a = backward*t*a*forward, bt = backward*t*bt*forward, b = backward*t*b*forward;

    MPI_Matrix m(bcx, comm,  3);
    m.offset()[0] = -n, m.offset()[1] = 0, m.offset()[2] = n;


    m.dataX()[0] = bt.data(), m.dataX()[1] = a.data(), m.dataX()[2] = b.data();
    return m;
}
}//namespace detail

MPI_Matrix laplacianM( const MPI_Grid2d& g, bc bcx, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1dX( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix lapx = detail::dxx( g1dX, bcx, no, dir, g.communicator() );
    Grid1d<double> g1dY( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    MPI_Matrix lapy = detail::dxx( g1dY, bcy, no, dir, g.communicator() );
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
    return lapx;
}

MPI_Matrix laplacianM( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return laplacianM( g, g.bcx(), g.bcy(), no, dir);
}

MPI_Matrix laplacianM( const MPI_Grid3d& g, bc bcx, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid1d<double> g1dX( g.x0(), g.x1(), g.n(), g.Nx(), bcx);
    MPI_Matrix lapx = detail::dxx( g1dX, bcx, no, dir, g.communicator() );
    Grid1d<double> g1dY( g.y0(), g.y1(), g.n(), g.Ny(), bcy);
    MPI_Matrix lapy = detail::dxx( g1dY, bcy, no, dir, g.communicator() );
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
    return lapx;
}

MPI_Matrix laplacianM( const MPI_Grid3d& g, norm no = normed, direction dir = symmetric)
{
    return laplacianM( g, g.bcx(), g.bcy(), no, dir);
}

} //namespace create
} //namespace dg
