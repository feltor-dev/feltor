#pragma once

#include "mpi_matrix.h"

namespace dg{
namespace create
{

namespace detail
{
MPI_Matrix dx( const Grid2d<double>& g, bc bcx, bc bcy, norm no, direction dir, MPI_Comm comm)
{
    unsigned n=g.n();
    double hx = g.hx(), hy = g.hy();
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> forward = g.dlt().forward();
    Operator<double> backward= g.dlt().backward();
    Operator<double> normx(g.n(), 0.), normy(g.n(), 0.);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./hx;
    //create norm and weights
    if( no == not_normed)
    {
        for( unsigned i=0; i<n; i++)
            normx( i,i) = normy( i,i) = g.dlt().weights()[i];
        normx *= hx/2.;
        normy *= hy/2.; // normalisation because F is invariant
    }
    else
    {
        normx = normy = create::delta(n);
    }

    Operator<double> a(n), b(n), bt(n);
    if( dir == dg::symmetric)
    {
        //for( unsigned i=0; i<weigh.size(); i++)
            //std::cout << weigh[i] << " \n";

        MPI_Matrix m(bcx, bcy, comm,  3);
        m.offset()[0] = -n, m.offset()[1] = 0, m.offset()[2] = n;
        if( no == not_normed)
            m.dataY()[0] = m.dataY()[1] = m.dataY()[2] = normy.data();

        bt = normx*backward*t*(-0.5*lr )*forward; 
        a  = normx*backward*t*( 0.5*(d-d.transpose()) )*forward;
        b  = normx*backward*t*( 0.5*rl )*forward;

        m.dataX()[0] = bt.data(), m.dataX()[1] = a.data(), m.dataX()[2] = b.data();
        //m.cusp_matrix() = dg::create::dx( g.local(), no, dir);
        
        return m;
    }
    if( dir == dg::forward)
    {
        MPI_Matrix m(bcx, bcy, comm,  2);
        m.offset()[0] = 0, m.offset()[1] = n;
        if( no == not_normed)
            m.dataY()[0] = m.dataY()[1] = normy.data();

        a = normx*backward*t*(-d.transpose()-l)*forward; 
        b = normx*backward*t*(rl)*forward;
        m.dataX()[0] = a.data(), m.dataX()[1] = b.data();
        //m.cusp_matrix() = dg::create::dx( g.local(), no, dir);
        return m;
    }
    //if dir == dg::backward
    MPI_Matrix m(bcx, bcy, comm,  2);
    m.offset()[0] = -n, m.offset()[1] = 0;
    if( no == not_normed)
        m.dataY()[0] = m.dataY()[1] = normy.data();
    bt = normx*backward*t*(-lr)*forward; 
    a  = normx*backward*t*(d+l)*forward;
    m.dataX()[0] = bt.data(), m.dataX()[1] = a.data();
    //m.cusp_matrix() = dg::create::dx( g.local(), no, dir);
    return m;
}

} //namespace detail
MPI_Matrix dx( const MPI_Grid2d& g, bc bcx, norm no = normed, direction dir = symmetric)
{
    return detail::dx( g.local(), bcx, g.bcy(), no, dir, g.communicator() );
}

MPI_Matrix dx( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return detail::dx( g.local(), g.bcx(), g.bcy(), no, dir, g.communicator() );
}

MPI_Matrix dy( const MPI_Grid2d& g, bc bcy, norm no = normed, direction dir = symmetric)
{
    Grid2d<double> swap( g.y0(), g.y1(), g.x0(), g.x1(), g.n(), g.Ny(), g.Nx());
    MPI_Matrix m = detail::dx( swap, bcy, g.bcx(), no, dir, g.communicator() );
    m.dataX().swap( m.dataY());
    for( unsigned i=0; i<m.offset().size(); i++)
        m.offset()[i] *= g.Nx()*g.n();
    return m;
}
MPI_Matrix dy( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return dy( g, g.bcy(), no, dir);
}

namespace detail
{
MPI_Matrix dxx( const Grid2d<double>& g, bc bcx, norm no , direction dir , MPI_Comm comm)
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

    MPI_Matrix m(bcx, g.bcy(), comm,  3);
    m.offset()[0] = -n, m.offset()[1] = 0, m.offset()[2] = n;

    if( no == not_normed)
        m.dataY()[0] = m.dataY()[1] = m.dataY()[2] = normy.data();
    m.dataX()[0] = bt.data(), m.dataX()[1] = a.data(), m.dataX()[2] = b.data();
    return m;
}
}//namespace detail

MPI_Matrix laplacianM( const MPI_Grid2d& g, bc bcx, bc bcy, norm no = normed, direction dir = symmetric)
{
    MPI_Matrix lapx = detail::dxx( g.local(), bcx, no, dir , g.communicator());
    Grid2d<double> swap( g.y0(), g.y1(), g.x0(), g.x1(), g.n(), g.Ny(), g.Nx());
    MPI_Matrix lapy = detail::dxx( swap, bcy, no, dir, g.communicator() );
    lapy.dataX().swap( lapy.dataY());
    //for( unsigned i=0; i<lapy.offset().size(); i++)
        //lapy.offset()[i] *= g.Nx()*g.n();

    for( unsigned i=0; i<g.n()*g.n(); i++)
    {
        lapx.dataX()[1][i] += lapy.dataX()[1][i];
        lapx.dataY()[1][i] += lapy.dataY()[1][i];
    }

    //append elements
    lapx.bcy() = bcy;
    lapx.dataX().push_back( lapx.dataX()[0]);
    lapx.dataY().push_back( lapy.dataY()[0]);
    lapx.offset().push_back( lapy.offset()[0] );
    lapx.dataX().push_back( lapx.dataX()[2]);
    lapx.dataY().push_back( lapy.dataY()[2]);
    lapx.offset().push_back( lapy.offset()[2] );
    return lapx;
}

MPI_Matrix laplacianM( const MPI_Grid2d& g, norm no = normed, direction dir = symmetric)
{
    return laplacianM( g, g.bcx(), g.bcy(), no, dir);
}

} //namespace create
} //namespace dg
