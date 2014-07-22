#pragma once

#include "mpi_matrix.h"

namespace dg{
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
    //problematisch wenn communicator blöd ist
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

//MPI_Matrix forward_transform( const MPI_Grid2d& g)
//{
//    unsigned n = g.n();
//    Operator<double> forward = g.dlt().forward();
//    std::vector<double> weigh(g.n());
//    for( unsigned i=0; i<weigh.size(); i++)
//        weigh[i] = 1.;
//
//    MPI_Matrix m(g.bcx(), g.bcy(), g.communicator(),  1);
//    m.offset()[0] = 0;
//    m.state()[0] = +1;
//    m.weights()[0] = weigh;
//
//    m.data()[0] = forward.data();
//        
//    return m;
//}
//MPI_Matrix backward_transform( const MPI_Grid2d& g)
//{
//    unsigned n = g.n();
//    Operator<double> op = g.dlt().backward();
//    std::vector<double> weigh(g.n());
//    for( unsigned i=0; i<weigh.size(); i++)
//        weigh[i] = 1.;
//
//    MPI_Matrix m(g.bcx(), g.bcy(), g.communicator(),  1);
//    m.offset()[0] = 0;
//    m.state()[0] = +1;
//    m.weights()[0] = weigh;
//
//    m.data()[0] = op.data();
//        
//    return m;
//}

} //namespace create
} //namespace dg
