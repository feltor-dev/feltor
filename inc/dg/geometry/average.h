#pragma once

#include "timer.cuh"
#include "dg/geometry/evaluation.cuh"
#include "dg/geometry/grid.h"
#include "dg/geometry/weights.cuh"
#ifdef MPI_VERSION
#include "dg/geometry/mpi_grid.h"
#endif
#include "dg/blas1.h"
#include "memory.h"
#include "dg/geometry/split_and_join.h"
#include "dg/functors.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

//template<class container>
//void poloidal_average( const container& in, container& out, const aTopology2d& g)
//{
//    dg::Timer t;
//    t.tic();
//    Grid1d g1d( g.x0(), g.x1(), g.n(), g.Ny());
//    container w1d = transfer<container>( create::weights( g1d));
//    VectorView<container> w1d_view(w1d);
//    blas1::scal( w1d, 1./g.ly());
//    t.toc();
//    std::cout << "w1d creation took "<<t.diff()<<"s\n";
//    t.tic();
//    int nx = g.n()*g.Nx(), ny = g.n()*g.Ny();
//    detail::invert_xy( nx, ny, in, out);
//    t.toc();
//    std::cout << "transposition ook "<<t.diff()<<"s\n";
//    t.tic();
//    for ( int i=0; i<nx; i++)
//    {
//        VectorView<container> row( thrust::raw_pointer_cast(out.data())+i*ny, thrust::raw_pointer_cast(out.data())+(i+1)*ny);
//        double avg = blas1::dot( row, w1d_view);
//        dg::blas1::transform( row, row, dg::CONSTANT(avg));
//    }
//    t.toc();
//    std::cout << "Averages     took "<<t.diff()<<"s\n";
//    t.tic();
//    container tmp(out);
//    detail::invert_xy( ny, nx, out, tmp);
//    tmp.swap(out);
//
//    t.toc();
//    std::cout << "Copy         took "<<t.diff()<<"s\n";
//}
//
//#ifdef MPI_VERSION
//template<class container>
//void poloidal_average( const MPI_Vector<container>& in, MPI_Vector<container>& out, const aMPITopology2d& g)
//{
//    const Grid2d& l = g.local();
//    Grid1d g1d( l.x0(), l.x1(), l.n(), l.Ny());
//    MPI_Vector<container > w1d( transfer<container>(create::weights( g1d)), g.get_poloidal_comm());
//    MPI_Vector<VectorView<container>> w1d_view( w1d.data(), w1d.communicator());
//    blas1::scal( w1d, 1./g.ly());
//    int nx = l.n()*l.Nx(), ny = l.n()*l.Ny();
//    detail::invert_xy( nx, ny, in.data(), out.data());
//    std::vector<double> avgs(nx);
//    for ( int i=0; i<nx; i++)
//    {
//        VectorView<container> row( thrust::raw_pointer_cast(out.data().data())+i*ny, thrust::raw_pointer_cast(out.data().data())+(i+1)*ny);
//        MPI_Vector<VectorView<container>> row( row, w1d.communicator());
//        avgs[i] = blas1::dot( row, w1d_view);
//    }
//    detail::copy_transpose( nx, ny, avgs, out.data());
//}
//#endif



/**
 * @brief MPI specialized class for y average computations
 *
 * @snippet backend/average_mpit.cu doxygen
 * @ingroup utilities
 * @tparam container Currently this is one of 
 *  - \c dg::HVec, \c dg::DVec, \c dg::MHVec or \c dg::MDVec  
 */
template< class Topology2d, class container>
struct Average
{
    /**
     * @brief Construct from grid mpi object
     * @param g 2d MPITopology
     */
    Average( const Topology2d& g, enum Coordinate direction): 
    m_g2d(g), m_dir(dir)
    {
        m_w1dy=dg::transfer<container>(dg::detail::create_weightsY1d(g));
        container w2d = dg::transfer<container>(dg::create::weights(g));
        dg::split_poloidal( w2d, m_split, g);
    }
    /**
     * @brief Compute the average in y-direction
     *
     * @param src 2D Source Vector (must have the same size as the grid given in the constructor)
     * @param res 2D result Vector (may alias src), every line contains the x-dependent average over
     the y-direction of src 
     */
    void operator() (const container& src, container& res)
    {
        dg::transpose( m_nx, m_ny, src, res);
        for( unsigned i=0; i<m_split.size(); i++)
        {
            double value = dg::blas1::dot( m_split[i], m_w1dy);
            dg::blas1::transform( m_split[i], m_split[i], dg::CONSTANT(value));
        }
        dg::join_poloidal(m_split, res, m_g2d);
    }
  private:
    unsigned m_nx, m_ny;
    container m_w1dy; 
    get_host_grid<Topology2d> m_g2d;
    enum Coordinate m_dir;

};


}//namespace dg
