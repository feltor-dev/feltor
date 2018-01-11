#pragma once

#include "timer.cuh"
#include "evaluation.cuh"
#include "grid.h"
#include "weights.cuh"
#ifdef MPI_VERSION
#include "mpi_grid.h"
#endif
#include "../blas1.h"
#include "memory.h"
#include "split_and_join.h"
#include "functors.h"

/*! @file 
  @brief contains classes for poloidal and toroidal average computations.
  */
namespace dg{

///@cond
namespace detail
{
/////////////////////////////////////////////poloidal split/////////////////////
void transpose_dispatch( SerialTag, unsigned nx, unsigned ny, const double* in, double* out)
{
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[j*ny+i] = in[i*nx+j];
}
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
__global__
void transpose_gpu_kernel( unsigned nx, unsigned ny, const double* in, double* out)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    const int size = nx*ny;
    for( int row = thread_id; row<size; row += grid_size)
    {
        int i=row/nx, j = row%nx;
        out[j*ny+i] = in[i*nx+j];
    }
}
void transpose_dispatch( CudaTag, unsigned nx, unsigned ny, const double* in, double* out){
    const size_t BLOCK_SIZE = 256; 
    const size_t NUM_BLOCKS = std::min<size_t>((nx*ny-1)/BLOCK_SIZE+1, 65000);
    transpose_gpu_kernel<<<NUM_BLOCKS, BLOCK_SIZE>>>( nx, ny, in, out);
}
#else
void transpose_dispatch( OmpTag, unsigned nx, unsigned ny, const double* in, double* out)
{
#pragma omp parallel for
    for( unsigned i=0; i<ny; i++)
        for( unsigned j=0; j<nx; j++)
            out[j*ny+i] = in[i*nx+j];
}
#endif
template<class container>
void invert_xy( unsigned nx, unsigned ny, const container& in, container& out)
{
    assert(&in != &out);
    const get_value_type<container>* in_ptr = thrust::raw_pointer_cast( in.data());
    get_value_type<container>* out_ptr = thrust::raw_pointer_cast( out.data());
    return transpose_dispatch( get_execution_policy<container>(), nx, ny, in_ptr, out_ptr);
}

}//namespace detail
///@endcond

template<class container>
void poloidal_average( const container& in, container& out, const aTopology2d& g)
{
    dg::Timer t;
    t.tic();
    Grid1d g1d( g.x0(), g.x1(), g.n(), g.Ny());
    container w1d = transfer<container>( create::weights( g1d));
    VectorView<container> w1d_view(w1d);
    blas1::scal( w1d, 1./g.ly());
    t.toc();
    std::cout << "w1d creation took "<<t.diff()<<"s\n";
    t.tic();
    int nx = g.n()*g.Nx(), ny = g.n()*g.Ny();
    detail::invert_xy( nx, ny, in, out);
    t.toc();
    std::cout << "transposition ook "<<t.diff()<<"s\n";
    t.tic();
    for ( int i=0; i<nx; i++)
    {
        VectorView<container> row( thrust::raw_pointer_cast(out.data())+i*ny, thrust::raw_pointer_cast(out.data())+(i+1)*ny);
        double avg = blas1::dot( row, w1d_view);
        dg::blas1::transform( row, row, dg::CONSTANT(avg));
    }
    t.toc();
    std::cout << "Averages     took "<<t.diff()<<"s\n";
    t.tic();
    container tmp(out);
    detail::invert_xy( ny, nx, out, tmp);
    tmp.swap(out);

    t.toc();
    std::cout << "Copy         took "<<t.diff()<<"s\n";
}

#ifdef MPI_VERSION
template<class container>
void poloidal_average( const MPI_Vector<container>& in, MPI_Vector<container>& out, const aMPITopology2d& g)
{
    const Grid2d& l = g.local();
    Grid1d g1d( l.x0(), l.x1(), l.n(), l.Ny());
    MPI_Vector<container > w1d( transfer<container>(create::weights( g1d)), g.get_poloidal_comm());
    MPI_Vector<VectorView<container>> w1d_view( w1d.data(), w1d.communicator());
    blas1::scal( w1d, 1./g.ly());
    int nx = l.n()*l.Nx(), ny = l.n()*l.Ny();
    detail::invert_xy( nx, ny, in.data(), out.data());
    std::vector<double> avgs(nx);
    for ( int i=0; i<nx; i++)
    {
        VectorView<container> row( thrust::raw_pointer_cast(out.data().data())+i*ny, thrust::raw_pointer_cast(out.data().data())+(i+1)*ny);
        MPI_Vector<VectorView<container>> row( row, w1d.communicator());
        avgs[i] = blas1::dot( row, w1d_view);
    }
    detail::copy_transpose( nx, ny, avgs, out.data());
}
#endif



/**
 * @brief MPI specialized class for y average computations
 *
 * @snippet backend/average_mpit.cu doxygen
 * @ingroup utilities
 * @tparam container Currently this is one of 
 *  - \c dg::HVec, \c dg::DVec, \c dg::MHVec or \c dg::MDVec  
 */
//template< class Topology2d, class container>
//struct PoloidalAverage
//{
//    /**
//     * @brief Construct from grid mpi object
//     * @param g 2d MPITopology
//     */
//    PoloidalAverage( const Topology2d& g): 
//    m_g2d(g)
//    {
//        m_w1dy=dg::transfer<container>(dg::detail::create_weightsY1d(g));
//        container w2d = dg::transfer<container>(dg::create::weights(g));
//        dg::split_poloidal( w2d, m_split, g);
//    }
//    /**
//     * @brief Compute the average in y-direction
//     *
//     * @param src 2D Source Vector (must have the same size as the grid given in the constructor)
//     * @param res 2D result Vector (may alias src), every line contains the x-dependent average over
//     the y-direction of src 
//     */
//    void operator() (const container& src, container& res)
//    {
//        dg::split_poloidal( src, m_split, m_g2d);
//        for( unsigned i=0; i<m_split.size(); i++)
//        {
//            double value = dg::blas1::dot( m_split[i], m_w1dy);
//            dg::blas1::transform( m_split[i], m_split[i], dg::CONSTANT(value));
//        }
//        dg::join_poloidal(m_split, res, m_g2d);
//    }
//  private:
//    container m_w1dy; 
//    std::vector<container> m_split;
//    get_host_grid<Topology2d> m_g2d;
//
//};


}//namespace dg
