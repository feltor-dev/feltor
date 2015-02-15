#pragma once
#include <cusp/system/cuda/arch.h>
#include <cusp/system/cuda/utils.h>
#include "../blas.h"
#include "../functors.h"



namespace dg
{

namespace create
{
namespace detail
{
void __host__ __device__ legendre( double* pxn, const double xn, const unsigned n, const double* forward)

{
    pxn[0] = 1.;
    if( n > 1)
    {
        pxn[1] = xn;
        for( unsigned i=1; i<n-1; i++)
            pxn[i+1] = ((double)(2*i+1)*xn*pxn[i]-(double)i*pxn[i-1])/(double)(i+1);
        double temp[4];
        for( unsigned k=0; k<n; k++)
        {
            for( unsigned i=0; i<n; i++)
                temp[k] = pxn[i]*forward[i*n+k];
        }
        for( unsigned k=0; k<n; k++)
            pxn[k] = temp[k];
    }
}

template<size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) //cuda performance hint macro, (max_threads_per_block, minBlocksPerMultiprocessor)
 __global__ void interpolation_kernel1d( 
         const int num_rows, 
         const int n, 
         const int* celln, const double* xn, 
         const int pitch, int* Aj, double* Av, 
         const double* forward)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<num_rows; row += grid_size)
    {
        double px[4];
        detail::legendre( px, xn[row], n, forward);
        
        int offset = row;
        unsigned col_begin = celln[row]*n;
        for(int k=0; k<n; k++)
        {
            Aj[offset] = col_begin + k*n;
            Av[offset] = px[k];
            offset +=pitch;
        }
    }

}

template<size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) //cuda performance hint macro, (max_threads_per_block, minBlocksPerMultiprocessor)
 __global__ void interpolation_kernel2d( 
         const int num_rows, const int n, const int Nx, 
         const int* celln, const int* cellm, 
         const double* xn, const double * yn, 
         const int pitch, int* Aj, double* Av, 
         const double * forward)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x;
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<num_rows; row += grid_size)
    {
        //evaluate 2d Legendre polynomials at (xn, yn)...
        double px[4], py[4];
        detail::legendre( px, xn[row], n, forward); 
        detail::legendre( py, yn[row], n, forward);
        
        int offset = row;
        int col_begin = cellm[row]*Nx*n*n + celln[row]*n;
        //evaluate 2d Legendre polynomials at (xn)...
        for( int k=0; k<n; k++)
            for( int l=0; l<n; l++)
            {
                Aj[offset] = col_begin + k*n*Nx + l;
                Av[offset] = py[k]*px[l];
                offset +=pitch;
            }
    }

}

} //namespace detail 

cusp::ell_matrix<double, int, cusp::device_memory> ell_interpolation( const thrust::device_vector<double>& x, const Grid1d<double>& g  )
{
    //allocate ell matrix storage
    cusp::ell_matrix<int, double, cusp::device_memory> A( x.size(), g.size(), x.size()*g.n(), g.n());

    //set up kernel parameters
    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks( 
            detail::interpolation_kernel1d<BLOCK_SIZE>, 
            BLOCK_SIZE, (size_t) 0  );
    const size_t NUM_BLOCKS = std::min<size_t>( 
            MAX_BLOCKS, 
            cusp::system::cuda::DIVIDE_INTO( A.num_rows, BLOCK_SIZE));
    const int pitch = A.column_indices.pitch;
    int* Aj = thrust::raw_pointer_cast(&A.column_indices(0,0));
    double * Av = thrust::raw_pointer_cast(&A.values(0,0));

    //compute normalized x values and cell numbers
    thrust::device_vector<double> xn(x.size()), cellnh(x.size());
    dg::blas1::transform( x, xn, dg::PLUS<double>( -g.x0()));
    dg::blas1::scal( xn, 1./g.h());
    thrust::device_vector<int> celln( x.size());
    thrust::transform( xn.begin(), xn.end(), celln.begin(), dg::FLOOR());
    thrust::transform( celln.begin(), celln.end(),cellnh.begin(), dg::PLUS<double>(0.5));
    const int* celln_ptr = thrust::raw_pointer_cast( &celln[0]);
    const double* xn_ptr = thrust::raw_pointer_cast( &xn[0]);
    //xn = 2*xn - 2*(celln+0.5)
    const thrust::device_vector<double> forward(std::vector<double> ( g.dlt().forward()));
    const double* forward_ptr = thrust::raw_pointer_cast( &forward[0]);
    dg::blas1::axpby( 2., xn, -2., cellnh, xn);
    detail::interpolation_kernel1d<BLOCK_SIZE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, 0>>> ( A.num_rows, g.n(), celln_ptr, xn_ptr, pitch, Aj, Av, forward_ptr);
    return A;

}

cusp::ell_matrix<double, int, cusp::device_memory> ell_interpolation( const thrust::device_vector<double>& x, const thrust::device_vector<double>& y, const Grid2d<double>& g  )
{
    //allocate ell matrix storage
    cusp::ell_matrix<int, double, cusp::device_memory> A( x.size(), g.size(), x.size()*g.n()*g.n(), g.n()*g.n());

    //set up kernel parameters
    const size_t BLOCK_SIZE = 256;
    const size_t MAX_BLOCKS = cusp::system::cuda::detail::max_active_blocks( detail::interpolation_kernel2d<BLOCK_SIZE>, BLOCK_SIZE, (size_t) 0  );
    const size_t NUM_BLOCKS = std::min<size_t>( 
            MAX_BLOCKS, 
            cusp::system::cuda::DIVIDE_INTO( A.num_rows, BLOCK_SIZE));
    const int pitch = A.column_indices.pitch;
    int* Aj = thrust::raw_pointer_cast(&A.column_indices(0,0));
    double * Av = thrust::raw_pointer_cast(&A.values(0,0));

    //compute normalized x and y values and cell numbers
    thrust::device_vector<double> xn(x.size()), yn(y.size()), cellnh(x.size()), cellmh(x.size());
    dg::blas1::transform( x, xn, dg::PLUS<double>( -g.x0()));
    dg::blas1::transform( y, yn, dg::PLUS<double>( -g.y0()));
    dg::blas1::scal( xn, 1./g.hx());
    dg::blas1::scal( yn, 1./g.hy());
    thrust::device_vector<int> celln( x.size());
    thrust::device_vector<int> cellm( x.size());
    thrust::transform( xn.begin(), xn.end(), celln.begin(), dg::FLOOR());
    thrust::transform( yn.begin(), yn.end(), cellm.begin(), dg::FLOOR());
    thrust::transform( celln.begin(), celln.end(),cellnh.begin(), dg::PLUS<double>(0.5));
    thrust::transform( cellm.begin(), cellm.end(),cellmh.begin(), dg::PLUS<double>(0.5));
    const int* celln_ptr = thrust::raw_pointer_cast( &celln[0]);
    const int* cellm_ptr = thrust::raw_pointer_cast( &cellm[0]);
    const double* xn_ptr = thrust::raw_pointer_cast( &xn[0]);
    const double* yn_ptr = thrust::raw_pointer_cast( &yn[0]);
    //xn = 2*xn - 2*(celln+0.5)
    dg::blas1::axpby( 2., xn, -2., cellnh, xn);
    dg::blas1::axpby( 2., yn, -2., cellmh, yn);
    thrust::device_vector<double> forward(std::vector<double> ( g.dlt().forward()));
    const double * forward_ptr = thrust::raw_pointer_cast( &forward[0]);
    detail::interpolation_kernel2d<BLOCK_SIZE> <<<NUM_BLOCKS, BLOCK_SIZE, 0, 0>>> ( A.num_rows, g.n(), g.Nx(), 
            celln_ptr, cellm_ptr, 
            xn_ptr, yn_ptr, pitch, Aj, Av, forward_ptr);
    return A;

}
} //namespace create
} //namespace dg
