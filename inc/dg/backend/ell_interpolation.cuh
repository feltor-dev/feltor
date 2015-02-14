#pragma once



namespace dg
{

namespace create
{
namespace detail
{
void __host__ __device__ legendre( double* pxn, const double* xn, unsigned n)

{
    pxn[0] = 1.;
    if( n > 1)
    {
        px[1] = xn;
        for( unsigned i=1; i<n-1; i++)
            px[i+1] = ((double)(2*i+1)*xn*px[i]-(double)i*px[i-1])/(double)(i+1);
    }
}

template<size_t BLOCK_SIZE>
__launch_bounds__(BLOCK_SIZE, 1) //cuda performance hint macro, (max_threads_per_block, minBlocksPerMultiprocessor)
 __global__ void 
interpolation_kernel( const int num_rows, const int n, int* celln, const double* xn, const int pitch, int* Aj, double Av*)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<num_rows; row += grid_size)
    {
        double px[n];
        detail::legendre( px, &xn[row], n), 
        
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
 __global__ void 
interpolation_kernel( const int num_rows, const int n, const int Nx, int* celln, int* cellm, const double* xn, const double * yn, const int pitch, int* Aj, double Av*)
{
    const int thread_id = blockDim.x * blockIdx.x + threadIdx.x;
    const int grid_size = gridDim.x*blockDim.x
    //every thread takes num_rows/grid_size rows
    for( int row = thread_id; row<num_rows; row += grid_size)
    {
        //evaluate 2d Legendre polynomials at (xn, yn)...
        double px[n], py[n], pxy[n*n];
        detail::legendre( px, &xn[row], n), 
        detail::legendre( py, &yn[row], n), 
        
        int offset = row;
        int col_begin = cellm[row]*Nx*n*n + celln*n;
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

cusp::ell_matrix<double, int, cusp::device_memory> interpolation( thrust::device_vector<double>& x, const Grid1d<double>& g  )
{
    cusp::ell_matrix<int, double, cusp::device_memory> A( x.size(), x.size(), g.size(), g.n());

    thrust::device_vector<double> xn(x.size()), cellnh(x.size());
    dg::blas1::transform( x, xn, dg::PLUS<double>( -g.x0()));
    dg::blas1::scal( xn, 1./g.h());
    thrust::device_vector<int> celln( x.size());
    thrust::transform( xn.begin(), xn.end(), celln.begin(), dg::FLOOR());
    thrust::transform( celln.begin(), celln.end(),cellnh.begin(), dg::PLUS<double>(0.5));
    //xn = 2*xn - 2*(celln+0.5)
    dg::blas1::axpby( 2., xn, -2., cellnh, xn);

    


}
} //namespace detail 
} //namespace create
} //namespace dg
