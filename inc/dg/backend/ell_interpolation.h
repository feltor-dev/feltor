#pragma once
#include "../blas.h"
#include "../functors.h"



namespace dg
{

namespace create
{
    ///@cond
namespace detail
{

//find cell number of given x
struct FindCell 
{
    FindCell( double x0, double h):x0_(x0), h_(h){}
        int operator()( double x)
        {
            return floor( (x - x0_)/h_);
        }
    private:
    double x0_, h_;

};

//normalize given x to interval [-1,1[
struct Normalize
{
    Normalize( double x0, double h):x0_(x0), h_(h){}
        double operator()( double x)
        {
            return 2*((x-x0_)/h_) - 2*floor( (x - x0_)/h_) - 1;
        }
    private:
    double x0_, h_;
};

//normalize given x to interval [0, 1[
struct NormalizeZ
{
    NormalizeZ( double x0, double h): x0_(x0), h_(h){}
        double operator()( double x)
        {
            return (x-x0_) - h_*floor( (x - x0_)/h_);
        }
    private:
    double x0_, h_;
};

void legendre( double* pxn, const double xn, const unsigned n, const double* forward)

{
    pxn[0] = 1.;
    if( n > 1)
    {
        pxn[1] = xn;
        for( unsigned i=1; i<n-1; i++)
            pxn[i+1] = ((double)(2*i+1)*xn*pxn[i]-(double)i*pxn[i-1])/(double)(i+1);
        double temp[4]; //it's a bit faster with less elements
        for( unsigned k=0; k<n; k++)
        {
            temp[k] = 0;
            for( unsigned i=0; i<n; i++)
                temp[k] += pxn[i]*forward[i*n+k];
        }
        for( unsigned k=0; k<n; k++)
            pxn[k] = temp[k];
    }
}


 void interpolation_kernel2dcpu( 
         const int num_rows, const int n, const int Nx, 
         const int* cellx, const int* celly, 
         const double* xn, const double * yn, 
         const int pitch, int* Aj, double* Av, 
         const double * forward)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for( int row = 0; row<num_rows; row ++ )
    {
        //evaluate 2d Legendre polynomials at (xn, yn)...
        double px[4], py[4];
        detail::legendre( px, xn[row], n, forward); 
        detail::legendre( py, yn[row], n, forward);
        
        int offset = row;
        //the column is the row in the vector with which to multiply
        for( int k=0; k<n; k++)
            for( int l=0; l<n; l++)
            {
                Aj[offset] = (celly[row]*n+k)*n*Nx + cellx[row]*n+l;
                Av[offset] = py[k]*px[l];
                offset +=pitch;
            }
    }

}
void interpolation_kernel3dcpu( 
         const int num_rows, const int n, const int Nx, const int Ny, const int Nz, 
         const int* cellx, const int* celly, const int *cellz,
         const double* xn, const double * yn, const double *zn,
         const int pitch, int* Aj, double* Av, 
         const double * forward)
{
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for( int row = 0; row<num_rows; row ++ )
    {
        //evaluate 2d Legendre polynomials at (xn, yn)...
        double px[4], py[4], pz[2] = {1-zn[row], zn[row]};
        detail::legendre( px, xn[row], n, forward); 
        detail::legendre( py, yn[row], n, forward);
        
        int offset = row;
        for( int m=0; m<2; m++)
            for( int k=0; k<n; k++)
                for( int l=0; l<n; l++)
                {
                    //cellz is from -1 to Nz-1
                    Aj[offset] = ((cellz[row] +m)%Nz)*n*n*Nx*Ny  + (celly[row]*n+k)*n*Nx + (cellx[row]*n + l);

                    Av[offset] = pz[m]*py[k]*px[l];
                    offset +=pitch;
                }
    }

}

} //namespace detail 
///@endcond


/**
 * @brief Create an interpolation matrix on the device
 *
 * @param x Vector of x-values
 * @param y Vector of y-values
 * @param g 2D Grid on which to interpolate 
 *
 * @return interpolation matrix
 * @note n must be smaller than 5
 * @attention no range check is performed on the input vectors
 */
cusp::ell_matrix<int, double, cusp::host_memory> ell_interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const Grid2d<double>& g  )
{
    //dg::Timer t;
    assert( x.size() == y.size());
    //allocate ell matrix storage
    cusp::ell_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*g.n()*g.n(), g.n()*g.n());
    //t.tic();

    const int pitch = A.column_indices.pitch; //# of cols in memory
    assert( pitch == A.values.pitch);
    int* Aj = thrust::raw_pointer_cast(&A.column_indices(0,0));
    double * Av = thrust::raw_pointer_cast(&A.values(0,0));

    //compute normalized x and y values and cell numbers
    thrust::host_vector<int> cellX( x.size()), cellY(x.size());
    thrust::host_vector<double> xn(x.size()), yn(y.size());
    thrust::transform( x.begin(), x.end(), cellX.begin(), detail::FindCell( g.x0(), g.hx()));
    thrust::transform( x.begin(), x.end(), xn.begin(), detail::Normalize( g.x0(), g.hx()));
    thrust::transform( y.begin(), y.end(), cellY.begin(), detail::FindCell( g.y0(), g.hy()));
    thrust::transform( y.begin(), y.end(), yn.begin(), detail::Normalize( g.y0(), g.hy()));
    const int* cellX_ptr = thrust::raw_pointer_cast( &cellX[0]);
    const int* cellY_ptr = thrust::raw_pointer_cast( &cellY[0]);
    const double* xn_ptr = thrust::raw_pointer_cast( &xn[0]);
    const double* yn_ptr = thrust::raw_pointer_cast( &yn[0]);
    thrust::host_vector<double> forward(std::vector<double> ( g.dlt().forward()));
    const double * forward_ptr = thrust::raw_pointer_cast( &forward[0]);
    //t.toc();
    //std::cout << "Prekernel took "<<t.diff()<<"s\n";
    //t.tic();
    detail::interpolation_kernel2dcpu( A.num_rows, g.n(), g.Nx(), 
            cellX_ptr, cellY_ptr, 
            xn_ptr, yn_ptr, pitch, Aj, Av, forward_ptr);
    //t.toc();
    //std::cout << "   kernel took "<<t.diff()<<"s\n";
    return A;

}

/**
 * @brief Create an interpolation matrix on the host
 *
 * uses dG interpolation in x and y and linear interpolation in z
 * @param x Vector of x-values
 * @param y Vector of y-values
 * @param z Vector of z-values
 * @param g 3D Grid on which to interpolate z direction is assumed periodic
 *
 * @return interpolation matrix
 * @note n must be smaller than or equal to 4
 * @attention no range check is performed on the input vectors
 */
cusp::ell_matrix<int, double, cusp::host_memory> ell_interpolation( const thrust::host_vector<double>& x, const thrust::host_vector<double>& y, const thrust::host_vector<double>& z, const Grid3d<double>& g )
{
    assert( x.size() == y.size());
    assert( x.size() == z.size());
    //allocate ell matrix storage
    cusp::ell_matrix<int, double, cusp::host_memory> A( x.size(), g.size(), x.size()*2*g.n()*g.n(), 2*g.n()*g.n());

    //set up kernel parameters
    const int pitch = A.column_indices.pitch; //# of cols in memory
    assert( pitch == A.values.pitch);
    int* Aj = thrust::raw_pointer_cast(&A.column_indices(0,0));
    double * Av = thrust::raw_pointer_cast(&A.values(0,0));

    //compute normalized x and y values and cell numbers
    thrust::host_vector<int> cellX( x.size()), cellY(x.size()), cellZ(x.size());
    thrust::host_vector<double> xn(x.size()), yn(y.size()), zn(z.size());
    thrust::transform( x.begin(), x.end(), cellX.begin(), detail::FindCell( g.x0(), g.hx()));
    thrust::transform( x.begin(), x.end(), xn.begin(), detail::Normalize( g.x0(), g.hx()));
    thrust::transform( y.begin(), y.end(), cellY.begin(), detail::FindCell( g.y0(), g.hy()));
    thrust::transform( y.begin(), y.end(), yn.begin(), detail::Normalize( g.y0(), g.hy()));
    //z-planes are not cell-centered so shift cells by h/2
    thrust::transform( z.begin(), z.end(), cellZ.begin(), detail::FindCell( g.z0()+g.hz()/2., g.hz()));
    thrust::transform( z.begin(), z.end(), zn.begin(), detail::NormalizeZ( g.z0()+g.hz()/2., g.hz()));
    const int* cellX_ptr = thrust::raw_pointer_cast( &cellX[0]);
    const int* cellY_ptr = thrust::raw_pointer_cast( &cellY[0]);
    const int* cellZ_ptr = thrust::raw_pointer_cast( &cellZ[0]);
    const double* xn_ptr = thrust::raw_pointer_cast( &xn[0]);
    const double* yn_ptr = thrust::raw_pointer_cast( &yn[0]);
    const double* zn_ptr = thrust::raw_pointer_cast( &zn[0]);
    thrust::host_vector<double> forward(std::vector<double> ( g.dlt().forward()));
    const double * forward_ptr = thrust::raw_pointer_cast( &forward[0]);
    detail::interpolation_kernel3dcpu( A.num_rows, g.n(), g.Nx(), g.Ny(), g.Nz(),
            cellX_ptr, cellY_ptr, cellZ_ptr,
            xn_ptr, yn_ptr, zn_ptr, pitch, Aj, Av, forward_ptr);
    return A;

}
/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 * 
 * @param g_new The new points 
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old brid must lie within the boundaries of the new grid
 */
cusp::ell_matrix<int, double, cusp::host_memory> ell_interpolation( const Grid3d<double>& g_new, const Grid3d<double>& g_old)
{
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    assert( g_new.z0() >= g_old.z0());
    assert( g_new.z1() <= g_old.z1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::coo1, g_new);
    thrust::host_vector<double> pointsY = dg::evaluate( dg::coo2, g_new);
    thrust::host_vector<double> pointsZ = dg::evaluate( dg::coo3, g_new);
    return ell_interpolation( pointsX, pointsY, pointsZ, g_old);
}
/**
 * @brief Create interpolation between two grids
 *
 * This matrix can be applied to vectors defined on the old grid to obtain
 * its values on the new grid.
 * 
 * @param g_new The new points 
 * @param g_old The old grid
 *
 * @return Interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
cusp::ell_matrix<int, double, cusp::host_memory> ell_interpolation( const Grid2d<double>& g_new, const Grid2d<double>& g_old)
{
    //assert both grids are on the same box
    assert( g_new.x0() >= g_old.x0());
    assert( g_new.x1() <= g_old.x1());
    assert( g_new.y0() >= g_old.y0());
    assert( g_new.y1() <= g_old.y1());
    thrust::host_vector<double> pointsX = dg::evaluate( dg::coo1, g_new);
    thrust::host_vector<double> pointsY = dg::evaluate( dg::coo2, g_new);
    return ell_interpolation( pointsX, pointsY, g_old);

}

} //namespace create
} //namespace dg
