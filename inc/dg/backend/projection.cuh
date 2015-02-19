#pragma once
#include <vector>
#include <cusp/coo_matrix.h>
#include "grid.h"
#include "xspacelib.cuh"
#include "matrix_traits_thrust.h"
/*!@file 
  
  contains the Difference Norm class that computes differences between vectors on different grids
 */
namespace dg{
/**
 * @brief Greatest common divisor
 *
 * @param a First number
 * @param b Second number
 *
 * @return greatest common divisor
 */
unsigned gcd( unsigned a, unsigned b)
{
    unsigned r2 = std::max(a,b);
    unsigned r1 = std::min(a,b);
    while( r1!=0)
    {
        r2 = r2%r1;
        std::swap( r1, r2);
    }
    return r2;
}
/**
 * @brief Least common multiple
 *
 * @param a Fist number
 * @param b Second number 
 *
 * @return Least common multiple
 */
unsigned lcm( unsigned a, unsigned b)
{
    unsigned g = gcd( a,b);
    return a/g*b;
}
namespace create{

/**
 * @brief Create a 1D projection matrix onto a finer grid
 *
 * Grid space must be equal. Nx of the second grid must be a multiple of 
 * Nx of the first grid.
 * @param g1 Grid of the original vector
 * @param g2 Grid of the target vector
 *
 * @return Projection matrix
 */
cusp::coo_matrix< int, double, cusp::host_memory> projection( const Grid1d<double>& g1, const Grid1d<double>& g2)
{
    assert( g1.x0() == g2.x0()); assert( g1.x1() == g2.x1());
    assert( g2.N() % g1.N() == 0);
    return dg::create::interpolation( g2, g1);
}
/**
 * @brief Create a 2D projection matrix onto a finer grid
 *
 * Grid space must be equal. Nx and Ny of the second grid must be multiples of 
 * Nx and Ny of the first grid.
 * @param g1 Grid of the original vector
 * @param g2 Grid of the target vector
 *
 * @return Projection matrix
 */
cusp::coo_matrix< int, double, cusp::host_memory> projection( const Grid2d<double>& g1, const Grid2d<double>& g2)
{
    //TODO: projection in y direction needs permutation
    assert( g1.x0() == g2.x0()); assert( g1.x1() == g2.x1());
    assert( g1.y0() == g2.y0()); assert( g1.y1() == g2.y1());
    //assert( g2.Nx() % g1.Nx() == 0);
    //assert( g2.Ny() % g1.Ny() == 0);
    return dg::create::interpolation( g2, g1);
}


}//namespace create


//eventuell könnte man zwei Projektionsmatrizen malnehmen um eine kleinere zu erhalten
/**
 * @brief Class to perform comparison of dG vectors on different grids
 *
 * it basically interpolates values from the rougher grid to values on the finer grid and then uses the existing methods to compute the norm
 *@ingroup utilities
 * @tparam container
 */
template <typename container>
struct DifferenceNorm
{
    /**
     * @brief Construct from two different grids
     *
     * @param g1 first grid
     * @param g2 second grid
     */
    DifferenceNorm( const Grid2d<double>& g1, const Grid2d<double>& g2)
    {
        //find common grid
        Grid2d<double> gC(    g1.x0(), g1.x1(), g1.y0(), g1.y1(), 
                            std::min( g1.n(), g2.n()), 
                            lcm( g1.Nx(), g2.Nx()), 
                            lcm( g1.Ny(), g2.Ny()) );
        p1 = dg::create::interpolation( gC, g1);
        p2 = dg::create::interpolation( gC, g2);
        w2d = dg::create::weights( gC); v11 = w2d, v22 = w2d;
        wg1 = dg::create::weights( g1); 
        wg2 = dg::create::weights( g2); 
    }
    /**
     * @brief Compute difference of two vectors
     *
     * \f[ ||v_1 - v_2|| = \sqrt{ \int (v_1-v_2)^2 dV} \f]
     * @param v1
     * @param v2
     *
     * @return 
     */
    double operator()( const container& v1, const container& v2)
    {
        double f2, g2, fg;
        f2 = blas2::dot( wg1, v1);
        g2 = blas2::dot( wg2, v2);

        blas2::gemv( p1, v1, v11);
        blas2::gemv( p2, v2, v22);
        fg = blas2::dot( v11, w2d, v22);
        return sqrt( f2 - 2.*fg + g2);
    }

    /**
     * @brief Compute the sum of two vectors
     *
     * \f[ ||v_1 + v_2|| = \sqrt{ \int (v_1+v_2)^2 dV} \f]
     * @param v1
     * @param v2
     *
     * @return 
     */
    double sum( const container& v1, const container& v2)
    {
        double f2, g2, fg;
        f2 = blas2::dot( wg1, v1);
        g2 = blas2::dot( wg2, v2);

        blas2::gemv( p1, v1, v11);
        blas2::gemv( p2, v2, v22);
        fg = blas2::dot( v11, w2d, v22);
        return sqrt( f2 + 2.*fg + g2);
    }
  private:
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    typedef cusp::csr_matrix<int, double, MemorySpace> Matrix;

    container wg1, wg2, w2d;
    container v11, v22;
    Matrix p1, p2;
};



}//namespace dg
