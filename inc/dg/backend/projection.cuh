#pragma once
#include <vector>
#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include "grid.h"
#include "interpolation.cuh"

/*!@file 
  
  @brief contains creation of projection matrices
 */
namespace dg{
///@addtogroup interpolation
///@{

/**
 * @brief Greatest common divisor
 *
 * @param a First number
 * @param b Second number
 *
 * @return greatest common divisor
 * @ingroup misc
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
 * @ingroup misc
 */
unsigned lcm( unsigned a, unsigned b)
{
    unsigned g = gcd( a,b);
    return a/g*b;
}

namespace create{

/**
 * @brief Create the transpose of the interpolation matrix from new to old
 *
 * @param g_new The new grid 
 * @param g_old The old grid
 *
 * @return transposed interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const Grid1d& g_new, const Grid1d& g_old)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_old, g_new), A;
    cusp::transpose( temp, A);
    return A;
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const aTopology2d& g_new, const aTopology2d& g_old)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_old, g_new), A;
    cusp::transpose( temp, A);
    return A;
}
///@copydoc interpolationT(const Grid1d&,const Grid1d&)
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const aTopology3d& g_new, const aTopology3d& g_old)
{
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_old, g_new), A;
    cusp::transpose( temp, A);
    return A;
}

/**
 * @brief Create a projection between two grids
 *
 * The projection matrix is the adjoint of the interpolation matrix
 * This matrix can be applied to vectors defined on the old (fine) grid to obtain
 * its values on the new (coarse) grid. 
 * If the fine grid is a multiple of the coarse grid, the integral value
 of the projected vector will be conserved and the difference in the L2 norm 
 between old and new vector small. 
 * 
 * @param g_new The new (coarse) grid 
 * @param g_old The old (fine) grid
 *
 * @return Projection matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 * @note also check the transformation matrix, which is the more general solution
 @attention Projection only works if the number of cells in the
 fine grid are multiple of the number of cells in the coarse grid
 */
cusp::coo_matrix< int, double, cusp::host_memory> projection( const Grid1d& g_new, const Grid1d& g_old)
{
    if( g_old.N() % g_new.N() != 0) std::cerr << "ATTENTION: you project between incompatible grids!! old N: "<<g_old.N()<<" new N: "<<g_new.N()<<"\n";
    //form the adjoint
    thrust::host_vector<double> w_f = dg::create::weights( g_old);
    thrust::host_vector<double> v_c = dg::create::inv_weights( g_new );
    cusp::coo_matrix<int, double, cusp::host_memory> Wf( w_f.size(), w_f.size(), w_f.size());
    cusp::coo_matrix<int, double, cusp::host_memory> Vc( v_c.size(), v_c.size(), v_c.size());
    for( int i =0; i<(int)w_f.size(); i++)
    {
        Wf.row_indices[i] = Wf.column_indices[i] = i;
        Wf.values[i] = w_f[i];
    }
    for( int i =0; i<(int)v_c.size(); i++)
    {
        Vc.row_indices[i] = Vc.column_indices[i] = i;
        Vc.values[i] = v_c[i];
    }
    cusp::coo_matrix<int, double, cusp::host_memory> A = interpolationT( g_new, g_old), temp;
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}


///@copydoc projection(const Grid1d&,const Grid1d&)
cusp::coo_matrix< int, double, cusp::host_memory> projection( const aTopology2d& g_new, const aTopology2d& g_old)
{
    if( g_old.Nx() % g_new.Nx() != 0) std::cerr << "ATTENTION: you project between incompatible grids in x!! old N: "<<g_old.Nx()<<" new N: "<<g_new.Nx()<<"\n";
    if( g_old.Ny() % g_new.Ny() != 0) std::cerr << "ATTENTION: you project between incompatible grids in y!! old N: "<<g_old.Ny()<<" new N: "<<g_new.Ny()<<"\n";
    //form the adjoint
    thrust::host_vector<double> w_f = dg::create::weights( g_old);
    thrust::host_vector<double> v_c = dg::create::inv_weights( g_new );
    cusp::coo_matrix<int, double, cusp::host_memory> Wf( w_f.size(), w_f.size(), w_f.size());
    cusp::coo_matrix<int, double, cusp::host_memory> Vc( v_c.size(), v_c.size(), v_c.size());
    for( int i =0; i<(int)w_f.size(); i++)
    {
        Wf.row_indices[i] = Wf.column_indices[i] = i;
        Wf.values[i] = w_f[i];
    }
    for( int i =0; i<(int)v_c.size(); i++)
    {
        Vc.row_indices[i] = Vc.column_indices[i] = i;
        Vc.values[i] = v_c[i];
    }
    cusp::coo_matrix<int, double, cusp::host_memory> A = interpolationT( g_new, g_old), temp;
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}

///@copydoc projection(const Grid1d&,const Grid1d&)
cusp::coo_matrix< int, double, cusp::host_memory> projection( const aTopology3d& g_new, const aTopology3d& g_old)
{
    if( g_old.Nx() % g_new.Nx() != 0) std::cerr << "ATTENTION: you project between incompatible grids in x!! old N: "<<g_old.Nx()<<" new N: "<<g_new.Nx()<<"\n";
    if( g_old.Ny() % g_new.Ny() != 0) std::cerr << "ATTENTION: you project between incompatible grids in y!! old N: "<<g_old.Ny()<<" new N: "<<g_new.Ny()<<"\n";
    //form the adjoint
    thrust::host_vector<double> w_f = dg::create::weights( g_old);
    thrust::host_vector<double> v_c = dg::create::inv_weights( g_new );
    cusp::coo_matrix<int, double, cusp::host_memory> Wf( w_f.size(), w_f.size(), w_f.size());
    cusp::coo_matrix<int, double, cusp::host_memory> Vc( v_c.size(), v_c.size(), v_c.size());
    for( int i =0; i<(int)w_f.size(); i++)
    {
        Wf.row_indices[i] = Wf.column_indices[i] = i;
        Wf.values[i] = w_f[i];
    }
    for( int i =0; i<(int)v_c.size(); i++)
    {
        Vc.row_indices[i] = Vc.column_indices[i] = i;
        Vc.values[i] = v_c[i];
    }
    cusp::coo_matrix<int, double, cusp::host_memory> A = interpolationT( g_new, g_old), temp;
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}

/**
 * @brief Create a transformation matrix between two grids
 *
 * The transformation matrix is probably the most correct way of 
 transforming dG vectors between any two grids of different resolution. 
 It first finds the least common multiple grid (lcm) of the old and the new grid. Then
 it interpolates the values to the lcm grid and finally projects them back to
 the new grid. In total we have
 \f[
 \mathcal T = P Q 
 \f] 
 where \f$ Q\f$ is the interpolation matrix and \f$ P \f$ the projection. If either new or
 old grid is already the lcm grid this function reduces to the interpolation/projection function. 
 * 
 * @param g_new The new grid 
 * @param g_old The old grid
 *
 * @return transformation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 * @note If the grid are very incompatible the matrix-matrix multiplication can take a while
 */
cusp::coo_matrix< int, double, cusp::host_memory> transformation( const aTopology3d& g_new, const aTopology3d& g_old)
{
    Grid3d g_lcm(g_new.x0(), g_new.x1(), g_new.y0(), g_new.y1(), g_new.z0(), g_new.z1(), 
                 lcm(g_new.n(), g_old.n()), lcm(g_new.Nx(), g_old.Nx()), lcm(g_new.Ny(), g_old.Ny()), 
                 lcm(g_new.Nz(), g_old.Nz()));
    cusp::coo_matrix< int, double, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
    cusp::coo_matrix< int, double, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
    cusp::multiply( P, Q, Y);
    Y.sort_by_row_and_column();
    return Y;
}

///@copydoc transformation(const aTopology3d&,const aTopology3d&)
cusp::coo_matrix< int, double, cusp::host_memory> transformation( const aTopology2d& g_new, const aTopology2d& g_old)
{
    Grid2d g_lcm(g_new.x0(), g_new.x1(), g_new.y0(), g_new.y1(), 
                 lcm(g_new.n(), g_old.n()), lcm(g_new.Nx(), g_old.Nx()), lcm(g_new.Ny(), g_old.Ny()));
    cusp::coo_matrix< int, double, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
    cusp::coo_matrix< int, double, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
    cusp::multiply( P, Q, Y);
    Y.sort_by_row_and_column();
    return Y;
}
///@copydoc transformation(const aTopology3d&,const aTopology3d&)
cusp::coo_matrix< int, double, cusp::host_memory> transformation( const Grid1d& g_new, const Grid1d& g_old)
{
    Grid1d g_lcm(g_new.x0(), g_new.x1(), lcm(g_new.n(), g_old.n()), lcm(g_new.N(), g_old.N()));
    cusp::coo_matrix< int, double, cusp::host_memory> Q = create::interpolation( g_lcm, g_old);
    cusp::coo_matrix< int, double, cusp::host_memory> P = create::projection( g_new, g_lcm), Y;
    cusp::multiply( P, Q, Y);
    Y.sort_by_row_and_column();
    return Y;
}
///@}

/*
///@deprecated
cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::aRefinedGrid2d& g_fine)
{
    //form the adjoint
    thrust::host_vector<double> w_f = dg::create::weights( g_fine);
    thrust::host_vector<double> v_c = dg::create::inv_weights( g_fine.associated() );
    cusp::coo_matrix<int, double, cusp::host_memory> Wf( w_f.size(), w_f.size(), w_f.size());
    cusp::coo_matrix<int, double, cusp::host_memory> Vc( v_c.size(), v_c.size(), v_c.size());
    for( int i =0; i<(int)w_f.size(); i++)
    {
        Wf.row_indices[i] = Wf.column_indices[i] = i;
        Wf.values[i] = w_f[i]/g_fine.weightsX()[i]/g_fine.weightsY()[i];
    }
    for( int i =0; i<(int)v_c.size(); i++)
    {
        Vc.row_indices[i] = Vc.column_indices[i] = i;
        Vc.values[i] = v_c[i];
    }
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}



///@deprecated
cusp::coo_matrix<int, double, cusp::host_memory> projection( const dg::aRefinedGrid3d& g_fine)
{
    //form the adjoint
    thrust::host_vector<double> w_f = dg::create::weights( g_fine);
    thrust::host_vector<double> v_c = dg::create::inv_weights( g_fine.associated() );
    cusp::coo_matrix<int, double, cusp::host_memory> Wf( w_f.size(), w_f.size(), w_f.size());
    cusp::coo_matrix<int, double, cusp::host_memory> Vc( v_c.size(), v_c.size(), v_c.size());
    for( int i =0; i<(int)w_f.size(); i++)
    {
        Wf.row_indices[i] = Wf.column_indices[i] = i;
        Wf.values[i] = w_f[i]/g_fine.weightsX()[i]/g_fine.weightsY()[i];
    }
    for( int i =0; i<(int)v_c.size(); i++)
    {
        Vc.row_indices[i] = Vc.column_indices[i] = i;
        Vc.values[i] = v_c[i];
    }
    cusp::coo_matrix<int, double, cusp::host_memory> temp = interpolation( g_fine), A;
    cusp::transpose( temp, A);
    cusp::multiply( A, Wf, temp);
    cusp::multiply( Vc, temp, A);
    A.sort_by_row_and_column();
    return A;
}
*/


}//namespace create




}//namespace dg
