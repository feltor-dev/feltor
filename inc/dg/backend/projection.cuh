#pragma once
#include <vector>
#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include "grid.h"
#include "interpolation.cuh"

/*!@file 
  
  contains creation of projection matrices
 */
namespace dg{
///@addtogroup interpolation
///@{

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
 @attention Projection only works if the number of cells in the
 fine grid are multiple of the number of cells in the coarse grid
 */
cusp::coo_matrix< int, double, cusp::host_memory> projection( const Grid1d& g_new, const Grid1d& g_old)
{
    if( g_old.N() % g_new.N() != 0) std::cerr << "ATTENTION: you project between incompatible grids!!\n";
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
 * @brief Create the transpose of the interpolation matrix from new to old
 *
 * @param g_new The new grid 
 * @param g_old The old grid
 *
 * @return transposed interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const Grid2d& g_new, const Grid2d& g_old)
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
 * @return transposed interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 @attention Projection only works if the number of cells in the
 fine grid are multiple of the number of cells in the coarse grid
 */
cusp::coo_matrix< int, double, cusp::host_memory> projection_( const Grid2d& g_new, const Grid2d& g_old)
{
    if( g_old.Nx() % g_new.Nx() != 0) std::cerr << "ATTENTION: you project between incompatible grids in x!!\n";
    if( g_old.Ny() % g_new.Ny() != 0) std::cerr << "ATTENTION: you project between incompatible grids in y!!\n";
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
 * @brief Create the transpose of the interpolation matrix from new to old
 *
 * @param g_new The new grid 
 * @param g_old The old grid
 *
 * @return transposed interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 */
cusp::coo_matrix<int, double, cusp::host_memory> interpolationT( const Grid3d& g_new, const Grid3d& g_old)
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
 * @return transposed interpolation matrix
 * @note The boundaries of the old grid must lie within the boundaries of the new grid
 @attention Projection only works if the number of cells in the
 fine grid are multiple of the number of cells in the coarse grid
 */
cusp::coo_matrix< int, double, cusp::host_memory> projection_( const Grid3d& g_new, const Grid3d& g_old)
{
    if( g_old.Nx() % g_new.Nx() != 0) std::cerr << "ATTENTION: you project between incompatible grids in x!!\n";
    if( g_old.Ny() % g_new.Ny() != 0) std::cerr << "ATTENTION: you project between incompatible grids in y!!\n";
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

///@}

}//namespace create




}//namespace dg
