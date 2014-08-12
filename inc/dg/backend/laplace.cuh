#pragma once

#include <cusp/coo_matrix.h>
#include <cusp/transpose.h>
#include <cusp/elementwise.h>

#include "grid.h"
#include "functions.h"
#include "operator.h"
#include "creation.cuh"
#include "dx.cuh"
#include "operator_tensor.cuh"

/*! @file 
  
  1d laplacians
  */

namespace dg
{

namespace create{
///@cond

/**
 * @brief Function for the creation of a 1d laplacian in LSPACE
 *
 * @ingroup highlevel
 * @tparam T value_type
 * @param g The grid on which to create the laplacian (including boundary condition)
 * @param no use normed if you want to compute e.g. diffusive terms
            use not_normed if you want to solve symmetric matrix equations (T is missing)
 *
 * @return Host Matrix in coordinate form
 */
template< class value_type>
cusp::coo_matrix<int, value_type, cusp::host_memory> laplace1d( const Grid1d<value_type>& g, norm no = not_normed, direction dir = forward )
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> HMatrix;
    HMatrix S = dg::tensor( g.N(), dg::create::pipj( g.n())); 
    cusp::blas::scal( S.values, g.h()/2.);
    HMatrix T = dg::tensor( g.N(), dg::create::pipj_inv( g.n())); 
    cusp::blas::scal( T.values, 2./g.h());
    HMatrix right;
    if( dir == forward)
        right = create::dx_plus_mt( g.n(), g.N(), g.h(), g.bcx());
    else if ( dir == backward) 
        right = create::dx_minus_mt( g.n(), g.N(), g.h(), g.bcx());
    else //dir == symmetric
    {
        if( g.bcx() == PER || g.bcx() == NEU_DIR)
            return laplace1d( g, no, forward); //per is symmetric, NEU_DIR cannot be
        if( g.bcx() == DIR_NEU)
            return laplace1d( g, no, backward);//cannot be symmetric
        HMatrix laplus = laplace1d( g, no, forward); //recursive call
        HMatrix laminus = laplace1d( g, no, backward);
        HMatrix laplace;
        
        cusp::add( laplus, laminus, laplace);//only add values??
        cusp::blas::scal( laplace.values, 0.5);
        return laplace;
    }
    HMatrix left, temp;
    cusp::transpose( right, left);
    cusp::multiply( left, S, temp);

    HMatrix laplace_oJ, laplace;
    cusp::multiply( temp, right, laplace_oJ);
    if( g.n() == 1 && g.bcx() == dg::PER)
    {
        if( no == normed) 
        {
            cusp::multiply( T, laplace_oJ, laplace);
            return laplace;
        }
        return laplace_oJ;
    }
    HMatrix J = dg::create::jump_ot<value_type>( g.n(), g.N(), g.bcx());
    cusp::add( laplace_oJ, J, laplace);
    laplace.sort_by_row_and_column();
    if( no == normed) 
    {
        cusp::multiply( T, laplace, laplace_oJ);
        return laplace_oJ;
    }
    return laplace;
}
///@endcond
} //namespace create

} //namespace dg
