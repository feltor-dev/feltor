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
namespace detail{

bc inverse( bc bound)
{
    if( bound==DIR)return NEU;
    if( bound==NEU)return DIR;
    if( bound==DIR_NEU)return NEU_DIR;
    if( bound==NEU_DIR)return DIR_NEU;
    return PER;
}
}//namespace detail

/**
 * @brief Function for the creation of a 1d laplacian in XSPACE
 *
 * @ingroup highlevel
 * @tparam T value_type
 * @param g The grid on which to create the laplacian (including boundary condition)
 * @param no use normed if you want to compute e.g. diffusive terms
            use not_normed if you want to solve symmetric matrix equations (V is missing)
 *
 * @return Host Matrix in coordinate form
 */
template< class value_type>
cusp::coo_matrix<int, value_type, cusp::host_memory> laplace1d( const Grid1d<value_type>& g, bc bcx, norm no = not_normed, direction dir = forward )
{
    typedef cusp::coo_matrix<int, value_type, cusp::host_memory> HMatrix;

    HMatrix W = dg::tensor( g.N(), dg::create::weights( g.n())); 
    cusp::blas::scal( W.values, g.h()/2.);
    HMatrix left, right;
    if( dir == forward)
    {
        right = create::dx_plus_normed( g.n(), g.N(), g.h(), bcx);
        left  = create::dx_minus_normed( g.n(), g.N(), g.h(), detail::inverse( bcx));
    }
    else if ( dir == backward) 
    {
        right = create::dx_minus_normed( g.n(), g.N(), g.h(), bcx);
        left  = create::dx_plus_normed( g.n(), g.N(), g.h(), detail::inverse( bcx));
    }
    else //dir == symmetric
    {
        if( bcx == PER || bcx == NEU_DIR)
            return laplace1d( g, bcx, no, forward); //per is symmetric, NEU_DIR cannot be
        if( bcx == DIR_NEU)
            return laplace1d( g, bcx, no, backward);//cannot be symmetric
        HMatrix laplus = laplace1d( g, bcx, no, forward); //recursive call
        HMatrix laminus = laplace1d( g, bcx, no, backward);
        HMatrix laplace;
        
        cusp::add( laplus, laminus, laplace);//only add values??
        cusp::blas::scal( laplace.values, 0.5);
        return laplace;
    }

    HMatrix laplace_oJ, laplace;
    cusp::multiply( left, right, laplace_oJ);
    cusp::blas::scal( laplace_oJ.values, -1.);
    //make 2nd order for n = 1  and periodic BC
    if( g.n() == 1 && bcx == dg::PER)
    {
        if( no == not_normed) 
        {
            cusp::multiply( W, laplace_oJ, laplace);
            return laplace;
        }
        return laplace_oJ;
    }
    HMatrix J = dg::create::jump_normed<value_type>( g.n(), g.N(), g.h(), bcx);
    cusp::add( laplace_oJ, J, laplace);
    laplace.sort_by_row_and_column();
    if( no == not_normed) 
    {
        cusp::multiply( W, laplace, laplace_oJ);
        return laplace_oJ;
    }
    return laplace;
}
///@endcond
} //namespace create

} //namespace dg
