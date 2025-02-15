#pragma once

#include <cassert>

#include "dg/backend/sparseblockmat.h"
#include "grid.h"
#include "functions.h"
#include "operator.h"
#include "weights.h"

/*! @file
  @brief Simple 1d derivatives
  */
namespace dg
{
namespace create
{
///@cond


/**
* @brief Create and assemble a host Matrix for the centered 1d single derivative
*
* The matrix isn't symmetric due to the normalisation T.
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size (used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> dx_symm(int n, int N, real_type h, bc bcx)
{
    constexpr int invalid_idx = EllSparseBlockMat<real_type>::invalid_index;

    SquareMatrix<real_type> l = create::lilj<real_type>(n);
    SquareMatrix<real_type> r = create::rirj<real_type>(n);
    SquareMatrix<real_type> lr = create::lirj<real_type>(n);
    SquareMatrix<real_type> rl = create::rilj<real_type>(n);
    SquareMatrix<real_type> d = create::pidxpj<real_type>(n);
    SquareMatrix<real_type> t = create::pipj_inv<real_type>(n);
    t *= 2./h;

    SquareMatrix< real_type> a = 1./2.*t*(d-d.transpose());
    //bcx = PER
    SquareMatrix<real_type> a_bound_right(a), a_bound_left(a);
    //left boundary
    if( bcx == DIR || bcx == DIR_NEU )
        a_bound_left += 0.5*t*l;
    else if (bcx == NEU || bcx == NEU_DIR)
        a_bound_left -= 0.5*t*l;
    //right boundary
    if( bcx == DIR || bcx == NEU_DIR)
        a_bound_right -= 0.5*t*r;
    else if( bcx == NEU || bcx == DIR_NEU)
        a_bound_right += 0.5*t*r;
    if( bcx == PER ) //periodic bc
        a_bound_left = a_bound_right = a;
    SquareMatrix<real_type> b = t*(1./2.*rl);
    SquareMatrix<real_type> bp = t*(-1./2.*lr); //pitfall: T*-m^T is NOT -(T*m)^T
    //transform to XSPACE
    SquareMatrix<real_type> backward=dg::DLT<real_type>::backward(n);
    SquareMatrix<real_type> forward=dg::DLT<real_type>::forward(n);
    a = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
    b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
    bp = backward*bp*forward;
    //assemble the matrix
    if( bcx != PER)
    {
        EllSparseBlockMat<real_type> A(N, N, 3, 5, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = bp(i,j);
            A.data[(1*n+i)*n+j] = a(i,j);
            A.data[(2*n+i)*n+j] = b(i,j);
            A.data[(3*n+i)*n+j] = a_bound_left(i,j);
            A.data[(4*n+i)*n+j] = a_bound_right(i,j);
        }
        A.data_idx[0*3+0] = 3; //a_bound_left
        A.cols_idx[0*3+0] = 0;
        A.data_idx[0*3+1] = 2; //b
        A.cols_idx[0*3+1] = 1;
        A.data_idx[0*3+2] = 2; //
        A.cols_idx[0*3+2] = invalid_idx; //prevent unnecessary data fetch
        for( int i=1; i<N-1; i++)
            for( int d=0; d<3; d++)
            {
                A.data_idx[i*3+d] = d; //bp, a, b
                A.cols_idx[i*3+d] = i+d-1;
            }
        A.data_idx[(N-1)*3+0] = 0; //bp
        A.cols_idx[(N-1)*3+0] = N-2;
        A.data_idx[(N-1)*3+1] = 4; //a_bound_right
        A.cols_idx[(N-1)*3+1] = N-1;
        A.data_idx[(N-1)*3+2] = 4; //0
        A.cols_idx[(N-1)*3+2] = invalid_idx; //prevent unnecessary data fetch
        return A;

    }
    else //periodic
    {
        EllSparseBlockMat<real_type> A(N, N, 3, 3, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = bp(i,j);
            A.data[(1*n+i)*n+j] = a(i,j);
            A.data[(2*n+i)*n+j] = b(i,j);
        }
        for( int i=0; i<N; i++)
            for( int d=0; d<3; d++)
            {
                A.data_idx[i*3+d] = d; //bp, a, b
                A.cols_idx[i*3+d] = (i+d-1+N)%N;
            }
        return A;
    }
};

/**
* @brief Create and assemble a host Matrix for the forward 1d single derivative
*
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> dx_plus( int n, int N, real_type h, bc bcx )
{
    constexpr int invalid_idx = EllSparseBlockMat<real_type>::invalid_index;

    SquareMatrix<real_type> l = create::lilj<real_type>(n);
    SquareMatrix<real_type> r = create::rirj<real_type>(n);
    SquareMatrix<real_type> lr = create::lirj<real_type>(n);
    SquareMatrix<real_type> rl = create::rilj<real_type>(n);
    SquareMatrix<real_type> d = create::pidxpj<real_type>(n);
    SquareMatrix<real_type> t = create::pipj_inv<real_type>(n);
    t *= 2./h;
    SquareMatrix<real_type>  a = t*(-l-d.transpose());
    //if( dir == backward) a = -a.transpose();
    SquareMatrix<real_type> a_bound_left = a; //PER, NEU and NEU_DIR
    SquareMatrix<real_type> a_bound_right = a; //PER, DIR and NEU_DIR
    if( bcx == dg::DIR || bcx == dg::DIR_NEU)
        a_bound_left = t*(-d.transpose());
    if( bcx == dg::NEU || bcx == dg::DIR_NEU)
        a_bound_right = t*(d);
    SquareMatrix<real_type> b = t*rl;
    //transform to XSPACE
    SquareMatrix<real_type> backward=dg::DLT<real_type>::backward(n);
    SquareMatrix<real_type> forward=dg::DLT<real_type>::forward(n);
    a = backward*a*forward, a_bound_left = backward*a_bound_left*forward;
    b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
    //assemble the matrix
    if( bcx != PER)
    {
        EllSparseBlockMat<real_type> A(N, N, 2, 4, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = a(i,j);
            A.data[(1*n+i)*n+j] = b(i,j);
            A.data[(2*n+i)*n+j] = a_bound_left(i,j);
            A.data[(3*n+i)*n+j] = a_bound_right(i,j);
        }
        A.data_idx[0*2+0] = 2; //a_bound_left
        A.cols_idx[0*2+0] = 0;
        A.data_idx[0*2+1] = 1; //b
        A.cols_idx[0*2+1] = 1;
        for( int i=1; i<N-1; i++) //a
            for( int d=0; d<2; d++)
            {
                A.data_idx[i*2+d] = d; //a, b
                A.cols_idx[i*2+d] = i+d; //0,1
            }
        A.data_idx[(N-1)*2+0] = 3; //a_bound_right
        A.cols_idx[(N-1)*2+0] = N-1;
        A.data_idx[(N-1)*2+1] = 3; //0
        A.cols_idx[(N-1)*2+1] = invalid_idx; //prevent unnecessary data fetch
        return A;

    }
    else //periodic
    {
        EllSparseBlockMat<real_type> A(N, N, 2, 2, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = a(i,j);
            A.data[(1*n+i)*n+j] = b(i,j);
        }
        for( int i=0; i<N; i++)
            for( int d=0; d<2; d++)
            {
                A.data_idx[i*2+d] = d; //a, b
                A.cols_idx[i*2+d] = (i+d+N)%N;
            }
        return A;
    }
};

/**
* @brief Create and assemble a host Matrix for the backward 1d single derivative
*
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> dx_minus( int n, int N, real_type h, bc bcx )
{
    constexpr int invalid_idx = EllSparseBlockMat<real_type>::invalid_index;
    SquareMatrix<real_type> l = create::lilj<real_type>(n);
    SquareMatrix<real_type> r = create::rirj<real_type>(n);
    SquareMatrix<real_type> lr = create::lirj<real_type>(n);
    SquareMatrix<real_type> rl = create::rilj<real_type>(n);
    SquareMatrix<real_type> d = create::pidxpj<real_type>(n);
    SquareMatrix<real_type> t = create::pipj_inv<real_type>(n);
    t *= 2./h;
    SquareMatrix<real_type>  a = t*(l+d);
    //if( dir == backward) a = -a.transpose();
    SquareMatrix<real_type> a_bound_right = a; //PER, NEU and DIR_NEU
    SquareMatrix<real_type> a_bound_left = a; //PER, DIR and DIR_NEU
    if( bcx == dg::DIR || bcx == dg::NEU_DIR)
        a_bound_right = t*(-d.transpose());
    if( bcx == dg::NEU || bcx == dg::NEU_DIR)
        a_bound_left = t*d;
    SquareMatrix<real_type> bp = -t*lr;
    //transform to XSPACE
    SquareMatrix<real_type> backward=dg::DLT<real_type>::backward(n);
    SquareMatrix<real_type> forward=dg::DLT<real_type>::forward(n);
    a  = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
    bp = backward*bp*forward, a_bound_right = backward*a_bound_right*forward;

    //assemble the matrix
    if(bcx != dg::PER)
    {
        EllSparseBlockMat<real_type> A(N, N, 2, 4, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = bp(i,j);
            A.data[(1*n+i)*n+j] = a(i,j);
            A.data[(2*n+i)*n+j] = a_bound_left(i,j);
            A.data[(3*n+i)*n+j] = a_bound_right(i,j);
        }
        A.data_idx[0*2+0] = 2; //a_bound_left
        A.cols_idx[0*2+0] = 0;
        A.data_idx[0*2+1] = 2; //0
        A.cols_idx[0*2+1] = invalid_idx; //prevent data fetch
        for( int i=1; i<N-1; i++) //a
            for( int d=0; d<2; d++)
            {
                A.data_idx[i*2+d] = d; //bp, a
                A.cols_idx[i*2+d] = i+d-1;
            }
        A.data_idx[(N-1)*2+0] = 0; //bp
        A.cols_idx[(N-1)*2+0] = N-2;
        A.data_idx[(N-1)*2+1] = 3; //a_bound_right
        A.cols_idx[(N-1)*2+1] = N-1;
        return A;

    }
    else //periodic
    {
        EllSparseBlockMat<real_type> A(N, N, 2, 2, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = bp(i,j);
            A.data[(1*n+i)*n+j] = a(i,j);
        }
        for( int i=0; i<N; i++)
            for( int d=0; d<2; d++)
            {
                A.data_idx[i*2+d] = d; //bp, a
                A.cols_idx[i*2+d] = (i+d-1+N)%N;  //-1, 0
            }
        return A;
    }
};

/**
* @brief Create and assemble a host Matrix for the jump terms in 1d
*
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> jump( int n, int N, real_type h, bc bcx)
{
    constexpr int invalid_idx = EllSparseBlockMat<real_type>::invalid_index;
    SquareMatrix<real_type> l = create::lilj<real_type>(n);
    SquareMatrix<real_type> r = create::rirj<real_type>(n);
    SquareMatrix<real_type> lr = create::lirj<real_type>(n);
    SquareMatrix<real_type> rl = create::rilj<real_type>(n);
    SquareMatrix<real_type> a = l+r;
    SquareMatrix<real_type> a_bound_left = a;//DIR and PER
    if( bcx == NEU || bcx == NEU_DIR)
        a_bound_left = r;
    SquareMatrix<real_type> a_bound_right = a; //DIR and PER
    if( bcx == NEU || bcx == DIR_NEU)
        a_bound_right = l;
    SquareMatrix<real_type> b = -rl;
    SquareMatrix<real_type> bp = -lr;
    //transform to XSPACE
    SquareMatrix<real_type> t = create::pipj_inv<real_type>(n);
    t *= 2./h;
    SquareMatrix<real_type> backward=dg::DLT<real_type>::backward(n);
    SquareMatrix<real_type> forward=dg::DLT<real_type>::forward(n);
    a = backward*t*a*forward, a_bound_left  = backward*t*a_bound_left*forward;
    b = backward*t*b*forward, a_bound_right = backward*t*a_bound_right*forward;
    bp = backward*t*bp*forward;
    //assemble the matrix
    if(bcx != dg::PER)
    {
        EllSparseBlockMat<real_type> A(N, N, 3, 5, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = bp(i,j);
            A.data[(1*n+i)*n+j] = a(i,j);
            A.data[(2*n+i)*n+j] = b(i,j);
            A.data[(3*n+i)*n+j] = a_bound_left(i,j);
            A.data[(4*n+i)*n+j] = a_bound_right(i,j);
        }
        A.data_idx[0*3+0] = 3; //a_bound_left
        A.cols_idx[0*3+0] = 0;
        A.data_idx[0*3+1] = 2; //b
        A.cols_idx[0*3+1] = 1;
        A.data_idx[0*3+2] = 2; //0
        A.cols_idx[0*3+2] = invalid_idx; //prevent unnecessary data fetch
        for( int i=1; i<N-1; i++) //a
            for( int d=0; d<3; d++)
            {
                A.data_idx[i*3+d] = d; //bp, a, b
                A.cols_idx[i*3+d] = i+d-1;
            }
        A.data_idx[(N-1)*3+0] = 0; //bp
        A.cols_idx[(N-1)*3+0] = N-2;
        A.data_idx[(N-1)*3+1] = 4; //a_bound_right
        A.cols_idx[(N-1)*3+1] = N-1;
        A.data_idx[(N-1)*3+2] = 4; //0
        A.cols_idx[(N-1)*3+2] = invalid_idx; //prevent unnecessary data fetch
        return A;

    }
    else //periodic
    {
        EllSparseBlockMat<real_type> A(N, N, 3, 3, n);
        for( int i=0; i<n; i++)
        for( int j=0; j<n; j++)
        {
            A.data[(0*n+i)*n+j] = bp(i,j);
            A.data[(1*n+i)*n+j] = a(i,j);
            A.data[(2*n+i)*n+j] = b(i,j);
        }
        for( int i=0; i<N; i++)
            for( int d=0; d<3; d++)
            {
                A.data_idx[i*3+d] = d; //bp, a, b
                A.cols_idx[i*3+d] = (i+d-1+N)%N;
            }
        return A;
    }
};

/**
* @brief Create and assemble a host Matrix for normed derivative in 1d
*
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
* @param dir The direction of the first derivative
*
* @return Host Matrix
*/
template<class real_type>
EllSparseBlockMat<real_type> dx_normed( int n, int N, real_type h, bc bcx, direction dir )
{
    if( dir == centered)
        return create::dx_symm(n, N, h, bcx);
    else if (dir == forward)
        return create::dx_plus(n, N, h, bcx);
    else if (dir == backward)
        return create::dx_minus(n, N, h, bcx);
    return EllSparseBlockMat<real_type>();
}
///@endcond

} //namespace create
} //namespace dg

