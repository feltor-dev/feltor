#pragma once

#include <cassert>

#include "grid.h"
#include "functions.h"
#include "operator.h"
#include "weights.cuh"
#include "sparseblockmat.h"

/*! @file 
  
  Simple 1d derivatives
  */
namespace dg
{
namespace create
{
///@addtogroup lowlevel
///@{


/**
* @brief Create and assemble a cusp Matrix for the symmetric 1d single derivative in XSPACE
*
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* The matrix isn't symmetric due to the normalisation T.
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size (used to compute normalisation)
* @param bcx boundary condition 
*
* @return Host Matrix in coordinate form 
*/
SparseBlockMat dx_symm(unsigned n, unsigned N, double h, bc bcx)
{

    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./h;

    Operator< double> a = 1./2.*t*(d-d.transpose());
    //bcx = PER
    Operator<double> a_bound_right(a), a_bound_left(a);
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
    Operator<double> b = t*(1./2.*rl);
    Operator<double> bp = t*(-1./2.*lr); //pitfall: T*-m^T is NOT -(T*m)^T
    //transform to XSPACE
    Grid1d<double> g( 0,1, n, N);
    Operator<double> backward=g.dlt().backward();
    Operator<double> forward=g.dlt().forward();
    a = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
    b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
    bp = backward*bp*forward;
    //assemble the matrix
    SparseBlockMat A(3, N, N);
    if( bcx != PER)
    {
    A.diag[0].resize( 3*n*n), A.diag[1].resize(n*n), A.diag[2].resize(n*n);
    A.row[0].resize(   N), A.row[1].resize(N-1), A.row[2].resize(N-1);
    A.col[0].resize(   N), A.col[1].resize(N-1), A.col[2].resize(N-1);
    A.block[0].resize(   N), A.block[1].resize(N-1), A.block[2].resize(N-1);
    A.n[0] = A.n[1] = A.n[2] = n;
    A.num_blocks[0] = N, A.num_blocks[1] = A.num_blocks[2] = N-1;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][(0*n+i)*n+j] = a_bound_left(i,j);
        A.diag[0][(1*n+i)*n+j] = a(i,j);
        A.diag[0][(2*n+i)*n+j] = a_bound_right(i,j);
        A.diag[1][(0*n+i)*n+j] = b(i,j);
        A.diag[2][(0*n+i)*n+j] = bp(i,j);
    }
    A.block[0][0] = 0; //a_bound_left
    A.row[0][0] = 0;
    A.col[0][0] = 0;
    for( unsigned i=1; i<N-1; i++) //a
    {
        A.block[0][i] = 1;
        A.row[0][i] = i;
        A.col[0][i] = i;
    }
    A.block[0][N-1] = 2; //a_bound_right
    A.row[0][N-1] = N-1;
    A.col[0][N-1] = N-1;
    for( unsigned i=0; i<N-1; i++)
    {
        A.block[1][i] = 0; //b
        A.row[1][i] = i;
        A.col[1][i] = i+1;
        A.block[2][i] = 0; //bp
        A.row[2][i] = i+1;
        A.col[2][i] = i;
    }

    }
    else //periodic
    {
    A.diag[0].resize( n*n), A.diag[1].resize(n*n), A.diag[2].resize(n*n);
    A.row[0].resize(   N), A.row[1].resize(N), A.row[2].resize(N);
    A.col[0].resize(   N), A.col[1].resize(N), A.col[2].resize(N);
    A.block[0].resize(   N), A.block[1].resize(N), A.block[2].resize(N);
    A.n[0] = A.n[1] = A.n[2] = n;
    A.num_blocks[0] = A.num_blocks[1] = A.num_blocks[2] = N;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][i*n+j] = a(i,j);
        A.diag[1][i*n+j] = b(i,j);
        A.diag[2][i*n+j] = bp(i,j);
    }
    for( unsigned i=0; i<N; i++) 
    {
        A.block[0][i] = 0; //a
        A.row[0][i] = i;
        A.col[0][i] = i;
        A.block[1][i] = 0; //b
        A.row[1][i] = i;
        A.col[1][i] = (i+1)%N;
        A.block[2][i] = 0; //bp
        A.row[2][i] = i;
        A.col[2][i] = (i-1+N)%N; //make positive number
    }
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the 1d single forward derivative in XSPACE
*
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* Neumann BC means inner value for flux
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix in coordinate form 
*/
SparseBlockMat dx_plus( unsigned n, unsigned N, double h, bc bcx )
{

    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./h;
    Operator<double>  a = t*(-l-d.transpose());
    //if( dir == backward) a = -a.transpose();
    Operator<double> a_bound_left = a; //PER, NEU and NEU_DIR
    Operator<double> a_bound_right = a; //PER, DIR and NEU_DIR
    if( bcx == dg::DIR || bcx == dg::DIR_NEU) 
        a_bound_left = t*(-d.transpose());
    if( bcx == dg::NEU || bcx == dg::DIR_NEU)
        a_bound_right = t*(d);
    Operator<double> b = t*rl;
    //transform to XSPACE
    Grid1d<double> g( 0,1, n, N);
    Operator<double> backward=g.dlt().backward();
    Operator<double> forward=g.dlt().forward();
    a = backward*a*forward, a_bound_left = backward*a_bound_left*forward;
    b = backward*b*forward, a_bound_right = backward*a_bound_right*forward;
    //assemble the matrix
    SparseBlockMat A(2, N, N);
    if( bcx != PER)
    {
    A.diag[0].resize( 3*n*n), A.diag[1].resize(n*n); 
    A.row[0].resize(  N), A.row[1].resize( N-1);
    A.col[0].resize(  N), A.col[1].resize( N-1);
    A.block[0].resize(  N), A.block[1].resize( N-1);
    A.n[0] = A.n[1] = n;
    A.num_blocks[0] = N; A.num_blocks[1] = N-1;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][(0*n+i)*n+j] = a_bound_left(i,j);
        A.diag[0][(1*n+i)*n+j] = a(i,j);
        A.diag[0][(2*n+i)*n+j] = a_bound_right(i,j);
        A.diag[1][(0*n+i)*n+j] = b(i,j);
    }
    A.block[0][0] = 0; //a_bound_left
    A.row[0][0] = 0;
    A.col[0][0] = 0;
    for( unsigned i=1; i<N-1; i++) //a
    {
        A.block[0][0*N+i] = 1;
        A.row[0][0*N+i] = i;
        A.col[0][0*N+i] = i;
    }
    A.block[0][N-1] = 2; //a_bound_right
    A.row[0][N-1] = N-1;
    A.col[0][N-1] = N-1;
    for( unsigned i=0; i<N-1; i++) //b
    {
        A.block[1][i] = 0;
        A.row[1][i] = i;
        A.col[1][i] = i+1;
    }

    }
    else //periodic
    {
    A.diag[0].resize( n*n), A.diag[1].resize(n*n);
    A.row[0].resize(   N), A.row[1].resize(N);
    A.col[0].resize(   N), A.col[1].resize(N);
    A.block[0].resize(   N), A.block[1].resize(N);
    A.n[0] = A.n[1] = n;
    A.num_blocks[0] = A.num_blocks[1] = N;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][i*n+j] = a(i,j);
        A.diag[1][i*n+j] = b(i,j);
    }
    for( unsigned i=0; i<N; i++) 
    {
        A.block[0][i] = 0; //a
        A.row[0][i] = i;
        A.col[0][i] = i;
        A.block[1][i] = 0; //b
        A.row[1][i] = i;
        A.col[1][i] = (i+1)%N;
    }
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the 1d single backward derivative in XSPACE
*
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* Neumann BC means inner value for flux
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix in coordinate form 
*/
SparseBlockMat dx_minus( unsigned n, unsigned N, double h, bc bcx )
{
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> d = create::pidxpj(n);
    Operator<double> t = create::pipj_inv(n);
    t *= 2./h;
    Operator<double>  a = t*(l+d);
    //if( dir == backward) a = -a.transpose();
    Operator<double> a_bound_right = a; //PER, NEU and DIR_NEU
    Operator<double> a_bound_left = a; //PER, DIR and DIR_NEU
    if( bcx == dg::DIR || bcx == dg::NEU_DIR) 
        a_bound_right = t*(-d.transpose());
    if( bcx == dg::NEU || bcx == dg::NEU_DIR)
        a_bound_left = t*d;
    Operator<double> bp = -t*lr;
    //transform to XSPACE
    Grid1d<double> g( 0,1, n, N);
    Operator<double> backward=g.dlt().backward();
    Operator<double> forward=g.dlt().forward();
    a  = backward*a*forward, a_bound_left  = backward*a_bound_left*forward;
    bp = backward*bp*forward, a_bound_right = backward*a_bound_right*forward;
    
    //assemble the matrix
    SparseBlockMat A(2, N, N);
    if( bcx != PER)
    {
    A.diag[0].resize( 3*n*n), A.diag[1].resize(n*n); 
    A.row[0].resize(  N), A.row[1].resize( N-1);
    A.col[0].resize(  N), A.col[1].resize( N-1);
    A.block[0].resize(  N), A.block[1].resize( N-1);
    A.n[0] = A.n[1] = n;
    A.num_blocks[0] = N; A.num_blocks[1] = N-1;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][(0*n+i)*n+j] = a_bound_left(i,j);
        A.diag[0][(1*n+i)*n+j] = a(i,j);
        A.diag[0][(2*n+i)*n+j] = a_bound_right(i,j);
        A.diag[1][i*n+j] = bp(i,j);
    }
    A.block[0][0] = 0; //a_bound_left
    A.row[0][0] = 0;
    A.col[0][0] = 0;
    for( unsigned i=1; i<N-1; i++) //a
    {
        A.block[0][i] = 1;
        A.row[0][i] = i;
        A.col[0][i] = i;
    }
    A.block[0][N-1] = 2; //a_bound_right
    A.row[0][N-1] = N-1;
    A.col[0][N-1] = N-1;
    for( unsigned i=0; i<N-1; i++) // bp
    {
        A.block[1][i] = 0;
        A.row[1][i] = i+1;
        A.col[1][i] = i;
    }

    }
    else //periodic
    {
    A.diag[0].resize( n*n), A.diag[1].resize(n*n);
    A.row[0].resize(   N), A.row[1].resize(N);
    A.col[0].resize(   N), A.col[1].resize(N);
    A.block[0].resize(   N), A.block[1].resize(N);
    A.n[0] = A.n[1] = n;
    A.num_blocks[0] = A.num_blocks[1] = N;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][i*n+j] = a(i,j);
        A.diag[1][i*n+j] = bp(i,j);
    }
    for( unsigned i=0; i<N; i++) 
    {
        A.block[0][i] = 0; //a
        A.row[0][i] = i;
        A.col[0][i] = i;
        A.block[1][i] = 0; //bp
        A.row[1][i] = i;
        A.col[1][i] = (i-1+N)%N;
    }
    }
    return A;
};

/**
* @brief Create and assemble a cusp Matrix for the normalised jump in 1d in XSPACE.
*
* @ingroup create
* Use cusp internal conversion to create e.g. the fast ell_matrix format.
* The matrix is symmetric. Normalisation is missing
* @tparam T value type
* @param n Number of Legendre nodes per cell
* @param N Vector size ( number of cells)
* @param h cell size ( used to compute normalisation)
* @param bcx boundary condition
*
* @return Host Matrix in coordinate form 
*/
SparseBlockMat jump( unsigned n, unsigned N, double h, bc bcx)
{

    //std::cout << A.row_indices.size(); 
    //std::cout << A.num_cols; //this works!!
    Operator<double> l = create::lilj(n);
    Operator<double> r = create::rirj(n);
    Operator<double> lr = create::lirj(n);
    Operator<double> rl = create::rilj(n);
    Operator<double> a = l+r;
    Operator<double> a_bound_left = a;//DIR and PER
    if( bcx == NEU || bcx == NEU_DIR)
        a_bound_left = r;
    Operator<double> a_bound_right = a; //DIR and PER
    if( bcx == NEU || bcx == DIR_NEU)
        a_bound_right = l;
    Operator<double> b = -rl;
    Operator<double> bp = -lr; 
    //transform to XSPACE
    Operator<double> t = create::pipj_inv(n);
    t *= 2./h;
    Grid1d<double> g( 0,1, n, N);
    Operator<double> backward=g.dlt().backward();
    Operator<double> forward=g.dlt().forward();
    a = backward*t*a*forward, a_bound_left  = backward*t*a_bound_left*forward;
    b = backward*t*b*forward, a_bound_right = backward*t*a_bound_right*forward;
    bp = backward*t*bp*forward;
    //assemble the matrix
    SparseBlockMat A(3, N, N);
    if( bcx != PER)
    {
    A.diag[0].resize( 3*n*n), A.diag[1].resize(n*n), A.diag[2].resize(n*n);
    A.row[0].resize(   N), A.row[1].resize(N-1), A.row[2].resize(N-1);
    A.col[0].resize(   N), A.col[1].resize(N-1), A.col[2].resize(N-1);
    A.block[0].resize(   N), A.block[1].resize(N-1), A.block[2].resize(N-1);
    A.n[0] = A.n[1] = A.n[2] = n;
    A.num_blocks[0] = N, A.num_blocks[1] = A.num_blocks[2] = N-1;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][(0*n+i)*n+j] = a_bound_left(i,j);
        A.diag[0][(1*n+i)*n+j] = a(i,j);
        A.diag[0][(2*n+i)*n+j] = a_bound_right(i,j);
        A.diag[1][i*n+j] = b(i,j);
        A.diag[2][i*n+j] = bp(i,j);
    }
    A.block[0][0] = 0; //a_bound_left
    A.row[0][0] = 0;
    A.col[0][0] = 0;
    for( unsigned i=1; i<N-1; i++) //a
    {
        A.block[0][0*N+i] = 1;
        A.row[0][0*N+i] = i;
        A.col[0][0*N+i] = i;
    }
    A.block[0][N-1] = 2; //a_bound_right
    A.row[0][N-1] = N-1;
    A.col[0][N-1] = N-1;
    for( unsigned i=0; i<N-1; i++) //b and bp
    {
        A.block[1][i] = 0;
        A.row[1][i] = i;
        A.col[1][i] = i+1;
        A.block[2][i] = 0;
        A.row[2][i] = i+1;
        A.col[2][i] = i;
    }

    }
    else //periodic
    {
    A.diag[0].resize( n*n), A.diag[1].resize(n*n), A.diag[2].resize(n*n);
    A.row[0].resize(   N), A.row[1].resize(N), A.row[2].resize(N);
    A.col[0].resize(   N), A.col[1].resize(N), A.col[2].resize(N);
    A.block[0].resize(   N), A.block[1].resize(N), A.block[2].resize(N);
    A.n[0] = A.n[1] = A.n[2] = n;
    A.num_blocks[0] = A.num_blocks[1] = A.num_blocks[2] = N;
    for( unsigned i=0; i<n; i++)
    for( unsigned j=0; j<n; j++)
    {
        A.diag[0][i*n+j] = a(i,j);
        A.diag[1][i*n+j] = b(i,j);
        A.diag[2][i*n+j] = bp(i,j);
    }
    for( unsigned i=0; i<N; i++) 
    {
        A.block[0][i] = 0; //a
        A.row[0][i] = i;
        A.col[0][i] = i;
        A.block[1][i] = 0; //b
        A.row[1][i] = i;
        A.col[1][i] = (i+1)%N;
        A.block[2][i] = 0; //bp
        A.row[2][i] = i;
        A.col[2][i] = (i-1+N)%N;
    }
    }
    return A;
};

SparseBlockMat dx_normed( unsigned n, unsigned N, double h, bc bcx, direction dir )
{
    if( dir == centered)
        return create::dx_symm(n, N, h, bcx);
    else if (dir == forward)
        return create::dx_plus(n, N, h, bcx);
    else if (dir == backward)
        return create::dx_minus(n, N, h, bcx);
    return SparseBlockMat();
}

SparseBlockMat dx( const Grid1d<double>& g, bc bcx, norm no = normed, direction dir = centered)
{
    SparseBlockMat dx;
    dx = dx_normed( g.n(), g.N(), g.h(), bcx, dir);
    if( no == not_normed)
        dx.norm = dg::create::weights( g);
    return dx;
}

///@}
} //namespace create
} //namespace dg

