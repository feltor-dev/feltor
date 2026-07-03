#pragma once


#include <vector>
#include "dg/algorithm.h"
#include "tridiaginv.h" // lapack wrapper

namespace dg{
namespace mat{

/// @cond
template<class index_type, class value_type>
dg::SquareMatrix<value_type> asSquareMatrix( const dg::SparseMatrix<index_type, value_type, thrust::host_vector>& mat)
{
    if( mat.num_rows() != mat.num_cols())
        throw dg::Error( dg::Message( _ping_) << "Cannot convert non square sparse matrix with "<<mat.num_rows()<<" rows and "<<mat.num_cols()<<" columns\n");
    dg::SquareMatrix<value_type> out( mat.num_rows(), 0);
    for( unsigned row=0; row<mat.num_rows(); row++)
        for( index_type jj = mat.row_offsets()[row]; jj< mat.row_offsets()[row+1]; jj++)
        {
            index_type col = mat.column_indices()[jj];
            value_type val = mat.values()[jj];
            out( row, col ) = val;
        }
    return out;
}

/**
* @brief The dG discretization of the 1d Laplacian \f$ -\frac{1}{v} \partial_x ( v \partial_x ) \f$
 in split form  \f$ D = B^{-1}A\f$ such that A is symmetric, and B is diagonal.

 The reason for such a format is that Lapack's symmetric generalized Eigenvalue
solver *sygv* can be used in the form \f$ A x = \lambda B x\f$
*/
template<class value_type>
std::array<dg::SquareMatrix<value_type>,2> sym_laplace1d(
   const thrust::host_vector<value_type>& volume,
   RealGrid1d<value_type> g1d, bc bcx,
   direction dir = forward,
   value_type jfactor = 1.)
{

    auto w1d = dg::create::weights( g1d);
    dg::SquareMatrix<value_type> VOL ( volume.size()),  W1D( volume.size());
    for( unsigned u=0; u<volume.size(); u++)
    {
        VOL(u,u) = volume[u];
        W1D(u,u) = w1d[u];
    }
    auto leftx =  asSquareMatrix( dg::create::dx( g1d, inverse( bcx), inverse(dir) ).asCuspMatrix() );
    auto rightx = asSquareMatrix( dg::create::dx( g1d, bcx, dir).asCuspMatrix());
    auto jumpx  = asSquareMatrix( dg::create::jumpX(g1d, bcx).asCuspMatrix());

    std::array<dg::SquareMatrix<value_type>,2> out;
    out[0] = W1D * (- leftx * (VOL * rightx) + jfactor*jumpx); // get symmetric part of A
    dg::blas1::pointwiseDot( W1D.data(), VOL.data(), W1D.data());
    out[1] = W1D;
    return out;
}

template< class ContainerType>
struct LaplaceDecomposition
{
    using value_type = dg::get_value_type<ContainerType>;

    /// Construct empty
    LaplaceDecomposition() = default;
    LaplaceDecomposition( RealGrid2d<value_type> g, bc bcx, bc bcy,
        direction dir = forward,
        value_type jfactor=1.)
    {
        m_v = m_f = m_weights = dg::create::weights( g);
        auto lapX = sym_laplace1d(dg::evaluate( dg::one, g.gx()), g.gx(), bcx,
                                  dir, jfactor);
        unsigned Nx = g.gx().size();
        m_EX.resize( Nx);
        thrust::host_vector<value_type> work( 3*Nx-1);
        lapack::sygv( 1, 'V', 'U', Nx, lapX[0].data(), Nx, lapX[1].data(), Nx, m_EX, work);
        // Eigenvalues are sorted in ascending order
        // now convert to device vectors
        m_VX.resize( Nx);
        for( unsigned i=0; i<Nx; i++)
        {
            dg::HVec temp(Nx);
            for( unsigned u=0; u<Nx; u++)
                temp[u] = lapX[0](i, u); // i-th row of lapX[0] is i-th Eigenvector
            dg::assign( temp, m_VX[i]);
        }
        // orthogonality is maintained with off-diagonal elements of order 1e-16
        //ContainerType w1d = dg::create::weights( g.gx());
        //for( unsigned i=0; i<m_EX.size(); i++)
        //{
        //    std::cout << std::endl;
        //    for( unsigned j=0; j<m_EX.size(); j++)
        //        std::cout << dg::blas2::dot( m_VX[i], w1d, m_VX[j])<<" ";
        //}

        auto lapY = sym_laplace1d(dg::evaluate( dg::one, g.gy()), g.gy(), bcy,
                                  dir, jfactor);
        unsigned Ny = g.gy().size();
        m_EY.resize( Ny);
        work.resize( 3*Ny-1);
        lapack::sygv( 1, 'V', 'U', Ny, lapY[0].data(), Ny, lapY[1].data(), Ny, m_EY, work);
        // Eigenvalues are sorted in ascending order
        // now convert to device vectors
        m_VY.resize( Ny);
        for( unsigned i=0; i<Nx; i++)
        {
            dg::HVec temp(Ny);
            for( unsigned u=0; u<Ny; u++)
                temp[u] = lapY[0](i, u); // i-th row of lapY[0] is i-th Eigenvector
            dg::assign( temp, m_VY[i]);
        }
        // Get the sorted indices of Eigenvalues
        thrust::host_vector<value_type> evs( m_EY.size()*m_EX.size());
        thrust::host_vector<unsigned> idx( evs.size());
        thrust::sequence( idx.begin(), idx.end());
        for( unsigned i=0; i<evs.size(); i++)
            evs[i] = m_EY[i/m_EX.size()] + m_EX[i%m_EX.size()];
        thrust::stable_sort_by_key( evs.begin(), evs.end(), idx.begin());
        m_idx = idx;
        //for( unsigned i=0; i<idx.size(); i++)
        //    std::cout << "("<<idx[i]%m_EX.size()<<","<<idx[i]/m_EX.size()<<") ";
        //std::cout << std::endl;
    }
      // f(Lap)b
    template<class ContainerType0, class UnaryOp,
        class ContainerType1>
    unsigned matrix_function(
            ContainerType0& x,
            UnaryOp op,
            const ContainerType1& b,
            value_type eps,
            value_type nrmb_correction = 1.)
    {
        unsigned size = m_idx.size(), Nx = m_EX.size();
        value_type normb = sqrt( dg::blas2::dot( b, m_weights, b));
        value_type normx2 = 0.;
        dg::blas1::copy( 0, x);
        for( unsigned i=0; i<size; i++)
        {
            unsigned ix = m_idx[i]%Nx, iy = m_idx[i]/Nx;
            value_type alpha = op(m_EX[ix]+m_EY[iy]);
            if( alpha*normb <= eps*(sqrt(normx2)+nrmb_correction))
                return i;
            //func(lambda_i)*(vyXVx)_i .dot ( Mb) (vyXVx)_i
            dg::blas1::kronecker( m_v, dg::equals(), dg::Product(), m_VX[ix], m_VY[iy]);
            alpha*=dg::blas2::dot( m_v, b, m_weights);
            dg::blas1::axpby( alpha, m_v, 1., x);
            normx2 += alpha*alpha;
        }
        return size;
    }
    // f(Lap, diag)b
    template<class ContainerType0, class BinaryOp,
        class ContainerType1, class ContainerType2>
    unsigned product_function_adjoint(
            ContainerType0& x,
            BinaryOp op,
            const ContainerType1& diag,
            const ContainerType2& b,
            value_type eps,
            value_type nrmb_correction = 1.)
    {
        unsigned size = m_idx.size(), Nx = m_EX.size();
        value_type normb = sqrt( dg::blas2::dot( b, m_weights, b));
        value_type dmin = dg::blas1::reduce( diag, (value_type)1e50, thrust::minimum<value_type>());
        value_type normx2 = 0.;
        dg::blas1::copy( 0, x);
        for( unsigned i=0; i<size; i++)
        {
            unsigned ix = m_idx[i]%Nx, iy = m_idx[i]/Nx;
            value_type err = normb*op(m_EX[ix] + m_EY[iy], dmin)*sqrt(size);
            //std::cout << "Eigenvalue "<<(m_EX[i%Nx]+m_EY[i/Nx])<<"\n";
            //std::cout << "alpha "<<op(m_EX[i%Nx]+m_EY[i/Nx], dmin)<<"\n";
            //std::cout << "normx2 "<<normx2<<"\n";
            if( err <= eps*(sqrt(normx2)+nrmb_correction))
                return i;
            //func( d, lambda_ij)*(vyXVx)_ij .dot ( Mx) (vyXVx)_ij
            dg::blas1::kronecker( m_v, dg::equals(), dg::Product(), m_VX[ix], m_VY[iy]);
            dg::blas1::evaluate( m_f, dg::equals(), op, m_EX[ix] + m_EY[iy], diag);
            dg::blas1::pointwiseDot( m_f, m_v, m_f);
            value_type gamma = dg::blas2::dot( m_f, b, m_weights);
            dg::blas1::axpby( gamma, m_v, 1., x);
            normx2 += gamma*gamma;
        }
        return size;
    }
    // f(diag, Lap)b
    // is this faster than Lanczos?
    template<class ContainerType0, class BinaryOp,
      class ContainerType1, class ContainerType2>
    unsigned product_function(
          ContainerType0& x,
          BinaryOp op,
          const ContainerType1& diag,
          const ContainerType2& b,
          value_type eps,
          value_type nrmb_correction = 1.)
    {
        unsigned size = m_idx.size(), Nx = m_EX.size();
        value_type normb = sqrt( dg::blas2::dot( b, m_weights, b));
        value_type dmin = dg::blas1::reduce( diag, (value_type)1e50, thrust::minimum<value_type>());
        dg::blas1::copy( 0, x);
        for( unsigned i=0; i<size; i++)
        {
            unsigned ix = m_idx[i]%Nx, iy = m_idx[i]/Nx;
            value_type err = normb*op(dmin, m_EX[ix] + m_EY[iy])*sqrt(size);
            value_type normx = sqrt(dg::blas2::dot( x, m_weights, x));
            if( err <= eps*(normx+nrmb_correction))
                return i;

            //func( d, lambda_i)*(vyXVx)_i .dot ( Mx) (vyXVx)_i
            dg::blas1::kronecker( m_v, dg::equals(), dg::Product(), m_VX[ix], m_VY[iy]);
            value_type beta = dg::blas2::dot( m_v, b, m_weights);
            dg::blas1::evaluate( m_f, dg::equals(), op, diag, m_EX[ix] + m_EY[iy]);
            dg::blas1::pointwiseDot( beta, m_f, m_v, 1., x);
        }
        return size;
    }
    private:
    thrust::host_vector<value_type> m_EX, m_EY;
    std::vector<ContainerType> m_VX, m_VY;
    ContainerType m_weights, m_v, m_f;
    thrust::host_vector<unsigned> m_idx;
};
/// @endcond


} // namespace mat
} // namespace dg
