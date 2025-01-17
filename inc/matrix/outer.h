#pragma once


#include <vector>
#include <cusp/elementwise.h>
#include <cusp/lapack/lapack.h>
#include <cusp/print.h>
#include "dg/algorithm.h"

namespace dg{
namespace mat{

template<class ContainerType0, class ContainerType1, class ContainerType2>
void outer_product( const ContainerType0& vx, const ContainerType1& vy, ContainerType2& y)
{
    using value_type = get_value_type<ContainerType0>;
    unsigned size = y.size();
    unsigned Nx = vx.size(), Ny = vy.size();
    unsigned product_size = Nx*Ny;
    if( size != product_size )
        throw dg::Error( Message( _ping_) << "Size " << size << " incompatible with outer produt size "<<product_size<<"!");
    dg::blas2::parallel_for([Nx ] DG_DEVICE(
        unsigned i, value_type* y, const value_type* vx, const value_type* vy)
    {
        y[i] = vx[i%Nx]*vy[i/Nx];
    }, size, y, vx, vy);
}

template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
void outer_product( const ContainerType0& vx, const ContainerType1& vy, ContainerType2& vz, ContainerType3& y)
{
    using value_type = get_value_type<ContainerType0>;
    unsigned size = y.size();
    unsigned Nx = vx.size(), Ny = vy.size(), Nz = vz.size();
    unsigned product_size = Nx*Ny*Nz;
    if( size != product_size )
        throw dg::Error( Message( _ping_) << "Size " << size << " incompatible with outer produt size "<<product_size<<"!");
    dg::blas2::parallel_for( [Nx, Ny] DG_DEVICE(
        unsigned i, value_type* y, const value_type* vx, const value_type* vy, const value_type* vz)
    {
        y[i] = vx[i%Nx]*vy[(i/Nx)%Ny]*vz[i/(Nx*Ny)];
    }, size, y, vx, vy, vz);
}

#ifdef MPI_VERSION
// Not tested yet
template<class ContainerType0, class ContainerType1, class ContainerType2>
void outer_product( const MPI_Vector<ContainerType0>& vx, const MPI_Vector<ContainerType1>& vy, MPI_Vector<ContainerType2>& y)
{
    outer_product( vx.data(), vy.data(), y.data());
}
template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
void outer_product( const MPI_Vector<ContainerType0>& vx, const MPI_Vector<ContainerType1>& vy, MPI_Vector<ContainerType2>& vz, MPI_Vector<ContainerType3>& y)
{
    outer_product( vx.data(), vy.data(), vz.data(), y.data());
}
#endif // MPI_VERSION



/**
* @brief The dG discretization of the 1d Laplacian \f$ -\frac{1}{v} \partial_x ( v \partial_x ) \f$
 in split form  \f$ D = B^{-1}A\f$ such that A is symmetric, and B is diagonal.

 The reason for such a format is that Lapack's symmetric generalized Eigenvalue
solver *sygv* can be used in the form \f$ A x = \lambda B x\f$
*/
template<class value_type>
std::array<cusp::array2d< value_type, cusp::host_memory >,2> sym_laplace1d(
const thrust::host_vector<value_type>& volume,
   RealGrid1d<value_type> g1d, bc bcx,
       direction dir = forward,
       value_type jfactor = 1.)
{
    auto leftx = dg::create::dx( g1d, inverse( bcx), inverse(dir)).asCuspMatrix();
    dg::blas1::scal( leftx.values, -1.);
    auto vol = dg::create::diagonal( volume );
    auto rightx =  dg::create::dx( g1d, bcx, dir).asCuspMatrix();
    auto jumpx = dg::create::jumpX(g1d, bcx).asCuspMatrix();
    dg::blas1::scal( jumpx.values, jfactor);
    cusp::coo_matrix<int, value_type, cusp::host_memory> A, CX, XX;
    cusp::multiply( vol, rightx, CX );
    cusp::multiply( leftx, CX, XX);
    cusp::add( XX, jumpx, CX);

    dg::HVec w1d = dg::create::weights( g1d);
    auto B = dg::create::diagonal( w1d);
    cusp::multiply( B, CX, A); // get symmetric part of A

    dg::blas1::pointwiseDot( w1d, volume, w1d);
    B = dg::create::diagonal( w1d);

    std::array<cusp::array2d< value_type, cusp::host_memory >,2> out;
    cusp::convert( A, out[0]);
    cusp::convert( B, out[1]);
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
        auto VX = lapX[0];
        m_EX.resize( g.gx().size());
        cusp::lapack::sygv( lapX[0], lapX[1], m_EX, VX);
        // Eigenvalues are sorted in ascending order
        // now convert to device vectors
        m_VX.resize( m_EX.size());
        for( unsigned i=0; i<m_EX.size(); i++)
        {
            dg::HVec temp(VX.column(i).begin(), VX.column(i).end());
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
        auto VY = lapY[0];
        m_EY.resize( g.gy().size());
        cusp::lapack::sygv( lapY[0], lapY[1], m_EY, VY);
        // Eigenvalues are sorted in ascending order
        // now convert to device vectors
        m_VY.resize( m_EY.size());
        for( unsigned i=0; i<m_EY.size(); i++)
        {
            dg::HVec temp(VY.column(i).begin(), VY.column(i).end());
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
            outer_product( m_VX[ix], m_VY[iy], m_v);
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
            outer_product( m_VX[ix], m_VY[iy], m_v);
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
            outer_product( m_VX[ix], m_VY[iy], m_v);
            value_type beta = dg::blas2::dot( m_v, b, m_weights);
            dg::blas1::evaluate( m_f, dg::equals(), op, diag, m_EX[ix] + m_EY[iy]);
            dg::blas1::pointwiseDot( beta, m_f, m_v, 1., x);
        }
        return size;
    }
    private:
    cusp::array1d<value_type, cusp::host_memory> m_EX, m_EY;
    std::vector<ContainerType> m_VX, m_VY;
    ContainerType m_weights, m_v, m_f;
    thrust::host_vector<unsigned> m_idx;
};


} // namespace mat
} // namespace dg
