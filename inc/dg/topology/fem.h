#pragma once

#include "../blas.h"
#include "grid.h"
#include "xspacelib.h"
#include "fem_weights.h"

namespace dg{

/*!@brief Fast (shared memory) tridiagonal sparse matrix
 *
 * Consists of the three diagonal vectors [M, O, P] (for "Minus" -1, "ZerO" 0,
 * "Plus +1), i.e.  M is the subdiagonal, O the diagonal and P the
 * superdiagonal vector.
 * \f$ M_0 \f$ and \f$ P_{N-1}\f$ are ignored
    \f[ T = \begin{pmatrix}
    O_0 & P_0 &   &   &   & \\
    M_1 & O_1 & P_1 &   &   & \\
      & M_2 & O_2 & P_2 &   & \\
      &   & M_3 & O_3 & P_3 & \\
      &   &   &...&   &
      \end{pmatrix}\f]
 * @tparam Container One of the shared memory containers
 * @ingroup sparsematrix
 * @sa dg::mat::TridiagInvDF dg::mat::compute_Tinv_y
 */
template<class Container>
struct TriDiagonal
{
    using value_type = dg::get_value_type<Container>;
    TriDiagonal() = default;
    /*! @brief Allocate size elements for M, O and P
     */
    TriDiagonal( unsigned size) : M(size), O(size), P(size){}

    /*! @brief Directly construct from M, O and P
     * @param M Subdiagonal
     * @param O Diagonal
     * @param P Superdiagonal
     */
    TriDiagonal( Container M, Container O, Container P)
        : M(M), O(O), P(P){}

    /*! @brief Assign M, O, and P from other matrix
     * @tparam Container2
     * @param other
     */
    template<class Container2>
    TriDiagonal( const TriDiagonal<Container2>& other){
        dg::assign( other.M, this->M);
        dg::assign( other.O, this->O);
        dg::assign( other.P, this->P);
    }
    unsigned size()const {return O.size();}
    /*! @brief Resize M, O, and P to given size
     * @param size New size
     */
    void resize( unsigned size)
    {
        M.resize( size);
        O.resize( size);
        P.resize( size);
    }
    /*! @brief Compute Matrix-vector product \f$y = Tx\f$
     *
     * @note Implemented using \c dg::blas2::parallel_for (which only works on
     * shared memory vectors)
     * @param x input
     * @param y result
     */
    void operator()( const Container& x, Container& y) const
    {
        unsigned size = M.size();
        dg::blas2::parallel_for( [size] DG_DEVICE(
                    unsigned i,
                    const value_type* M,
                    const value_type* O,
                    const value_type* P,
                    const value_type* x, value_type* y)
            {
                if(i==0)
                    y[i] = O[i]*x[i] + P[i]*x[i+1];
                else if ( i == size -1 )
                    y[i] = M[i]*x[i-1] + O[i] * x[i];
                else
                    y[i] = M[i]*x[i-1] + O[i] * x[i] + P[i] *x[i+1];
            }, M.size(), M, O, P, x, y);
    }

    ///convert to a sparse matrix format
    dg::IHMatrix_t<value_type> asIMatrix() const{
        unsigned size = M.size();
        thrust::host_vector<int> A_row_offsets(size+1), A_column_indices( 3*size-2);
        thrust::host_vector<value_type> A_values( 3*size-2);
        A_row_offsets[0] = 0;
        A_column_indices[0] = 0;
        A_values[0] = O[0];

        A_column_indices[1] = 1;
        A_values[1] = P[0];

        A_row_offsets[1] = 2;

        for( unsigned i=1;i<size; i++)
        {
            A_column_indices[3*i-1+0] = i-1;
            A_values[3*i-1+0] = M[i];

            A_column_indices[3*i-1+1] = i;
            A_values[3*i-1+1] = O[i];

            if( i != (size-1))
            {
                A_column_indices[3*i-1+2] = i+1;
                A_values[3*i-1+2] = P[i];
            }
            A_row_offsets[i+1] = A_row_offsets[i] + ( i != (size-1) ? 3 : 2);
        }
        return {size, size, A_row_offsets, A_column_indices, A_values};
    }

    Container M; //!< Subdiagonal ["Minus" -1] <tt>M[0]</tt> is ignored <tt>M[1]</tt> maps to <tt>T_10</tt>
    Container O; //!< Diagonal ["zerO" 0]       <tt>O[0]</tt> maps to <tt>T_00</tt>
    Container P; //!< Uper diagonal ["Plus" +1] <tt>P[0]</tt> maps to <tt>T_01</tt>
};

///@addtogroup fem
///@{
/*!@brief DEPRECATED/UNTESTED Fast inverse tridiagonal sparse matrix
 *
 * When applied to a vector, uses Thomas algorithm to compute \f$ T^{-1} v\f$
 * @attention Only for shared memory host vectors
 * @sa dg::mat::TridiagInvDF dg::mat::compute_Tinv_y
 */
template<class value_type>
struct InverseTriDiagonal
{
    InverseTriDiagonal() = default;
    InverseTriDiagonal( const TriDiagonal<thrust::host_vector<value_type>>& tri)
    {
        dg::assign( tri.M, this->M);
        dg::assign( tri.O, this->O);
        dg::assign( tri.P, this->P);
    }

    /// \f$ x = T^{-1} y\f$
    void operator()( const thrust::host_vector<value_type>& y, thrust::host_vector<value_type>& x) const
    {
        unsigned size = M.size();
        thrust::host_vector<value_type> ci(size), di(size);
        x.resize(size);
        ci[0] = P[0]/O[0];
        di[0] = y[0]/O[0];
        for( unsigned i=1; i<size; i++)
        {
            ci[i] = P[i]/ ( O[i] -M[i]*ci[i-1]);
            di[i] = (y[i]-M[i]*di[i-1])/(O[i] -M[i]*ci[i-1]);
        }
        x[size-1] = di[size-1];
        for( int i=size-2; i>=0; i--)
            x[i] = di[i] - ci[i]*x[i+1];
    }
    private:
    thrust::host_vector<value_type> M, O, P;
};

/*!@brief Fast tridiagonal sparse matrix in 2d \f$ T_y\otimes T_x\f$
 *
 * Consists of Two \c TriDiagonal matrices \f$ T_x\f$ and \f$ T_y\f$
 * @note It is fast to apply using \c dg::blas2::parallel_for (which only works on shared memory vectors though)
 * @tparam Container One of the shared memory containers
 */
template<class Container>
struct KroneckerTriDiagonal2d
{
    using value_type = dg::get_value_type<Container>;
    KroneckerTriDiagonal2d() = default;
    KroneckerTriDiagonal2d( TriDiagonal<Container> my, TriDiagonal<Container> mx): m_nz(1), m_y(my), m_x(mx){}
    KroneckerTriDiagonal2d( unsigned nz, TriDiagonal<Container> my, TriDiagonal<Container> mx): m_nz(nz), m_y(my), m_x(mx){}

    unsigned& nz() { return m_nz;}
    unsigned nz() const { return m_nz;}
    template<class Container2>
    KroneckerTriDiagonal2d( const KroneckerTriDiagonal2d<Container2>& other){
        m_x = other.x();
        m_y = other.y();
        m_nz = other.nz();
    }
    const TriDiagonal<Container>& x() const {return m_x;}
    const TriDiagonal<Container>& y() const {return m_y;}
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0& x, ContainerType1& y) const
    {
        unsigned size = m_y.size()*m_x.size()*m_nz;
        unsigned Nx = m_x.size(), Ny = m_y.size();
        dg::blas2::parallel_for( [Nx, Ny] DG_DEVICE(
                    unsigned i,
                    const value_type* yM,
                    const value_type* yO,
                    const value_type* yP,
                    const value_type* xM,
                    const value_type* xO,
                    const value_type* xP,
                    const value_type* x, value_type* y){
            unsigned j = (i/Nx)/Ny;
            unsigned k = (i/Nx)%Ny, l = i%Nx;
            value_type a, b, c;
            if(l==0)
            {
                if( k==0)
                {
                    b = xO[l]*x[(j*Ny+k)*Nx+l] + xP[l]*x[(j*Ny+k)*Nx+l+1];
                    c = xO[l]*x[(j*Ny+k+1)*Nx+l] + xP[l]*x[(j*Ny+k+1)*Nx+l+1];
                    y[i] = yO[k]*b + yP[k]*c;
                }
                else if( k == Ny-1)
                {
                    a = xO[l]*x[(j*Ny+k-1)*Nx+l] + xP[l]*x[(j*Ny+k-1)*Nx+l+1];
                    b = xO[l]*x[(j*Ny+k)*Nx+l] + xP[l]*x[(j*Ny+k)*Nx+l+1];
                    y[i] = yM[k]*a + yO[k]*b;
                }
                else
                {
                    a = xO[l]*x[(j*Ny+k-1)*Nx+l] + xP[l]*x[(j*Ny+k-1)*Nx+l+1];
                    b = xO[l]*x[(j*Ny+k)*Nx+l] + xP[l]*x[(j*Ny+k)*Nx+l+1];
                    c = xO[l]*x[(j*Ny+k+1)*Nx+l] + xP[l]*x[(j*Ny+k+1)*Nx+l+1];
                    y[i] = yM[k]*a + yO[k]*b + yP[k]*c;
                }
            }
            else if ( l == Nx -1 )
            {
                if( k==0)
                {
                    b = xM[l]*x[(j*Ny+k)*Nx+l-1] + xO[l]*x[(j*Ny+k)*Nx+l];
                    c = xM[l]*x[(j*Ny+k+1)*Nx+l-1] + xO[l]*x[(j*Ny+k+1)*Nx+l];
                    y[i] = yO[k]*b + yP[k]*c;
                }
                else if ( k == Ny -1)
                {
                    a = xM[l]*x[(j*Ny+k-1)*Nx+l-1] + xO[l]*x[(j*Ny+k-1)*Nx+l];
                    b = xM[l]*x[(j*Ny+k)*Nx+l-1] + xO[l]*x[(j*Ny+k)*Nx+l];
                    y[i] = yM[k]*a + yO[k]*b;
                }
                else
                {
                    a = xM[l]*x[(j*Ny+k-1)*Nx+l-1] + xO[l]*x[(j*Ny+k-1)*Nx+l];
                    b = xM[l]*x[(j*Ny+k)*Nx+l-1] + xO[l]*x[(j*Ny+k)*Nx+l];
                    c = xM[l]*x[(j*Ny+k+1)*Nx+l-1] + xO[l]*x[(j*Ny+k+1)*Nx+l];
                    y[i] = yM[k]*a + yO[k]*b + yP[k]*c;
                }
            }
            else
            {
                if( k==0)
                {
                    b = xM[l]*x[(j*Ny+k)*Nx+l-1] + xO[l]*x[(j*Ny+k)*Nx+l] +
                        xP[l]*x[(j*Ny+k)*Nx+l+1];
                    c = xM[l]*x[(j*Ny+k+1)*Nx+l-1] + xO[l]*x[(j*Ny+k+1)*Nx+l] +
                        xP[l]*x[(j*Ny+k+1)*Nx+l+1];
                    y[i] = yO[k]*b + yP[k]*c;
                }
                else if ( k == Ny -1)
                {
                    a = xM[l]*x[(j*Ny+k-1)*Nx+l-1] + xO[l]*x[(j*Ny+k-1)*Nx+l] +
                        xP[l]*x[(j*Ny+k-1)*Nx+l+1];
                    b = xM[l]*x[(j*Ny+k)*Nx+l-1] + xO[l]*x[(j*Ny+k)*Nx+l] +
                        xP[l]*x[(j*Ny+k)*Nx+l+1];
                    y[i] = yM[k]*a + yO[k]*b;
                }
                else
                {
                    a = xM[l]*x[(j*Ny+k-1)*Nx+l-1] + xO[l]*x[(j*Ny+k-1)*Nx+l] +
                        xP[l]*x[(j*Ny+k-1)*Nx+l+1];
                    b = xM[l]*x[(j*Ny+k)*Nx+l-1] + xO[l]*x[(j*Ny+k)*Nx+l] +
                        xP[l]*x[(j*Ny+k)*Nx+l+1];
                    c = xM[l]*x[(j*Ny+k+1)*Nx+l-1] + xO[l]*x[(j*Ny+k+1)*Nx+l] +
                        xP[l]*x[(j*Ny+k+1)*Nx+l+1];
                    y[i] = yM[k]*a + yO[k]*b + yP[k]*c;
                }
            }
        }, size, m_y.M, m_y.O, m_y.P, m_x.M, m_x.O, m_x.P, x, y);
    }
    private:
    unsigned m_nz;
    dg::TriDiagonal<Container> m_y, m_x;
};

/*!@brief Fast inverse tridiagonal sparse matrix in 2d \f$ T_y^{-1}\otimes T_x^{-1}\f$
 *
 * When applied to a vector, uses Thomas algorithm to compute \f$ T^{-1} v\f$ first
 * row-wise in x and then column-wise in y
 * @attention Only for shared memory vectors (works for GPUs but is not fast)
 */
template<class Container>
struct InverseKroneckerTriDiagonal2d
{
    using value_type = dg::get_value_type<Container>;
    InverseKroneckerTriDiagonal2d() = default;
    InverseKroneckerTriDiagonal2d( const KroneckerTriDiagonal2d<Container>& tri)
    {
        m_t = tri;
        unsigned size = m_t.x().size()*m_t.y().size()*m_t.nz();
        m_ci.resize( size);
        m_di.resize( size);
        m_tmp.resize( size);
    }
    template<class Container2>
    InverseKroneckerTriDiagonal2d( const InverseKroneckerTriDiagonal2d<Container2>& inv_tri)
    {
        m_t = inv_tri.tri();
        unsigned size = m_t.x().size()*m_t.y().size()*m_t.nz();
        m_ci.resize( size);
        m_di.resize( size);
        m_tmp.resize( size);
    }
    const KroneckerTriDiagonal2d<Container>& tri() const {return m_t;}
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0& y, ContainerType1& x)
    {
        unsigned Nx = m_t.x().size(), Ny = m_t.y().size();
        // solve in two passes, first x then y
        dg::blas2::parallel_for( [ this, Nx] DG_DEVICE(
                    unsigned k,
                    const value_type* M,
                    const value_type* O,
                    const value_type* P,
                    value_type* ci,
                    value_type* di,
                    const value_type* y, value_type* x){
            ci[k*Nx + 0] = P[0]/O[0];
            di[k*Nx + 0] = y[k*Nx + 0]/O[0];
            for( unsigned i=1; i<Nx; i++)
            {
                ci[k*Nx+i] = P[i]/ ( O[i] -M[i]*ci[k*Nx+i-1]);
                di[k*Nx+i] = (y[k*Nx+i]-M[i]*di[k*Nx+i-1])/(O[i] -M[i]*ci[k*Nx+i-1]);
            }
            x[k*Nx + Nx-1] = di[k*Nx + Nx-1];
            for( int i=Nx-2; i>=0; i--)
                x[k*Nx+i] = di[k*Nx+i] - ci[k*Nx+i]*x[k*Nx +i+1];

        }, m_t.y().size()*m_t.nz(), m_t.x().M, m_t.x().O, m_t.x().P, m_ci, m_di, y, m_tmp);

        dg::blas2::parallel_for( [ this, Nx, Ny] DG_DEVICE(
                    unsigned l,
                    const value_type* M,
                    const value_type* O,
                    const value_type* P,
                    value_type* ci,
                    value_type* di,
                    const value_type* y, value_type* x){
            unsigned i = l%Nx, j = l/Nx;
            ci[(j*Ny+0)*Nx + i] = P[0]/O[0];
            di[(j*Ny+0)*Nx + i] = y[(j*Ny+0)*Nx + i]/O[0];
            for( unsigned k=1; k<Ny; k++)
            {
                ci[(j*Ny+k)*Nx+i] = P[k]/ ( O[k] -M[k]*ci[(j*Ny+k-1)*Nx+i]);
                di[(j*Ny+k)*Nx+i] = (y[(j*Ny+k)*Nx+i]-M[k]*di[(j*Ny+k-1)*Nx+i])/(O[k] -M[k]*ci[(j*Ny+k-1)*Nx+i]);
            }
            x[(j*Ny+Ny-1)*Nx + i] = di[(j*Ny+Ny-1)*Nx + i];
            for( int k=Ny-2; k>=0; k--)
                x[(j*Ny+k)*Nx+i] = di[(j*Ny+k)*Nx+i] - ci[(j*Ny+k)*Nx+i]*x[(j*Ny+k+1)*Nx +i];

        }, m_t.x().size()*m_t.nz(), m_t.y().M, m_t.y().O, m_t.y().P, m_ci, m_di,m_tmp, x);
    }

    private:
    KroneckerTriDiagonal2d<Container> m_t;
    Container m_ci, m_di, m_tmp;

};
///@}

namespace create{

///@addtogroup fem
///@{

/*!@class hide_fem_mass_doc
* @brief \f$ S_{ij} = \frac{1}{w_i}\int v_i(x) v_j(x) \f$ finite element projection matrix
*
* where \f$ v_j\f$ are triangle finite elements
* @tparam real_type The value type
* @param g The grid
* @return Host Matrix
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
* @attention No periodic boundary conditions
*/
/*!@class hide_fem_inv_mass_doc
* @brief Inverse finite element mass matrix \f$ S^{-1} \f$
*
* @tparam real_type The value type
* @param g The grid
* @return Host Matrix
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
* @attention No periodic boundary conditions
*/
/*!@class hide_fem_linear2const_doc
* @brief \f$ S_{ij} = \frac{1}{w_i}\int c_i(x) v_j(x) \f$ finite element projection matrix
*
* where \f$ c_i\f$ are the constant finite elements and \f$ v_j\f$ are triangles
* @tparam real_type The value type
* @param g The grid
* @return Host Matrix
* @sa <a href="https://www.overleaf.com/read/rpbjsqmmfzyj" target="_blank">Introduction to dg methods</a>
* @attention No periodic boundary conditions
*/

///@copydoc hide_fem_mass_doc
template<class real_type>
dg::TriDiagonal<dg::HVec_t<real_type>> fem_mass(
    const RealGrid1d<real_type>& g)
{
    dg::TriDiagonal<dg::HVec_t<real_type>> A(g.size());
    std::vector<real_type> xx = dg::DLT<real_type>::abscissas(g.n());
    std::vector<real_type> xa( g.n()+2);
    xa[0] = (xx[g.n()-1]-2)*g.h()/2.; // the last one from the previous cell
    for( unsigned i=0; i<g.n(); i++)
        xa[i+1]=xx[i]*g.h()/2.;
    xa[g.n()+1] = (xx[0]+2)*g.h()/2.; // the first one from next cell
    const real_type* x = &xa[1];
    real_type xleft = -g.h()/2., xright = g.h()/2.;
    auto weights = fem_weights(g);
    for( unsigned i=0; i<g.N(); i++)
        for( int k=0; k<(int)g.n(); k++)
        {
            if( i==0 && k == 0)
            {
                A.M[0] = 0.;
                A.O[0] = (4*x[0]-6*xleft+2*x[1])/6./weights[0];
                A.P[0] = (x[1]-x[0])/6./weights[0];
                continue;
            }
            int I = (i*g.n()+k);
            if( (i==g.N()-1) && (k == (int)g.n()-1))
            {
                A.M[I] = (x[k]-x[k-1])/6./weights[I];
                A.O[I] = (-4*x[k]+6*xright-2*x[k-1])/6./weights[I];
                A.P[I] = 0.;
                continue;
            }
            A.M[I] =    (x[k]-x[k-1])/6./weights[I];
            A.O[I] = 2.*(x[k+1]-x[k-1])/6./weights[I];
            A.P[I] =    (x[k+1]-x[k])/6./weights[I];
        }
    return A;
}

///@copydoc hide_fem_linear2const_doc
template<class real_type>
dg::TriDiagonal<dg::HVec_t<real_type>> fem_linear2const(
    const RealGrid1d<real_type>& g)
{
    //bug! periodic boundary conditions
    dg::TriDiagonal<dg::HVec_t<real_type>> A(g.size());
    std::vector<real_type> xx = dg::DLT<real_type>::abscissas(g.n());
    std::vector<real_type> xa( g.n()+2);
    xa[0] = (xx[g.n()-1]-2)*g.h()/2.; // the last one from the previous cell
    for( unsigned i=0; i<g.n(); i++)
        xa[i+1]=xx[i]*g.h()/2.;
    xa[g.n()+1] = (xx[0]+2)*g.h()/2.; // the first one from next cell
    const real_type* x = &xa[1];
    real_type xleft = -g.h()/2., xright = g.h()/2.;
    auto weights = fem_weights(g);

    for( unsigned i=0; i<g.N(); i++)
        for( int k=0; k<(int)g.n(); k++)
        {
            if( i==0 && k == 0)
            {
                A.M[0] = 0.;
                A.O[0] = (5*x[0]-8*xleft+3*x[1])/8./weights[0];
                A.P[0] = (x[1]-x[0])/8./weights[0];
                continue;
            }
            int I = (i*g.n()+k);
            if( (i==g.N()-1) && (k == (int)g.n()-1))
            {
                A.M[I] = (x[k]-x[k-1])/8./weights[I];
                A.O[I] = (-5*x[k]+8*xright-3*x[k-1])/8./weights[I];
                A.P[I] = 0.;
                continue;
            }
            A.M[I] =    (x[k]-x[k-1])/8./weights[I];
            A.O[I] = 3.*(x[k+1]-x[k-1])/8./weights[I];
            A.P[I] =    (x[k+1]-x[k])/8./weights[I];
        }
    return A;
}

///@copydoc hide_fem_mass_doc
template<class real_type>
dg::KroneckerTriDiagonal2d<dg::HVec_t<real_type>> fem_mass(
    const aRealTopology2d<real_type>& g)
{
    auto mx = fem_mass(g.gx());
    auto my = fem_mass(g.gy());
    return {my, mx};
}

///@copydoc hide_fem_inv_mass_doc
template<class real_type>
dg::InverseKroneckerTriDiagonal2d<dg::HVec_t<real_type>> inv_fem_mass(
    const aRealTopology2d<real_type>& g)
{
    auto tri = fem_mass( g);
    return {tri};
}

///@copydoc hide_fem_linear2const_doc
template<class real_type>
dg::KroneckerTriDiagonal2d<dg::HVec_t<real_type>> fem_linear2const(
    const aRealTopology2d<real_type>& g)
{
    auto mx = fem_linear2const(g.gx());
    auto my = fem_linear2const(g.gy());
    return {my, mx};
}

///@copydoc hide_fem_inv_mass_doc
template<class real_type>
dg::InverseKroneckerTriDiagonal2d<dg::HVec_t<real_type>> inv_fem_linear2const(
    const aRealTopology2d<real_type>& g)
{
    auto tri = fem_linear2const( g);
    return {tri};
}

///@copydoc hide_fem_mass_doc
template<class real_type>
dg::KroneckerTriDiagonal2d<dg::HVec_t<real_type>> fem_mass2d(
    const aRealTopology3d<real_type>& g)
{
    auto mx = fem_mass(g.gx());
    auto my = fem_mass(g.gy());
    return {g.gz().size(), my, mx};
}

///@copydoc hide_fem_inv_mass_doc
template<class real_type>
dg::InverseKroneckerTriDiagonal2d<dg::HVec_t<real_type>> inv_fem_mass2d(
    const aRealTopology3d<real_type>& g)
{
    auto tri = fem_mass2d( g);
    return {tri};
}

///@copydoc hide_fem_linear2const_doc
template<class real_type>
dg::KroneckerTriDiagonal2d<dg::HVec_t<real_type>> fem_linear2const2d(
    const aRealTopology3d<real_type>& g)
{
    auto mx = fem_linear2const(g.gx());
    auto my = fem_linear2const(g.gy());
    return {g.gz().size(), my, mx};
}

///@copydoc hide_fem_inv_mass_doc
template<class real_type>
dg::InverseKroneckerTriDiagonal2d<dg::HVec_t<real_type>> inv_fem_linear2const2d(
    const aRealTopology3d<real_type>& g)
{
    auto tri = fem_linear2const2d( g);
    return {tri};
}

///@}
}//namespace create
}//namespace dg
