#pragma once

#include "cusp/transpose.h"
#include "dg/backend/memory.h"
#include "dg/blas.h"
#include "grid.h"
#include "weights.cuh"
#include "interpolation.cuh"

#include "base_geometry.h"


namespace dg
{

///@cond
namespace detail
{

/**
 * @brief Normalize the given weights and compute the abscissas of the grid
 *
 * @param g The grid to be refined
 * @param weights the unnormalized weights
 *
 * @return The abscissas of the new grid
 */
template<class real_type>
thrust::host_vector<real_type> normalize_weights_and_compute_abscissas( const BasicGrid1d<real_type>& g, thrust::host_vector<real_type>& weights)
{
    //normalize weights
    unsigned Nx_new = weights.size()/g.n();
    for( unsigned i=0;i<weights.size(); i++)
        weights[i] *= (real_type)g.N()/(real_type)Nx_new;

    thrust::host_vector<real_type> boundaries(Nx_new+1), abs(g.n()*Nx_new);
    boundaries[0] = g.x0();
    for( unsigned i=0; i<Nx_new; i++)
    {
        boundaries[i+1] = boundaries[i] + g.lx()/(real_type)Nx_new/weights[g.n()*i];
        for( unsigned j=0; j<g.n(); j++)
        {
            abs[i*g.n()+j] =  (boundaries[i+1]+boundaries[i])/2. +
                (boundaries[i+1]-boundaries[i])/2.*g.dlt().abscissas()[j];
        }
    }
    return abs;
}

}//namespace detail
///@endcond

///@addtogroup generators
///@{

/**
* @brief Abstract base class for 1d grid refinement that increases the number of grid cells of a fixed basis grid
*/
template<class real_type>
struct aBasicRefinement1d
{
    /*! @brief Generate the grid transformation
     *
     * @param g_old The 1d grid to refine
     * @param weights A 1d vector of size N_new. These represent the Jacobian of the transformation \f[\frac{\partial \zeta}{\partial x} \f]. The new metric element has thus to be multiplied by weights^2 and the volume by 1/weights
     * @param abscissas A 1d vector of size N_new. These are the new abscissas \f$ x(\zeta) \f$ of the grid.
    */
    void generate( const BasicGrid1d<real_type>& g_old, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const
    {
        weights.resize( N_new(g_old.N(), g_old.bcx()));
        abscissas.resize( N_new(g_old.N(), g_old.bcx()));
        do_generate(g_old,weights,abscissas);
    }
    /*! @brief the new number of cells
     * @param N_old the old number of cells
     * @param bcx the boundary condition of the grid
     */
    unsigned N_new( unsigned N_old, bc bcx) const
    {
        return do_N_new(N_old, bcx);
    }
    virtual aBasicRefinement1d* clone()const=0;
    virtual ~aBasicRefinement1d(){}
    protected:
    aBasicRefinement1d(){}
    aBasicRefinement1d(const aBasicRefinement1d& src){}
    aBasicRefinement1d& operator=(const aBasicRefinement1d& src){
        return *this;
    }
    private:
    virtual void do_generate( const BasicGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const =0;
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const =0;
};

/**
* @brief No refinement
*/
template<class real_type>
struct BasicIdentityRefinement : public aBasicRefinement1d<real_type>
{
    BasicIdentityRefinement* clone()const{return new BasicIdentityRefinement();}
    private:
    virtual void do_generate( const BasicGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final{
        weights=dg::create::weights(g);
        abscissas=dg::create::abscissas(g);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final{
        return N_old;
    }
};

/**
 * @brief BasicLinear refinement consists of multiplying every cell in the grid by a factor
 */
template<class real_type>
struct BasicLinearRefinement : public aBasicRefinement1d<real_type>
{
    /**
     * @brief Refine every cell in the grid by an integer number of new cells
     * @param multiple multiply every cell
     */
    BasicLinearRefinement( unsigned multiple): m_(multiple){
        assert( multiple>= 1);
    }
    BasicLinearRefinement* clone()const{return new BasicLinearRefinement(*this);}
    private:
    unsigned m_;
    virtual void do_generate( const BasicGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final
    {
        thrust::host_vector< real_type> left( g.n()*g.N()*m_, 1);
        for( unsigned k=0; k<left.size(); k++)
            left[k] = (real_type)m_;
        weights = left;
        abscissas = dg::detail::normalize_weights_and_compute_abscissas( g, weights);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final{
        return N_old*m_;
    }
};

/**
 * @brief BasicEquidistant cell refinement around a given node
 */
template<class real_type>
struct BasicEquidistRefinement : public aBasicRefinement1d<real_type>
{
    /**
     * @brief Divide a number of cells left and right of a node into an equidistant number of new cells
     *
     * There will be 2*howmany refined cells (each containing 1+add_x equidistant cells) at the end except if a
     * corner node is given and the boundary condition is not periodic.
     * @param add_x number of additional cells in the cells around the node
     * @param node the node around which to refine. We count nodes from 0 (left corner) to N (right corner).
     * @param howmany  number of cells around a node to refine
     */
    BasicEquidistRefinement( unsigned add_x, unsigned node, unsigned howmany=1): add_x_(add_x), node_(node), howm_(howmany){ }
    BasicEquidistRefinement* clone()const{return new BasicEquidistRefinement(*this);}
    private:
    unsigned add_x_, node_, howm_;
    virtual void do_generate( const BasicGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final
    {
        if( add_x_ == 0 || howm_ == 0)
        {
            thrust::host_vector<real_type> w_( g.size(), 1);
            abscissas = dg::create::abscissas(g);
            weights = w_;
            return;
        }
        weights = equidist_ref( add_x_, node_, g.n(), g.N(), g.bcx(), howm_);
        abscissas = detail::normalize_weights_and_compute_abscissas( g, weights);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final
    {
        if( bcx == dg::PER) return N_old + 2*add_x_*howm_;
        return N_old + add_x_*howm_;
    }
    thrust::host_vector<real_type> equidist_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx, unsigned howmany) const
    {
        assert( howm_ <= N);
        assert( node_ <= N);
        if( node_ != 0 && node_ != N)
            assert( howm_ <= node_ && howm_ <= N-node_);
        if( add_x_ == 0 || howm_ == 0)
        {
            thrust::host_vector<real_type> w_( n*N, 1);
            return w_;
        }
        //there are add_x+1 finer cells per refined cell ...
        thrust::host_vector< real_type> left( n*N+n*add_x_*howm_, 1), right(left);
        for( unsigned i=0; i<(add_x_+1)*howm_; i++)//the original cell and the additional ones
            for( unsigned k=0; k<n; k++)
                left[i*n+k] = add_x_ + 1;
        //mirror left into right
        for( unsigned i=0; i<right.size(); i++)
            right[i] = left[ (left.size()-1)-i];
        thrust::host_vector< real_type> both( n*N+2*n*add_x_*howm_, 1);
        for( unsigned i=0; i<left.size(); i++)
            both[i] *= left[i];
        for( unsigned i=0; i<right.size(); i++)
            both[i+n*add_x_*howm_] *= right[i];
        if(      node_ == 0     && bcx != dg::PER) { return left; }
        else if( node_ == N && bcx != dg::PER) { return right; }
        else if((node_ == N || node_ == 0) && bcx == dg::PER) { return both; }
        else
        {
            thrust::host_vector<real_type> w_ = both;
            //now shift indices so that refinement is around node_s
            for( unsigned i=0; i<both.size(); i++)
                w_[((howm_*add_x_+node_)*n+i)%both.size()] = both[i];
            return w_;
        }
    }

};

/**
 * @brief The exponential refinement around a node
 */
template<class real_type>
struct BasicExponentialRefinement : public aBasicRefinement1d<real_type>
{
    /**
     * @brief Construct exponential refinement
     *
     * There will be two refined cells at the end except if a corner node is
     * given and the boundary condition is not periodic.
     * @param add_x number of additional cells in the cells node-1 and node. Every new additional cell divides the cell closest to the node in half.
     * @param node The cells node-1 and node will be refined. We count nodes from
     * 0 (left corner) to N (right corner).
     */
    BasicExponentialRefinement( unsigned add_x, unsigned node): add_x_(add_x), node_(node) {}
    BasicExponentialRefinement* clone()const{return new BasicExponentialRefinement(*this);}
    private:
    unsigned add_x_, node_;
    virtual void do_generate( const BasicGrid1d<real_type>& g, thrust::host_vector<real_type>& weights, thrust::host_vector<real_type>& abscissas) const override final
    {
        if( add_x_ == 0)
        {
            thrust::host_vector<real_type> w_( g.size(), 1);
            abscissas= dg::create::abscissas(g);
            weights = w_;
            return;
        }
        weights = exponential_ref( add_x_, node_, g.n(), g.N(), g.bcx());
        abscissas = detail::normalize_weights_and_compute_abscissas( g, weights);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const override final
    {
        if( bcx == dg::PER) return N_old + 2*add_x_;
        return N_old + add_x_;
    }
    thrust::host_vector<real_type> exponential_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx) const override final
    {
        if( add_x_ == 0)
        {
            thrust::host_vector<real_type> w_( n*N, 1);
            return w_;
        }
        assert( node_ <= N);
        //there are add_x_+1 finer cells per refined cell ...
        thrust::host_vector< real_type> left( n*N+n*add_x_, 1), right(left);
        for( unsigned k=0; k<n; k++)//the original cell and the additional ones
            left[k] = pow( 2, add_x_);
        for( unsigned i=0; i<add_x_; i++)
            for( unsigned k=0; k<n; k++)
                left[(i+1)*n+k] = pow( 2, add_x_-i);
        //mirror left into right
        for( unsigned i=0; i<right.size(); i++)
            right[i] = left[ (left.size()-1)-i];
        thrust::host_vector< real_type> both( n*N+2*n*add_x_, 1);
        for( unsigned i=0; i<left.size(); i++)
            both[i] *= left[i];
        for( unsigned i=0; i<right.size(); i++)
            both[i+n*add_x_] *= right[i];
        if(      node_ == 0     && bcx != dg::PER) { return left; }
        else if( node_ == N && bcx != dg::PER) { return right; }
        else if((node_ == N || node_ == 0) && bcx == dg::PER) { return both; }
        else
        {
            thrust::host_vector<real_type> w_ = both;
            //now shift indices so that refinement is around node_s
            for( unsigned i=0; i<both.size(); i++)
                w_[((add_x_+node_)*n+i)%both.size()] = both[i];
            return w_;
        }
    }
};

using aRefinement1d = aBasicRefinement1d<double>;
using IdentityRefinement = BasicIdentityRefinement<double>;
using LinearRefinement = BasicLinearRefinement<double>;
using EquidistRefinement = BasicEquidistRefinement<double>;
using ExponentialRefinement = BasicExponentialRefinement<double>;

///@}

/**
 * @brief Refined BasicCartesian grid
 * @ingroup geometry
 */
template<class real_type>
struct BasicCartesianRefinedGrid2d : public dg::aBasicGeometry2d<real_type>
{
    BasicCartesianRefinedGrid2d( const aBasicRefinement1d<real_type>& refX, const aBasicRefinement1d<real_type>& refY, real_type x0, real_type x1, real_type y0, real_type y1,
            unsigned n, unsigned Nx, unsigned Ny, bc bcx = dg::PER, bc bcy = dg::PER) : dg::aGeometry2d( x0, x1, y0, y1, n, refX.N_new(Nx, bcx), refY.N_new(Ny,bcy), bcx, bcy), refX_(refX), refY_(refY), w_(2), a_(2)
    {
        construct_weights_and_abscissas(n,Nx,Ny);
    }

    BasicCartesianRefinedGrid2d* clone()const{return new BasicCartesianRefinedGrid2d(*this);}
    private:
    ClonePtr<aBasicRefinement1d<real_type>> refX_, refY_;
    std::vector<thrust::host_vector<real_type> > w_, a_;
    void construct_weights_and_abscissas(unsigned n, unsigned Nx, unsigned Ny)
    {
        BasicGrid1d<real_type> gx( this->x0(), this->x1(), n, Nx, this->bcx());
        BasicGrid1d<real_type> gy( this->y0(), this->y1(), n, Ny, this->bcy());
        thrust::host_vector<real_type> wx, ax, wy, ay;
        refX_.get().generate( gx, wx, ax);
        refY_.get().generate( gy, wy, ay);
        w_[0].resize(this->size()), w_[1].resize(this->size());
        a_[0].resize(this->size()), a_[1].resize(this->size());
        //now make product space
        for( unsigned i=0; i<wy.size(); i++)
            for( unsigned j=0; j<wx.size(); j++)
            {
                w_[0][i*wx.size()+j] = wx[j];
                w_[1][i*wx.size()+j] = wy[i];
                a_[0][i*wx.size()+j] = ax[j];
                a_[1][i*wx.size()+j] = ay[i];
            }
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny)override final{
        aBasicTopology2d<real_type>::do_set(new_n,refX_.get().N_new(new_Nx,this->bcx()),refY_.get().N_new(new_Ny, this->bcy()));
        construct_weights_and_abscissas(new_n,new_Nx,new_Ny);
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final{
        SparseTensor<thrust::host_vector<real_type> > t(w_);
        dg::blas1::pointwiseDot( w_[0], w_[0], t.values()[0]);
        dg::blas1::pointwiseDot( w_[1], w_[1], t.values()[1]);
        t.idx(0,0)=0, t.idx(1,1)=1;
        return t;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const override final{
        SparseTensor<thrust::host_vector<real_type> > t(w_);
        t.idx(0,0)=0, t.idx(1,1)=1;
        return t;
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final{
        return a_;
    }
};

/**
 * @brief Refined BasicCartesian grid
 * @ingroup geometry
 */
template< class real_type>
struct BasicCartesianRefinedGrid3d : public dg::aBasicGeometry3d<real_type>
{
    BasicCartesianRefinedGrid3d( const aBasicRefinement1d<real_type>& refX, const aBasicRefinement1d<real_type>& refY, aBasicRefinement1d<real_type>& refZ, real_type x0, real_type x1, real_type y0, real_type y1, real_type z0, real_type z1,
            unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = dg::PER, bc bcy = dg::PER, bc bcz=dg::PER) : dg::aGeometry3d( x0, x1, y0, y1,z0,z1, n, refX.N_new(Nx, bcx), refY.N_new(Ny,bcy), refZ.N_new(Nz,bcz), bcx, bcy, bcz), refX_(refX), refY_(refY), refZ_(refZ), w_(3), a_(3)
    {
        construct_weights_and_abscissas(n, Nx, Ny,Nz);
    }

    BasicCartesianRefinedGrid3d* clone()const{return new BasicCartesianRefinedGrid3d(*this);}
    private:
    ClonePtr<aBasicRefinement1d<real_type>> refX_, refY_, refZ_;
    std::vector<thrust::host_vector<real_type> > w_, a_;
    void construct_weights_and_abscissas(unsigned n, unsigned Nx, unsigned Ny,unsigned Nz)
    {
        BasicGrid1d<real_type> gx( this->x0(), this->x1(), n, Nx, this->bcx());
        BasicGrid1d<real_type> gy( this->y0(), this->y1(), n, Ny, this->bcy());
        BasicGrid1d<real_type> gz( this->y0(), this->y1(), 1, Nz, this->bcz());
        thrust::host_vector<real_type> w[3], a[3];
        refX_.get().generate( gx, w[0], a[0]);
        refY_.get().generate( gy, w[1], a[1]);
        refZ_.get().generate( gz, w[2], a[2]);
        w_[0].resize(this->size()), w_[1].resize(this->size()), w_[2].resize(this->size());
        a_[0].resize(this->size()), a_[1].resize(this->size()), a_[2].resize(this->size());
        //now make product space
    for( unsigned s=0; s<w[2].size(); s++)
        for( unsigned i=0; i<w[1].size(); i++)
            for( unsigned j=0; j<w[0].size(); j++)
            {
                w_[0][(s*w[1].size()+i)*w[0].size()+j] = w[0][j];
                w_[1][(s*w[1].size()+i)*w[0].size()+j] = w[1][i];
                w_[2][(s*w[1].size()+i)*w[0].size()+j] = w[2][s];
                a_[0][(s*w[1].size()+i)*w[0].size()+j] = a[0][j];
                a_[1][(s*w[1].size()+i)*w[0].size()+j] = a[1][i];
                a_[2][(s*w[1].size()+i)*w[0].size()+j] = a[1][s];
            }
    }
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz) override final{
        aBasicTopology3d<real_type>::do_set(new_n,refX_.get().N_new(new_Nx, this->bcx()),refY_.get().N_new(new_Ny,this->bcy()), refZ_.get().N_new(new_Nz,this->bcz()));
        construct_weights_and_abscissas(new_n, new_Nx, new_Ny, new_Nz);
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_metric()const override final {
        SparseTensor<thrust::host_vector<real_type> > t(w_);
        dg::blas1::pointwiseDot( w_[0], w_[0], t.values()[0]);
        dg::blas1::pointwiseDot( w_[1], w_[1], t.values()[1]);
        dg::blas1::pointwiseDot( w_[2], w_[2], t.values()[2]);
        t.idx(0,0)=0, t.idx(1,1)=1, t.idx(2,2)=2;
        return t;
    }
    virtual SparseTensor<thrust::host_vector<real_type> > do_compute_jacobian()const override final{
        SparseTensor<thrust::host_vector<real_type> > t(w_);
        t.idx(0,0)=0, t.idx(1,1)=1, t.idx(2,2)=2;
        return t;
    }
    virtual std::vector<thrust::host_vector<real_type> > do_compute_map()const override final {
        return a_;
    }
};

using CartesianRefinedGrid2d = BasicCartesianRefinedGrid2d<double>;
using CartesianRefinedGrid3d = BasicCartesianRefinedGrid3d<double>;

}//namespace dg
