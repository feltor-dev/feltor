#pragma once

#include "cusp/transpose.h"
#include "dg/backend/memory.h"
#include "dg/backend/grid.h"
#include "dg/backend/weights.cuh"
#include "dg/backend/interpolation.cuh"
#include "dg/blas.h"

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
thrust::host_vector<double> normalize_weights_and_compute_abscissas( const Grid1d& g, thrust::host_vector<double>& weights)
{
    //normalize weights
    unsigned Nx_new = weights.size()/g.n();
    for( unsigned i=0;i<weights.size(); i++)
        weights[i] *= (double)g.N()/(double)Nx_new;

    thrust::host_vector<double> boundaries(Nx_new+1), abs(g.n()*Nx_new);
    boundaries[0] = g.x0();
    for( unsigned i=0; i<Nx_new; i++)
    {
        boundaries[i+1] = boundaries[i] + g.lx()/(double)Nx_new/weights[g.n()*i];
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
struct aRefinement1d
{
    /*! @brief Generate the grid transformation
     *  
     * @param g_old The 1d grid to refine
     * @param weights A 1d vector of size N_new. These represent the Jacobian of the transformation \f[\frac{\partial \zeta}{\partial x} \f]. The new metric element has thus to be multiplied by weights^2 and the volume by 1/weights
     * @param abscissas A 1d vector of size N_new. These are the new abscissas \f$ x(\zeta) \f$ of the grid. 
    */
    void generate( const Grid1d& g_old, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const
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
    virtual aRefinement1d* clone()const=0;
    virtual ~aRefinement1d(){}
    protected:
    aRefinement1d(){}
    aRefinement1d(const aRefinement1d& src){}
    aRefinement1d& operator=(const aRefinement1d& src){
        return *this;
    }
    private:
    virtual void do_generate( const Grid1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const =0;
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const =0;
};

/**
* @brief No refinement
*/
struct IdentityRefinement : public aRefinement1d
{
    IdentityRefinement* clone()const{return new IdentityRefinement();}
    private:
    virtual void do_generate( const Grid1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const {
        weights=dg::create::weights(g);
        abscissas=dg::create::abscissas(g);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const {
        return N_old;
    }
};

/**
 * @brief Linear refinement consists of multiplying every cell in the grid by a factor
 */
struct LinearRefinement : public aRefinement1d
{
    /**
     * @brief Refine every cell in the grid by an integer number of new cells
     * @param multiple multiply every cell
     */
    LinearRefinement( unsigned multiple): m_(multiple){
        assert( multiple>= 1);
    }
    LinearRefinement* clone()const{return new LinearRefinement(*this);}
    private:
    unsigned m_;
    virtual void do_generate( const Grid1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const 
    {
        thrust::host_vector< double> left( g.n()*g.N()*m_, 1);
        for( unsigned k=0; k<left.size(); k++)
            left[k] = (double)m_;
        weights = left;
        abscissas = dg::detail::normalize_weights_and_compute_abscissas( g, weights);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const {
        return N_old*m_;
    }
};

/**
 * @brief Equidistant cell refinement around a given node 
 */
struct EquidistRefinement : public aRefinement1d
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
    EquidistRefinement( unsigned add_x, unsigned node, unsigned howmany=1): add_x_(add_x), node_(node), howm_(howmany){ }
    EquidistRefinement* clone()const{return new EquidistRefinement(*this);}
    private:
    unsigned add_x_, node_, howm_;
    virtual void do_generate( const Grid1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const 
    {
        if( add_x_ == 0 || howm_ == 0)
        {
            thrust::host_vector<double> w_( g.size(), 1);
            abscissas = dg::create::abscissas(g);
            weights = w_; 
            return;
        }
        weights = equidist_ref( add_x_, node_, g.n(), g.N(), g.bcx(), howm_);
        abscissas = detail::normalize_weights_and_compute_abscissas( g, weights);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const
    {
        if( bcx == dg::PER) return N_old + 2*add_x_*howm_; 
        return N_old + add_x_*howm_;
    }
    thrust::host_vector<double> equidist_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx, unsigned howmany) const
    {
        assert( howm_ <= N);
        assert( node_ <= N);
        if( node_ != 0 && node_ != N) 
            assert( howm_ <= node_ && howm_ <= N-node_);
        if( add_x_ == 0 || howm_ == 0)
        {
            thrust::host_vector<double> w_( n*N, 1);
            return w_;
        }
        //there are add_x+1 finer cells per refined cell ...
        thrust::host_vector< double> left( n*N+n*add_x_*howm_, 1), right(left);
        for( unsigned i=0; i<(add_x_+1)*howm_; i++)//the original cell and the additional ones
            for( unsigned k=0; k<n; k++)
                left[i*n+k] = add_x_ + 1;
        //mirror left into right
        for( unsigned i=0; i<right.size(); i++)
            right[i] = left[ (left.size()-1)-i];
        thrust::host_vector< double> both( n*N+2*n*add_x_*howm_, 1);
        for( unsigned i=0; i<left.size(); i++)
            both[i] *= left[i];
        for( unsigned i=0; i<right.size(); i++)
            both[i+n*add_x_*howm_] *= right[i];
        if(      node_ == 0     && bcx != dg::PER) { return left; }
        else if( node_ == N && bcx != dg::PER) { return right; }
        else if((node_ == N || node_ == 0) && bcx == dg::PER) { return both; }
        else 
        {
            thrust::host_vector<double> w_ = both;
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
struct ExponentialRefinement : public aRefinement1d
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
    ExponentialRefinement( unsigned add_x, unsigned node): add_x_(add_x), node_(node) {}
    ExponentialRefinement* clone()const{return new ExponentialRefinement(*this);}
    private:
    unsigned add_x_, node_;
    virtual void do_generate( const Grid1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const 
    {
        if( add_x_ == 0)
        {
            thrust::host_vector<double> w_( g.size(), 1);
            abscissas= dg::create::abscissas(g);
            weights = w_; 
            return;
        }
        weights = exponential_ref( add_x_, node_, g.n(), g.N(), g.bcx());
        abscissas = detail::normalize_weights_and_compute_abscissas( g, weights);
    }
    virtual unsigned do_N_new( unsigned N_old, bc bcx) const
    {
        if( bcx == dg::PER) return N_old + 2*add_x_; 
        return N_old + add_x_;
    }
    thrust::host_vector<double> exponential_ref( unsigned add_x, unsigned node, unsigned n, unsigned N, dg::bc bcx) const
    {
        if( add_x_ == 0)
        {
            thrust::host_vector<double> w_( n*N, 1);
            return w_;
        }
        assert( node_ <= N);
        //there are add_x_+1 finer cells per refined cell ...
        thrust::host_vector< double> left( n*N+n*add_x_, 1), right(left);
        for( unsigned k=0; k<n; k++)//the original cell and the additional ones
            left[k] = pow( 2, add_x_);
        for( unsigned i=0; i<add_x_; i++) 
            for( unsigned k=0; k<n; k++)
                left[(i+1)*n+k] = pow( 2, add_x_-i);
        //mirror left into right
        for( unsigned i=0; i<right.size(); i++)
            right[i] = left[ (left.size()-1)-i];
        thrust::host_vector< double> both( n*N+2*n*add_x_, 1);
        for( unsigned i=0; i<left.size(); i++)
            both[i] *= left[i];
        for( unsigned i=0; i<right.size(); i++)
            both[i+n*add_x_] *= right[i];
        if(      node_ == 0     && bcx != dg::PER) { return left; }
        else if( node_ == N && bcx != dg::PER) { return right; }
        else if((node_ == N || node_ == 0) && bcx == dg::PER) { return both; }
        else 
        {
            thrust::host_vector<double> w_ = both;
            //now shift indices so that refinement is around node_s
            for( unsigned i=0; i<both.size(); i++)
                w_[((add_x_+node_)*n+i)%both.size()] = both[i];
            return w_;
        }
    }
};

///@}

/**
 * @brief Refined Cartesian grid 
 * @ingroup geometry
 */
struct CartesianRefinedGrid2d : public dg::aGeometry2d
{
    CartesianRefinedGrid2d( const aRefinement1d& refX, const aRefinement1d& refY, double x0, double x1, double y0, double y1, 
            unsigned n, unsigned Nx, unsigned Ny, bc bcx = dg::PER, bc bcy = dg::PER) : dg::aGeometry2d( x0, x1, y0, y1, n, refX.N_new(Nx, bcx), refY.N_new(Ny,bcy), bcx, bcy), refX_(refX), refY_(refY), w_(2), a_(2)
    {
        construct_weights_and_abscissas(n,Nx,Ny);
    }

    CartesianRefinedGrid2d* clone()const{return new CartesianRefinedGrid2d(*this);}
    private:
    Handle<aRefinement1d> refX_, refY_;
    std::vector<thrust::host_vector<double> > w_, a_;
    void construct_weights_and_abscissas(unsigned n, unsigned Nx, unsigned Ny)
    { 
        Grid1d gx( x0(), x1(), n, Nx, bcx());
        Grid1d gy( y0(), y1(), n, Ny, bcy());
        thrust::host_vector<double> wx, ax, wy, ay;
        refX_.get().generate( gx, wx, ax);
        refY_.get().generate( gy, wy, ay);
        w_[0].resize(size()), w_[1].resize(size()); 
        a_[0].resize(size()), a_[1].resize(size());
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
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny){
        aTopology2d::do_set(new_n,refX_.get().N_new(new_Nx,bcx()),refY_.get().N_new(new_Ny, bcy()));
        construct_weights_and_abscissas(new_n,new_Nx,new_Ny);
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric()const {
        SparseTensor<thrust::host_vector<double> > t(w_);
        dg::blas1::pointwiseDot( w_[0], w_[0], t.value(0));
        dg::blas1::pointwiseDot( w_[1], w_[1], t.value(1));
        t.idx(0,0)=0, t.idx(1,1)=1; 
        return t;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian()const {
        SparseTensor<thrust::host_vector<double> > t(w_);
        t.idx(0,0)=0, t.idx(1,1)=1; 
        return t;
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{
        return a_;
    }
};

/**
 * @brief Refined Cartesian grid 
 * @ingroup geometry
 */
struct CartesianRefinedGrid3d : public dg::aGeometry3d
{
    CartesianRefinedGrid3d( const aRefinement1d& refX, const aRefinement1d& refY, aRefinement1d& refZ, double x0, double x1, double y0, double y1, double z0, double z1,
            unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, bc bcx = dg::PER, bc bcy = dg::PER, bc bcz=dg::PER) : dg::aGeometry3d( x0, x1, y0, y1,z0,z1, n, refX.N_new(Nx, bcx), refY.N_new(Ny,bcy), refZ.N_new(Nz,bcz), bcx, bcy, bcz), refX_(refX), refY_(refY), refZ_(refZ), w_(3), a_(3)
    {
        construct_weights_and_abscissas(n, Nx, Ny,Nz);
    }

    CartesianRefinedGrid3d* clone()const{return new CartesianRefinedGrid3d(*this);}
    private:
    Handle<aRefinement1d> refX_, refY_, refZ_;
    std::vector<thrust::host_vector<double> > w_, a_;
    void construct_weights_and_abscissas(unsigned n, unsigned Nx, unsigned Ny,unsigned Nz)
    { 
        Grid1d gx( x0(), x1(), n, Nx, bcx());
        Grid1d gy( y0(), y1(), n, Ny, bcy());
        Grid1d gz( y0(), y1(), 1, Nz, bcz());
        thrust::host_vector<double> w[3], a[3];
        refX_.get().generate( gx, w[0], a[0]);
        refY_.get().generate( gy, w[1], a[1]);
        refZ_.get().generate( gz, w[2], a[2]);
        w_[0].resize(size()), w_[1].resize(size()), w_[2].resize(size()); 
        a_[0].resize(size()), a_[1].resize(size()), a_[2].resize(size()); 
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
    virtual void do_set(unsigned new_n, unsigned new_Nx, unsigned new_Ny, unsigned new_Nz){
        aTopology3d::do_set(new_n,refX_.get().N_new(new_Nx, bcx()),refY_.get().N_new(new_Ny,bcy()), refZ_.get().N_new(new_Nz,bcz()));
        construct_weights_and_abscissas(new_n, new_Nx, new_Ny, new_Nz);
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric()const {
        SparseTensor<thrust::host_vector<double> > t(w_);
        dg::blas1::pointwiseDot( w_[0], w_[0], t.value(0));
        dg::blas1::pointwiseDot( w_[1], w_[1], t.value(1));
        dg::blas1::pointwiseDot( w_[2], w_[2], t.value(2));
        t.idx(0,0)=0, t.idx(1,1)=1, t.idx(2,2)=2; 
        return t;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian()const {
        SparseTensor<thrust::host_vector<double> > t(w_);
        t.idx(0,0)=0, t.idx(1,1)=1, t.idx(2,2)=2; 
        return t;
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{
        return a_;
    }
};

}//namespace dg
