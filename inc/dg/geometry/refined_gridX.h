#pragma once

#include "cusp/transpose.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/evaluationX.cuh"
#include "dg/backend/weightsX.cuh"
#include "dg/backend/gridX.h"
#include "base_geometryX.h"
#include "refined_grid.h"


namespace dg
{
///@addtogroup generators
///@{
/**
* @brief Abstract base class for 2d grid refinement that increases the number of grid cells of a fixed basis grid
*/
struct aRefinementX2d
{
    /*! @brief Generate the grid transformation
     *  
     * @param g_old The 1d grid to refine
     * @param weightsX A 2d vector of size nx_new()*ny_new(). These represent the Jacobian of the transformation \f[\frac{\partial \zeta}{\partial x} \f]. The new metric element has thus to be multiplied by weights^2 and the volume by 1/weights
     * @param weightsY A 2d vector of size nx_new()*ny_new(). These represent the Jacobian of the transformation \f[\frac{\partial \zeta}{\partial x} \f]. The new metric element has thus to be multiplied by weights^2 and the volume by 1/weights
     * @param abscissasX A 2d vector of size nx_new()*ny_new(). These are the new abscissas \f$ x(\zeta) \f$ of the grid. 
     * @param abscissasY A 2d vector of size nx_new()*ny_new(). These are the new abscissas \f$ x(\zeta) \f$ of the grid. 
    */
    void generate( const GridX2d& g_old, thrust::host_vector<double>& weightsX, thrust::host_vector<double>& weightsY, thrust::host_vector<double>& abscissasX, thrust::host_vector<double>& abscissasY) const
    {
        thrust::host_vector<double> wx, ax, wy, ay;
        Grid1d gx( g_old.x0(), g_old.x1(), g_old.n(), g_old.Nx(), g_old.bcx());
        do_generateX(gx,g_old.inner_Nx(), wx,ax);
        GridX1d gy( g_old.y0(), g_old.y1(), g_old.fy(), g_old.n(), g_old.Ny(), g_old.bcy());
        do_generateY(gy,wy,ay);
        unsigned size=wx.size()*wy.size();
        weightsX.resize(size), weightsY.resize(size); 
        abscissasX.resize(size), abscissasY.resize(size);
        //now make product space
        for( unsigned i=0; i<wy.size(); i++)
            for( unsigned j=0; j<wx.size(); j++)
            {
                weightsX[i*wx.size()+j] = wx[j];
                weightsY[i*wx.size()+j] = wy[i];
                abscissasX[i*wx.size()+j] = ax[j];
                abscissasY[i*wx.size()+j] = ay[i];
            }
    }

    /*! @brief the new number of cells
     * @param Nx_old the old number of cells
     * @param fx_old Factor to partition x-axis
     */
    unsigned nx_new( unsigned Nx_old, double fx_old) const
    {
        return do_Nx_new(Nx_old, fx_old);
    }
    unsigned ny_new( unsigned Ny_old, double fy_old) const
    {
        return do_Ny_new(Ny_old, fy_old);
    }
    double fx_new( unsigned Nx_old, double fx_old) const
    {
        return do_fx_new(Nx_old, fx_old);
    }
    double fy_new( unsigned Ny_old, double fy_old) const
    {
        return do_fy_new(Ny_old, fy_old);
    }
    virtual aRefinementX2d* clone()const=0;
    virtual ~aRefinementX2d(){}
    protected:
    aRefinementX2d(){}
    aRefinementX2d(const aRefinementX2d& src){}
    aRefinementX2d& operator=(const aRefinementX2d& src){
        return *this;
    }
    private:
    virtual void do_generateX( const Grid1d& gx, unsigned nodeXX, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const =0;
    virtual void do_generateY( const GridX1d& gy, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const =0;
    virtual unsigned do_Nx_new( unsigned Nx_old, double fx) const =0;
    virtual unsigned do_Ny_new( unsigned Ny_old, double fy) const =0;
    virtual double do_fx_new( unsigned Nx_old, double fx) const =0;
    virtual double do_fy_new( unsigned Ny_old, double fy) const =0;
};

/**
* @brief No refinement
*/
struct IdentityXRefinement : public aRefinementX2d
{
    IdentityXRefinement* clone()const{return new IdentityXRefinement(*this);}
    private:
    virtual void do_generateX( const Grid1d& g, unsigned nodeXX, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const {
        weights=dg::create::weights(g);
        abscissas=dg::create::abscissas(g);
    }
    virtual void do_generateY( const GridX1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const {
        weights=dg::create::weights(g);
        abscissas=dg::create::abscissas(g);
    }
    virtual unsigned do_Nx_new( unsigned Nx_old, double fx) const { return Nx_old; }
    virtual unsigned do_Ny_new( unsigned Ny_old, double fy) const { return Ny_old; }
    virtual double do_fx_new( unsigned Nx_old, double fx) const { return fx; }
    virtual double do_fy_new( unsigned Ny_old, double fy) const { return fy; }
};
/**
 * @brief Equidistant cell refinement around the X-point
 */
struct EquidistXRefinement : public aRefinementX2d
{
    EquidistXRefinement( unsigned add_x, unsigned add_y, unsigned howmanyX=1, unsigned howmanyY=1): add_x_(add_x), howm_x_(howmanyX), add_y_(add_y), howm_y_(howmanyY){ }
    EquidistXRefinement* clone()const{return new EquidistXRefinement(*this);}
    private:
    unsigned add_x_, howm_x_, add_y_, howm_y_;
    virtual void do_generateX( const Grid1d& gx, unsigned nodeXX, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const 
    {
        EquidistRefinement equi(add_x_, nodeXX, howm_x_);
        equi.generate(gx,weights,abscissas);
    }
    virtual void do_generateY( const GridX1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const 
    {
        EquidistRefinement equi0(add_y_,0,howm_y_);
        if( add_y_ == 0 || howm_y_ == 0) { 
            equi0.generate(g.grid(), weights, abscissas);
            return;
        }
        if( g.f() == 0) { 
            equi0.generate( Grid1d( g.x0(), g.x1(), g.n(), g.N(), dg::PER), weights, abscissas); 
            return;
        }
        thrust::host_vector<double> w1, w2, w3;
        thrust::host_vector<double> a1, a2, a3;
        unsigned node1 = g.outer_N();
        EquidistRefinement equi1(add_y_, node1, howm_y_);
        equi1.generate(Grid1d(g.x0(),g.x1(), g.n(), g.outer_N(), dg::DIR), w1,a1); //left side
        equi0.generate( Grid1d(g.x0(),g.x1(), g.n(), g.inner_N(), dg::PER), w2,a2);//inner side
        equi0.generate( Grid1d(g.x0(),g.x1(), g.n(), g.outer_N(), dg::DIR), w3,a3);//right side
        //now combine unnormalized weights
        thrust::host_vector<double> wtot( w1.size() + w2.size() + w3.size());
        for( unsigned i=0; i<w1.size() ; i++)
            wtot[i] = w1[i];
        for( unsigned i=0; i<w2.size(); i++)
            wtot[w1.size()+i] = w2[i];
        for( unsigned i=0; i<w3.size(); i++)
            wtot[w1.size()+w2.size()+i] = w3[i];
        weights = wtot;
        abscissas = detail::normalize_weights_and_compute_abscissas( g.grid(), weights);
    }
    virtual unsigned do_Ny_new( unsigned Ny, double fy) const {
        if( fy==0 ) return Ny + 2*add_y_; 
        return Ny + 4*add_y_;
    }
    virtual unsigned do_Nx_new( unsigned Nx, double fx) const {
        if( fx==0 ) return Nx + add_x_; 
        return Nx + 2*add_x_;
    }
    virtual double do_fx_new( unsigned Nx, double fx) const {
        if( fx==0 ) return 0; 
        return (fx*(double)Nx + (double)add_x_)/(double)(Nx+2.*add_x_);
    }
    virtual double do_fy_new( unsigned Ny, double fy) const { 
        if( fy==0 ) return 0; 
        return (fy*(double)Ny + (double)add_y_)/(double)(Ny+4.*add_y_);
    }
};

/**
 * @brief The exponential refinement around the X-point
 */
struct ExponentialXRefinement : public aRefinementX2d
{
    ExponentialXRefinement( unsigned add_x, unsigned add_y, unsigned howmanyX=1, unsigned howmanyY=1): add_x_(add_x), howm_x_(howmanyX), add_y_(add_y), howm_y_(howmanyY){ }
    ExponentialXRefinement* clone()const{return new ExponentialXRefinement(*this);}
    private:
    unsigned add_x_, howm_x_, add_y_, howm_y_;
    virtual void do_generateX( const Grid1d& gx, unsigned nodeXX, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const 
    {
        EquidistRefinement equi(add_x_, nodeXX, howm_x_);
        equi.generate(gx,weights,abscissas);
    }
    virtual void do_generateY( const GridX1d& g, thrust::host_vector<double>& weights, thrust::host_vector<double>& abscissas) const 
    {
        ExponentialRefinement expo0( add_x_, 0);
        if( add_y_ == 0) { return expo0.generate( g.grid(), weights, abscissas); }
        if( g.f()  == 0) { return expo0.generate( Grid1d( g.x0(), g.x1(), g.n(), g.N(), dg::PER), weights, abscissas); }
        thrust::host_vector<double> w1, w2, w3;
        thrust::host_vector<double> a1, a2, a3;
        unsigned node1 = g.outer_N();
        ExponentialRefinement expo1( add_y_, node1);
        expo1.generate( Grid1d(g.x0(),g.x1(),g.n(), g.outer_N(), dg::DIR), w1,a1); //left side
        expo0.generate( Grid1d(g.x0(),g.x1(),g.n(), g.inner_N(), dg::PER), w2,a2); //inner side
        expo0.generate( Grid1d(g.x0(),g.x1(),g.n(), g.outer_N(), dg::DIR), w3,a3); //right side
        //now combine unnormalized weights
        thrust::host_vector<double> wtot( g.size() + 4*g.n()*add_x_);
        for( unsigned i=0; i<w1.size() ; i++)
            wtot[i] = w1[i];
        for( unsigned i=0; i<w2.size(); i++)
            wtot[w1.size()+i] = w2[i];
        for( unsigned i=0; i<w3.size(); i++)
            wtot[w1.size()+w2.size()+i] = w3[i];
        weights = wtot;

        abscissas = detail::normalize_weights_and_compute_abscissas( g.grid(), weights);
    }
    virtual unsigned do_Ny_new( unsigned Ny, double fy) const {
        if( fy==0 ) return Ny + 2*add_y_; 
        return Ny + 4*add_y_;
    }
    virtual unsigned do_Nx_new( unsigned Nx, double fx) const {
        if( fx==0 ) return Nx + add_x_; 
        return Nx + 2*add_x_;
    }
    virtual double do_fx_new( unsigned Nx, double fx) const {
        if( fx==0 ) return 0; 
        return (fx*(double)Nx + (double)add_x_)/(double)(Nx+2.*add_x_);
    }
    virtual double do_fy_new( unsigned Ny, double fy) const { 
        if( fy==0 ) return 0; 
        return (fy*(double)Ny + (double)add_y_)/(double)(Ny+4.*add_y_);
    }
};
///@}


/**
 * @brief Refined X-point grid 
 * @ingroup geometry
 */
struct CartesianRefinedGridX2d : public dg::aGeometryX2d
{
    CartesianRefinedGridX2d( const aRefinementX2d& ref,
            double x0, double x1, double y0, double y1, 
            double fx, double fy, 
            unsigned n, unsigned Nx, unsigned Ny, 
            bc bcx = dg::PER, bc bcy = dg::PER) : dg::aGeometryX2d( x0, x1, y0, y1, 
                ref.fx_new(Nx, fx), ref.fy_new(Ny, fy), n, ref.nx_new(Nx, fx), ref.ny_new(Ny, fy), bcx, bcy), w_(2), abs_(2)
    {
        GridX2d g( x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy);
        ref.generate(g,w_[0],w_[1],abs_[0],abs_[1]);
    }

    CartesianRefinedGridX2d* clone()const{return new CartesianRefinedGridX2d(*this);}
    private:
    std::vector<thrust::host_vector<double> > w_,abs_;
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
        return abs_;
    }
};

/**
 * @brief Refined X-point grid 
 * @ingroup geometry
 */
struct CartesianRefinedGridX3d : public dg::aGeometryX3d
{
    CartesianRefinedGridX3d( const aRefinementX2d& ref,
            double x0, double x1, double y0, double y1, double z0, double z1,
            double fx, double fy, 
            unsigned n, unsigned Nx, unsigned Ny, unsigned Nz,
            bc bcx = dg::PER, bc bcy = dg::PER, bc bcz = dg::PER) : dg::aGeometryX3d( 
                x0, x1, y0, y1,z0,z1,
                ref.fx_new(Nx, fx), ref.fy_new(Ny, fy), 
                n, ref.nx_new(Nx, fx), ref.ny_new(Ny, fy), Nz, 
                bcx, bcy, bcz), w_(2), abs_(2)
    {
        GridX2d g( x0,x1,y0,y1,fx,fy,n,Nx,Ny,bcx,bcy);
        ref.generate(g,w_[0],w_[1],abs_[0],abs_[1]);
        //lift to 3d
        w_[0].resize(size()), w_[1].resize(size()), abs_[0].resize(size()), abs_[1].resize(size());
        unsigned size2d=this->n()*this->n()*this->Nx()*this->Ny();
        for( unsigned i=1; i<Nz; i++)
            for(unsigned k=0; k<size2d; k++)
            {
                w_[0][i*size2d+k]=w_[0][k];
                w_[1][i*size2d+k]=w_[1][k];
                abs_[0][i*size2d+k]=abs_[0][k];
                abs_[1][i*size2d+k]=abs_[1][k];
            }
    }

    CartesianRefinedGridX3d* clone()const{return new CartesianRefinedGridX3d(*this);}
    private:
    std::vector<thrust::host_vector<double> > w_,abs_;
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
        return abs_;
    }
};


}//namespace dg
