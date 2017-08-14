#pragma once

#include "generatorX.h"
#include "refined_gridX.h"

namespace dg
{
///@addtogroup geometry
///@{

/**
 * @brief A three-dimensional grid based on curvilinear coordinates
 * 
 * The base coordinate system is the cylindrical coordinate system R,Z,phi
 */
struct CurvilinearProductRefinedGridX3d : public dg::aGeometryX3d
{
    /*!@brief Constructor
    
     * the coordinates of the computational space are called x,y,z
     * @param ref a X-point refinement
     * @param generator must generate a grid
     * @param n number of %Gaussian nodes in x and y
     * @param fx
     * @param fy
     * @param Nx number of cells in x
     * @param Ny number of cells in y 
     * @param Nz  number of cells z
     * @param bcx boundary condition in x
     * @param bcy boundary condition in y
     * @param bcz boundary condition in z
     */
    CurvilinearProductRefinedGridX3d( const aRefinementX2d& ref, const aGeneratorX2d& generator, 
        double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, double fx, double fy, bc bcx=dg::DIR, bc bcy=dg::PER, bc bcz=dg::PER):
        dg::aGeometryX3d( generator.zeta0(), generator.zeta1(), generator.eta0(), generator.eta1(), 0., 2.*M_PI, ref.fx_new(Nx,fx),ref.fy_new(Ny,fy),n, ref.nx_new(Nx,fx), ref.ny_new(Ny,fy), Nz, bcx, bcy, bcz), map_(3)
    { 
        handle_ = generator;
        ref_=ref;
        constructPerp( fx,fy,n, Nx, Ny);
        constructParallel(Nz);
    }

    const aGeneratorX2d & generator() const{return handle_.get();}
    virtual CurvilinearProductRefinedGridX3d* clone()const{return new CurvilinearProductRefinedGridX3d(*this);}
    private:
    //construct phi and lift rest to 3d
    void constructParallel(unsigned Nz)
    {
        map_[2]=dg::evaluate(dg::cooZ3d, *this);
        unsigned size = this->size();
        unsigned size2d = this->n()*this->n()*this->Nx()*this->Ny();
        //resize for 3d values
        for( unsigned r=0; r<4;r++)
            jac_.value(r).resize(size);
        map_[0].resize(size);
        map_[1].resize(size);
        //lift to 3D grid
        for( unsigned k=1; k<Nz; k++)
            for( unsigned i=0; i<size2d; i++)
            {
                for(unsigned r=0; r<4; r++)
                    jac_.value(r)[k*size2d+i] = jac_.value(r)[(k-1)*size2d+i];
                map_[0][k*size2d+i] = map_[0][(k-1)*size2d+i];
                map_[1][k*size2d+i] = map_[1][(k-1)*size2d+i];
            }
    }
    //construct 2d plane
    void constructPerp( double fx, double fy, unsigned n, unsigned Nx, unsigned Ny)
    {
        std::vector<thrust::host_vector<double> > w(2),abs(2);
        GridX2d g( x0(),x1(),y0(),y1(),fx,fy,n,Nx,Ny,bcx,bcy);
        ref.generate(g,w[0],w[1],abs[0],abs[1]);
        thrust::host_vector<double> x_vec(n()*Nx()), y_vec(n()*Ny());
        for( unsigned i=0; i<x_vec.size(); i++)
            x_vec[i] = abs[0][i];
        for( unsigned i=0; i<y_vec.size(); i++)
            x_vec[i] = abs[1][i*x_vec.size()];
        handle_.get().generate( x_vec, y_vec, n()*outer_Ny(), n()*(inner_Ny()+outer_Ny()), map_[0], map_[1], jac_.value(0), jac_.value(1), jac_.value(2), jac_.value(3));
        //multiply by weights
        dg::blas1::pointwiseDot( jac.value(0), w[0], jac.value(0));
        dg::blas1::pointwiseDot( jac.value(1), w[0], jac.value(1));
        dg::blas1::pointwiseDot( jac.value(2), w[1], jac.value(2));
        dg::blas1::pointwiseDot( jac.value(3), w[1], jac.value(3));
        jac_.idx(0,0) = 0, jac_.idx(0,1) = 1, jac_.idx(1,0)=2, jac_.idx(1,1) = 3;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian( ) const {
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric( ) const
    {
        thrust::host_vector<double> tempxx( size()), tempxy(size()), tempyy(size()), temppp(size());
        for( unsigned i=0; i<size(); i++)
        {
            tempxx[i] = (jac_.value(0,0)[i]*jac_.value(0,0)[i]+jac_.value(0,1)[i]*jac_.value(0,1)[i]);
            tempxy[i] = (jac_.value(0,0)[i]*jac_.value(1,0)[i]+jac_.value(0,1)[i]*jac_.value(1,1)[i]);
            tempyy[i] = (jac_.value(1,0)[i]*jac_.value(1,0)[i]+jac_.value(1,1)[i]*jac_.value(1,1)[i]);
            temppp[i] = 1./map_[2][i]/map_[2][i]; //1/R^2
        }
        SparseTensor<thrust::host_vector<double> > metric;
        metric.idx(0,0) = 0; metric.value(0) = tempxx;
        metric.idx(1,1) = 1; metric.value(1) = tempyy;
        metric.idx(2,2) = 2; metric.value(2) = temppp;
        if( !handle_.get().isOrthogonal())
        {
            metric.idx(0,1) = metric.idx(1,0) = 3; 
            metric.value(3) = tempxy;
        }
        return metric;
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{return map_;}
    std::vector<thrust::host_vector<double> > map_;
    SparseTensor<thrust::host_vector<double> > jac_;
    dg::Handle<aGeneratorX2d> handle_;
    dg::Handle<aRefinementX2d> ref_;
};

/**
 * @brief A two-dimensional grid based on curvilinear coordinates
 */
struct CurvilinearRefinedGridX2d : public dg::aGeometryX2d
{
    /*!@brief Constructor
    
     * @param generator must generate an orthogonal grid (class takes ownership of the pointer)
     * @param ref a X-point refinement
     * @param fx
     * @param fy
     * @param n number of polynomial coefficients
     * @param Nx number of cells in first coordinate
     * @param Ny number of cells in second coordinate
     * @param bcx boundary condition in first coordinate
     * @param bcy boundary condition in second coordinate
     */
    CurvilinearRefinedGridX2d( const aRefinementX2d& ref, const aGeneratorX2d& generator, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx=dg::DIR, bc bcy=dg::PER):
        dg::aGeometryX2d( generator.zeta0(), generator.zeta1(), generator.eta0(), generator.eta1(),ref.fx_new(Nx,fx),ref.fy_new(Ny,fy),n, ref.nx_new(Nx,fx), ref.ny_new(Ny,fy), bcx, bcy)
    {
        handle_ = generator;
        ref_=ref;
        construct( double fx,double fy,n,Nx,Ny);
    }

    const aGeneratorX2d& generator() const{return handle_.get();}
    virtual CurvilinearGridX2d* clone()const{return new CurvilinearGridX2d(*this);}
    private:
    void construct(double fx, double fy, unsigned n, unsigned Nx, unsigned Ny)
    {
        CurvilinearProductRefinedGridX3d g( ref_.get(), handle_.get(),fx,fy,n,Nx,Ny,1,bcx());
        jac_=g.jacobian();
        map_=g.map();
        metric_=g.metric();
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_jacobian( ) const {
        return jac_;
    }
    virtual SparseTensor<thrust::host_vector<double> > do_compute_metric( ) const {
        return metric_;
    }
    virtual std::vector<thrust::host_vector<double> > do_compute_map()const{return map_;}
    dg::SparseTensor<thrust::host_vector<double> > jac_, metric_;
    std::vector<thrust::host_vector<double> > map_;
    dg::Handle<aGeneratorX2d> handle_;
    dg::Handle<aRefinementX2d> ref_;
};

///@}


} //namespace dg

