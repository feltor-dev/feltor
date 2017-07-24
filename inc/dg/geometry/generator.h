#pragma once

namespace dg
{
namespace geo
{
/**
* @brief The abstract generator base class 

A generator is there to construct coordinate transformations from physical coordinates
\f$ x,y\f$ to the computational domain \f$\zeta, \eta\f$, which
is a product space. 
@note the origin of the computational space is assumed to be (0,0)
 @ingroup generators
*/
struct aGridGenerator
{
    virtual double width()  const=0; //!<length in \f$ \zeta\f$ of the computational space
    virtual double height() const=0; //!<length in \f$ \eta\f$ of the computational space
    virtual bool isOrthonormal() const{return false;} //!< true if coordinate system is orthonormal (false by default)
    virtual bool isOrthogonal() const{return false;} //!< true if coordinate system is orthogonal (false by default)
    virtual bool isConformal()const{return false;} //!< true if coordinate system is conformal (false by default)
    /**
    * @brief Generate grid points and elements of the Jacobian 
    *
    * @param zeta1d (input) a list of \f$ N_\zeta\f$ points \f$ 0<\zeta_i<\f$width() 
    * @param eta1d (input) a list of \f$ N_\eta\f$ points \f$ 0<\eta_j<\f$height() 
    * @param x (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ x(\zeta_i, \eta_j)\f$ 
    * @param y (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ y(\zeta_i, \eta_j)\f$ 
    * @param zetaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial x (\zeta_i, \eta_j)\f$ 
    * @param zetaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial y (\zeta_i, \eta_j)\f$ 
    * @param etaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial x (\zeta_i, \eta_j)\f$ 
    * @param etaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial y (\zeta_i, \eta_j)\f$ 
    * @note the \f$ \zeta\f$ coordinate is contiguous in memory
    * @note All the resulting vectors are write-only and get properly resized
    */
    void operator()( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) const
    {
        unsigned size = zeta1d.size()*eta1d.size();
        x.resize(size), y.resize(size);
        zetaX = zetaY = etaX = etaY =x ;
        generate( zeta1d, eta1d, x,y,zetaX,zetaY,etaX,etaY);
    }
    /**
    * @brief Abstract clone method that returns a copy on the heap
    *
    * @return a copy of *this on the heap
    */
    virtual aGridGenerator* clone() const=0;

    protected:
    aGridGenerator(const aGridGenerator& src){}
    aGridGenerator& operator=(const aGridGenerator& src){}
    ///@copydoc operator()()
    virtual void generate(
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) const = 0;
    
   virtual ~aGridGenerator(){}
};

/**
* @brief The shifted identity coordinate transformation

@note in fact it's not completely the identity because we assume that the origin is always (0,0) in the computational space
 @ingroup generators
*/
struct IdentityGenerator: public aGridGenerator
{
    virtual double width()  const{return lx_;} 
    virtual double height() const{return ly_;}
    virtual bool isOrthonormal() const{return true;}
    virtual bool isOrthogonal() const{return true;}
    virtual bool isConformal()const{return true;}
    virtual IdentityGenerator* clone() const{return new IdentityGenerator(*this);}

    /**
    * @brief Define the 2d box in the physical domain in which to construct the coordinates
    *
    * @param x0 x-coordinate of lower left point
    * @param x1 x-coordinate of upper right point
    * @param y0 y-coordinate of lower left point
    * @param y1 y-coordainte of upper right point
    */
    IdentityGenerator( double x0, double x1, double y0, double y1){
        x0_ = x0; lx_ = (x1-x0);
        y0_ = y0; ly_ = (y1-y0);
    }

    protected:
    virtual void generate(
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) const
     {
         for(unsigned i=0; i<eta1d.size();i++)
             for(unsigned j=0; j<zeta1d.size();j++)
             {
                 x[i*zeta1d.size()+j] = x0_ + zeta1d[j];
                 y[i*zeta1d.size()+j] = y0_ + eta1d[i];
                 zetaX[i*zeta1d.size()+j] = 1;
                 zetaY[i*zeta1d.size()+j] = 0;
                 etaX[i*zeta1d.size()+j] = 0.;
                 etaY[i*zeta1d.size()+j] = 1.;
             }
                 
     }
     private:
     double x0_,y0_,lx_,ly_;
    
};

}//namespace geo
}//namespace dg
