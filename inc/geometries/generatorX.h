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
 @ingroup generators_geo
*/
template<class real_type>
struct aRealGeneratorX2d
{
    real_type zeta0(real_type fx) const{return do_zeta0(fx);}
    real_type zeta1(real_type fx) const{return do_zeta1(fx);}
    real_type eta0(real_type fy) const{return do_eta0(fy);}
    real_type eta1(real_type fy) const{return do_eta1(fy);}
    ///@brief sparsity pattern for metric
    bool isOrthogonal() const { return do_isOrthogonal(); }

    /**
    * @brief Generate grid points and elements of the Jacobian
    *
    * @param zeta1d (input) a list of \f$ N_\zeta\f$ points \f$ 0<\zeta_i<\f$width()
    * @param eta1d (input) a list of \f$ N_\eta\f$ points \f$ 0<\eta_j<\f$height()
    * @param nodeX0 is the index of the first point in eta1d  after the first jump in topology in \f$ \eta\f$
    * @param nodeX1 is the index of the first point in eta1d  after the second jump in topology in \f$ \eta\f$
    * @param x (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ x(\zeta_i, \eta_j)\f$
    * @param y (output) the list of \f$ N_\eta N_\zeta\f$ coordinates \f$ y(\zeta_i, \eta_j)\f$
    * @param zetaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial x (\zeta_i, \eta_j)\f$
    * @param zetaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\zeta/\partial y (\zeta_i, \eta_j)\f$
    * @param etaX (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial x (\zeta_i, \eta_j)\f$
    * @param etaY (output) the list of \f$ N_\eta N_\zeta\f$ elements \f$ \partial\eta/\partial y (\zeta_i, \eta_j)\f$
    * @note the first (\f$ \zeta\f$) coordinate shall be constructed contiguously in memory, i.e. the resuling lists are \f$ x_0=x(\zeta_0, \eta_0),\ x_1=x(\zeta_1, \eta_0)\ x_2=x(\zeta_2, \eta_0)\dots x_{NM-1}=x(\zeta_{N-1} \eta_{M-1})\f$
    * @note All the resulting vectors are write-only and get properly resized
    */
    void generate(
         const thrust::host_vector<real_type>& zeta1d,
         const thrust::host_vector<real_type>& eta1d,
         unsigned nodeX0, unsigned nodeX1,
         thrust::host_vector<real_type>& x,
         thrust::host_vector<real_type>& y,
         thrust::host_vector<real_type>& zetaX,
         thrust::host_vector<real_type>& zetaY,
         thrust::host_vector<real_type>& etaX,
         thrust::host_vector<real_type>& etaY) const
    {
        unsigned size = zeta1d.size()*eta1d.size();
        x.resize(size), y.resize(size);
        zetaX = zetaY = etaX = etaY =x ;
        do_generate( zeta1d, eta1d,nodeX0,nodeX1,x,y,zetaX,zetaY,etaX,etaY);
    }

    /**
    * @brief Abstract clone method that returns a copy on the heap
    *
    * @return a copy of *this on the heap
    */
    virtual aRealGeneratorX2d* clone() const=0;
    virtual ~aRealGeneratorX2d(){}

    protected:
    aRealGeneratorX2d(){}
    aRealGeneratorX2d(const aRealGeneratorX2d&){}
    aRealGeneratorX2d& operator=(const aRealGeneratorX2d&){
        return *this;
    }
    private:
    virtual void do_generate(
         const thrust::host_vector<real_type>& zeta1d,
         const thrust::host_vector<real_type>& eta1d,
         unsigned nodeX0, unsigned nodeX1,
         thrust::host_vector<real_type>& x,
         thrust::host_vector<real_type>& y,
         thrust::host_vector<real_type>& zetaX,
         thrust::host_vector<real_type>& zetaY,
         thrust::host_vector<real_type>& etaX,
         thrust::host_vector<real_type>& etaY) const = 0;
    virtual bool do_isOrthogonal()const{return false;}
    virtual real_type do_zeta0(real_type fx) const=0;
    virtual real_type do_zeta1(real_type fx) const=0;
    virtual real_type do_eta0(real_type fy) const=0;
    virtual real_type do_eta1(real_type fy) const=0;


};
using aGeneratorX2d = dg::geo::aRealGeneratorX2d<double>;

}//namespace geo
}//namespace dg
