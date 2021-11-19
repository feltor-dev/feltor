#pragma once

#include "dg/algorithm.h"
#include "generatorX.h"
#include "utilitiesX.h"

#include "simple_orthogonal.h"



namespace dg
{
namespace geo
{
///@cond
namespace orthogonal
{

namespace detail
{

//compute the vector of r and z - values that form one psi surface
//assumes y_0 = 0
template<class real_type>
void computeX_rzy( const CylindricalFunctorsLvl1& psi,
        const thrust::host_vector<real_type>& y_vec,
        const unsigned nodeX0, const unsigned nodeX1,
        thrust::host_vector<real_type>& r, //output r - values
        thrust::host_vector<real_type>& z, //output z - values
        const real_type* R_init, const real_type* Z_init,  //2 input coords on perp line
        real_type f_psi,  //input f
        int mode, bool verbose = false )
{
    thrust::host_vector<real_type> r_old(y_vec.size(), 0), r_diff( r_old);
    thrust::host_vector<real_type> z_old(y_vec.size(), 0), z_diff( z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size());
    std::array<real_type,2> begin{ {0,0} }, end(begin), temp(begin);
    begin[0] = R_init[0], begin[1] = Z_init[0];
    dg::geo::ribeiro::FieldRZY fieldRZYconf(psi);
    dg::geo::equalarc::FieldRZY fieldRZYequi(psi);
    fieldRZYconf.set_f(f_psi);
    fieldRZYequi.set_f(f_psi);
    unsigned steps = 1; real_type eps = 1e10, eps_old=2e10;
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-11)
    {
        eps_old = eps, r_old = r, z_old = z;
        //////////////////////bottom left region/////////////////////
        if( nodeX0 != 0)
        {
            begin[0] = R_init[1], begin[1] = Z_init[1];
            if(mode==0)dg::stepperRK( "Feagin-17-8-10",  fieldRZYconf, 0, begin, y_vec[nodeX0-1], end, steps);
            if(mode==1)dg::stepperRK( "Feagin-17-8-10",  fieldRZYequi, 0, begin, y_vec[nodeX0-1], end, steps);
            r[nodeX0-1] = end[0], z[nodeX0-1] = end[1];
        }
        for( int i=nodeX0-2; i>=0; i--)
        {
            temp = end;
            if(mode==0)dg::stepperRK( "Feagin-17-8-10",  fieldRZYconf, y_vec[i+1], temp, y_vec[i], end, steps);
            if(mode==1)dg::stepperRK( "Feagin-17-8-10",  fieldRZYequi, y_vec[i+1], temp, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1];
        }
        ////////////////middle region///////////////////////////
        begin[0] = R_init[0], begin[1] = Z_init[0];
        if(mode==0)dg::stepperRK( "Feagin-17-8-10",  fieldRZYconf, 0, begin, y_vec[nodeX0], end, steps);
        if(mode==1)dg::stepperRK( "Feagin-17-8-10",  fieldRZYequi, 0, begin, y_vec[nodeX0], end, steps);
        r[nodeX0] = end[0], z[nodeX0] = end[1];
        for( unsigned i=nodeX0+1; i<nodeX1; i++)
        {
            temp = end;
            if(mode==0)dg::stepperRK( "Feagin-17-8-10",  fieldRZYconf, y_vec[i-1], temp, y_vec[i], end, steps);
            if(mode==1)dg::stepperRK( "Feagin-17-8-10",  fieldRZYequi, y_vec[i-1], temp, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1];
        }
        temp = end;
        if(mode==0)dg::stepperRK( "Feagin-17-8-10",  fieldRZYconf, y_vec[nodeX1-1], temp, 2.*M_PI, end, steps);
        if(mode==1)dg::stepperRK( "Feagin-17-8-10",  fieldRZYequi, y_vec[nodeX1-1], temp, 2.*M_PI, end, steps);
        eps = sqrt( (end[0]-R_init[0])*(end[0]-R_init[0]) + (end[1]-Z_init[0])*(end[1]-Z_init[0]));
        if(verbose)std::cout << "abs. error is "<<eps<<" with "<<steps<<" steps\n";
        ////////////////////bottom right region
        if( nodeX0!= 0)
        {
            begin[0] = R_init[1], begin[1] = Z_init[1];
            if(mode==0)dg::stepperRK( "Feagin-17-8-10",  fieldRZYconf, 2.*M_PI, begin, y_vec[nodeX1], end, steps);
            if(mode==1)dg::stepperRK( "Feagin-17-8-10",  fieldRZYequi, 2.*M_PI, begin, y_vec[nodeX1], end, steps);
            r[nodeX1] = end[0], z[nodeX1] = end[1];
        }
        for( unsigned i=nodeX1+1; i<y_vec.size(); i++)
        {
            temp = end;
            if(mode==0)dg::stepperRK( "Feagin-17-8-10",  fieldRZYconf, y_vec[i-1], temp, y_vec[i], end, steps);
            if(mode==1)dg::stepperRK( "Feagin-17-8-10",  fieldRZYequi, y_vec[i-1], temp, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1];
        }
        //compute error in R,Z only
        dg::blas1::axpby( 1., r, -1., r_old, r_diff);
        dg::blas1::axpby( 1., z, -1., z_old, z_diff);
        real_type er = dg::blas1::dot( r_diff, r_diff);
        real_type ez = dg::blas1::dot( z_diff, z_diff);
        real_type ar = dg::blas1::dot( r, r);
        real_type az = dg::blas1::dot( z, z);
        eps =  sqrt( er + ez)/sqrt(ar+az);
        if(verbose)std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        steps*=2;
    }
    r = r_old, z = z_old;
}


} //namespace detail
}//namespace orthogonal

/**
 * @brief Choose points on inside or outside line
 *
 * @attention Not consistent, do not use unless you know what you are doing
 * @ingroup generators_geo
 */
struct SimpleOrthogonalX : public aGeneratorX2d
{
    SimpleOrthogonalX(): f0_(1), firstline_(0){}
    SimpleOrthogonalX( const CylindricalFunctorsLvl2& psi, double psi_0,
            double xX, double yX, double x0, double y0, int firstline =0): psi_(psi)
    {
        firstline_ = firstline;
        orthogonal::detail::Fpsi fpsi(psi_, CylindricalSymmTensorLvl1(), x0, y0, firstline);
        double R0, Z0;
        f0_ = fpsi.construct_f( psi_0, R0, Z0);
        zeta0_=f0_*psi_0;
        dg::geo::orthogonal::detail::InitialX initX(psi_, xX, yX);
        initX.find_initial(psi_0, R0_, Z0_);
    }
    virtual SimpleOrthogonalX* clone()const override final{return new SimpleOrthogonalX(*this);}
    private:
    bool isConformal()const{return false;}
    bool do_isOrthogonal()const{return true;}
    double f0() const{return f0_;}
    virtual void do_generate( //this one doesn't know if the separatrix comes to lie on a cell boundary or not
         const thrust::host_vector<double>& zeta1d,
         const thrust::host_vector<double>& eta1d,
         unsigned nodeX0, unsigned nodeX1,
         thrust::host_vector<double>& x,
         thrust::host_vector<double>& y,
         thrust::host_vector<double>& zetaX,
         thrust::host_vector<double>& zetaY,
         thrust::host_vector<double>& etaX,
         thrust::host_vector<double>& etaY) const override final
    {

        thrust::host_vector<double> r_init, z_init;
        orthogonal::detail::computeX_rzy( psi_, eta1d, nodeX0, nodeX1, r_init, z_init, R0_, Z0_, f0_, firstline_);
        dg::geo::orthogonal::detail::Nemov nemov(psi_, CylindricalSymmTensorLvl1(), f0_, firstline_);
        thrust::host_vector<double> h;
        orthogonal::detail::construct_rz(nemov, zeta0_, zeta1d, r_init, z_init, x, y, h);
        unsigned size = x.size();
        zetaX.resize(size), zetaY.resize(size),
        etaX.resize(size), etaY.resize(size);
        for( unsigned idx=0; idx<size; idx++)
        {
            double psipR = psi_.dfx()(x[idx], y[idx]);
            double psipZ = psi_.dfy()(x[idx], y[idx]);
            zetaX[idx] = f0_*psipR;
            zetaY[idx] = f0_*psipZ;
            etaX[idx] = -h[idx]*psipZ;
            etaY[idx] = +h[idx]*psipR;
        }
    }
    double do_zeta0(double fx) const override final{ return zeta0_; }
    double do_zeta1(double fx) const override final{ return -fx/(1.-fx)*zeta0_;}
    double do_eta0(double fy) const override final{ return -2.*M_PI*fy/(1.-2.*fy); }
    double do_eta1(double fy) const override final{ return 2.*M_PI*(1.+fy/(1.-2.*fy));}
    CylindricalFunctorsLvl2 psi_;
    double R0_[2], Z0_[2];
    double zeta0_, f0_;
    int firstline_;
};
///@endcond

/**
 * @brief Choose points on separatrix and construct grid from there
 *
 * This is the algorithm described in:
 * "M. Wiesenberger, M. Held, L. Einkemmer, A. Kendl Streamline integration as a method for structured grid generation in X-point geometry Journal of Computational Physics 373, 370-384 (2018)"
 * @note The resulting coordinate transformation for \f$ \zeta\f$ will by linear in \f$ \psi\f$
 * @attention Assumes that the separatrix is given by \f$ \psi = 0\f$. If this
 * is not the case, then use the \c normalize_solovev_t program to change the parameters.
 * Further, it is assumed that closed flux surfaces inside of the separatrix exist.
 * @ingroup generators_geo
 */
struct SeparatrixOrthogonal : public aGeneratorX2d
{
    /**
     * @brief Construct
     *
     * @param psi the flux function, the separatrix must be at \f$ \psi = 0\f$
     * @param chi the monitor tensor, see \c dg::geo::make_Xconst_monitor or \c dg::geo::make_Xbump_monitor
     * @param psi_0 The left boundary of the grid this generator will generate. Must be a closed flux surface.
       @param xX the X-point x - coordinate
     * @param yX the X-point y - coordinate
     * @param x0 a point in the inside of the separatrix (can be the O-point, defines the angle for initial separatrix integration)
     * @param y0 a point in the inside of the separatrix (can be the O-point, defines the angle for initial separatrix integration)
     * @param firstline =0 means conformal, =1 means equalarc discretization of the separatrix
     * @param verbose if true the integrators will write additional information to \c std::cout
     */
    SeparatrixOrthogonal( const CylindricalFunctorsLvl2& psi, const CylindricalSymmTensorLvl1& chi, double psi_0, //psi_0 must be the closed surface, 0 the separatrix
            double xX, double yX, double x0, double y0, int firstline, bool verbose = false ):
        psi_(psi), chi_(chi),
        sep_( psi, chi, xX, yX, x0, y0, firstline, verbose), m_verbose( verbose)
    {
        firstline_ = firstline;
        f0_ = sep_.get_f();
        psi_0_=psi_0;
    }
    SeparatrixOrthogonal* clone()const{return new SeparatrixOrthogonal(*this);}
    private:
    bool isConformal()const{return false;}
    bool do_isOrthogonal()const{return false;}
    double f0() const{return sep_.get_f();}
    virtual void do_generate(  //this one doesn't know if the separatrix comes to lie on a cell boundary or not
         const thrust::host_vector<double>& zeta1d,
         const thrust::host_vector<double>& eta1d,
         unsigned nodeX0, unsigned nodeX1,
         thrust::host_vector<double>& x,
         thrust::host_vector<double>& y,
         thrust::host_vector<double>& zetaX,
         thrust::host_vector<double>& zetaY,
         thrust::host_vector<double>& etaX,
         thrust::host_vector<double>& etaY) const
    {

        thrust::host_vector<double> r_init, z_init;
        sep_.compute_rzy( eta1d, nodeX0, nodeX1, r_init, z_init);
        dg::geo::orthogonal::detail::Nemov nemov(psi_, chi_, f0_, firstline_);

        //separate integration of inside and outside
        unsigned inside=0;
        for(unsigned i=0; i<zeta1d.size(); i++)
            if( zeta1d[i]< 0) inside++;//how many points are inside
        thrust::host_vector<double> zeta1dI( inside, 0), zeta1dO( zeta1d.size() - inside, 0);
        for( unsigned i=0; i<inside; i++)
            zeta1dI[i] = zeta1d[ inside-1-i];
        for( unsigned i=inside; i<zeta1d.size(); i++)
            zeta1dO[i-inside] = zeta1d[ i];
        //separate integration close and far from separatrix
        //this is done due to performance reasons (it takes more steps to integrate close to the X-point)
        thrust::host_vector<int> idxC, idxF;
        thrust::host_vector<double> r_initC, r_initF, z_initC, z_initF;
        for( unsigned i=0; i<eta1d.size(); i++)
        {
            if( fabs(eta1d[i]) < 0.05 || fabs( eta1d[i] - 2.*M_PI) < 0.05)
            {
                idxC.push_back( i);
                r_initC.push_back( r_init[i]);
                z_initC.push_back( z_init[i]);
            }
            else
            {
                idxF.push_back( i);
                r_initF.push_back( r_init[i]);
                z_initF.push_back( z_init[i]);
            }
        }

        thrust::host_vector<double> xIC, yIC, hIC, xOC,yOC,hOC;
        thrust::host_vector<double> xIF, yIF, hIF, xOF,yOF,hOF;
        orthogonal::detail::construct_rz(nemov, 0., zeta1dI, r_initC, z_initC,
                xIC, yIC, hIC);
        orthogonal::detail::construct_rz(nemov, 0., zeta1dO, r_initC, z_initC,
                xOC, yOC, hOC);
        orthogonal::detail::construct_rz(nemov, 0., zeta1dI, r_initF, z_initF,
                xIF, yIF, hIF);
        orthogonal::detail::construct_rz(nemov, 0., zeta1dO, r_initF, z_initF,
                xOF, yOF, hOF);
        //now glue far and close back together
        thrust::host_vector<double> xI(inside*eta1d.size()), xO( (zeta1d.size()-inside)*eta1d.size());
        thrust::host_vector<double> yI(xI), hI(xI), yO(xO),hO(xO);
        for( unsigned i=0; i<idxC.size(); i++)
            for(unsigned j=0; j<zeta1dI.size(); j++)
            {
                xI[idxC[i]*zeta1dI.size() + j] = xIC[i*zeta1dI.size() + j];
                yI[idxC[i]*zeta1dI.size() + j] = yIC[i*zeta1dI.size() + j];
                hI[idxC[i]*zeta1dI.size() + j] = hIC[i*zeta1dI.size() + j];
            }
        for( unsigned i=0; i<idxF.size(); i++)
            for(unsigned j=0; j<zeta1dI.size(); j++)
            {
                xI[idxF[i]*zeta1dI.size() + j] = xIF[i*zeta1dI.size() + j];
                yI[idxF[i]*zeta1dI.size() + j] = yIF[i*zeta1dI.size() + j];
                hI[idxF[i]*zeta1dI.size() + j] = hIF[i*zeta1dI.size() + j];
            }
        for( unsigned i=0; i<idxC.size(); i++)
            for(unsigned j=0; j<zeta1dO.size(); j++)
            {
                xO[idxC[i]*zeta1dO.size() + j] = xOC[i*zeta1dO.size() + j];
                yO[idxC[i]*zeta1dO.size() + j] = yOC[i*zeta1dO.size() + j];
                hO[idxC[i]*zeta1dO.size() + j] = hOC[i*zeta1dO.size() + j];
            }
        for( unsigned i=0; i<idxF.size(); i++)
            for(unsigned j=0; j<zeta1dO.size(); j++)
            {
                xO[idxF[i]*zeta1dO.size() + j] = xOF[i*zeta1dO.size() + j];
                yO[idxF[i]*zeta1dO.size() + j] = yOF[i*zeta1dO.size() + j];
                hO[idxF[i]*zeta1dO.size() + j] = hOF[i*zeta1dO.size() + j];
            }

        //now glue inside and outside together
        unsigned size = zeta1d.size()*eta1d.size();
        x.resize( size); y.resize( size);
        thrust::host_vector<double> h(size);
        for( unsigned i=0; i<eta1d.size(); i++)
            for( unsigned j=0; j<inside; j++)
            {
                x[i*zeta1d.size()+j] = xI[i*zeta1dI.size() + inside-1-j];
                y[i*zeta1d.size()+j] = yI[i*zeta1dI.size() + inside-1-j];
                h[i*zeta1d.size()+j] = hI[i*zeta1dI.size() + inside-1-j];
            }
        for( unsigned i=0; i<eta1d.size(); i++)
            for( unsigned j=inside; j<zeta1d.size(); j++)
            {
                x[i*zeta1d.size()+j] = xO[i*zeta1dO.size() + j-inside];
                y[i*zeta1d.size()+j] = yO[i*zeta1dO.size() + j-inside];
                h[i*zeta1d.size()+j] = hO[i*zeta1dO.size() + j-inside];
            }

        zetaX.resize(size), zetaY.resize(size),
        etaX.resize(size), etaY.resize(size);
        for( unsigned idx=0; idx<size; idx++)
        {
            double psipX = psi_.dfx()(x[idx], y[idx]);
            double psipY = psi_.dfy()(x[idx], y[idx]);
            double chiXX = chi_.xx()( x[idx], y[idx]),
                   chiXY = chi_.xy()( x[idx], y[idx]),
                   chiYY = chi_.yy()( x[idx], y[idx]);
            zetaX[idx] = f0_*psipX;
            zetaY[idx] = f0_*psipY;
            etaX[idx] = -h[idx]*(chiXY*psipX + chiYY*psipY);
            etaY[idx] = +h[idx]*(chiXX*psipX + chiXY*psipY);
        }
    }
    virtual double do_zeta0(double fx) const { return f0_*psi_0_; }
    virtual double do_zeta1(double fx) const { return -fx/(1.-fx)*f0_*psi_0_;}
    virtual double do_eta0(double fy) const { return -2.*M_PI*fy/(1.-2.*fy); }
    virtual double do_eta1(double fy) const { return 2.*M_PI*(1.+fy/(1.-2.*fy));}
    private:
    double R0_[2], Z0_[2];
    double f0_, psi_0_;
    int firstline_;
    CylindricalFunctorsLvl2 psi_;
    CylindricalSymmTensorLvl1 chi_;
    dg::geo::detail::SeparatriX sep_;
    bool m_verbose;
};

}//namespace geo
}//namespace dg

