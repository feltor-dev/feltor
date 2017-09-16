#pragma once

#include "dg/backend/gridX.h"
#include "dg/backend/interpolationX.cuh"
#include "dg/backend/evaluationX.cuh"
#include "dg/backend/weightsX.cuh"
#include "dg/runge_kutta.h"
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
void computeX_rzy( const BinaryFunctorsLvl1& psi,
        const thrust::host_vector<double>& y_vec, 
        const unsigned nodeX0, const unsigned nodeX1,
        thrust::host_vector<double>& r, //output r - values
        thrust::host_vector<double>& z, //output z - values
        const double* R_init, const double* Z_init,  //2 input coords on perp line
        double f_psi,  //input f
        int mode ) 
{
    thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old);
    thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size());
    thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
    begin[0] = R_init[0], begin[1] = Z_init[0];
    dg::geo::ribeiro::FieldRZY fieldRZYconf(psi);
    dg::geo::equalarc::FieldRZY fieldRZYequi(psi);
    fieldRZYconf.set_f(f_psi);
    fieldRZYequi.set_f(f_psi);
    unsigned steps = 1; double eps = 1e10, eps_old=2e10;
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-11)
    {
        eps_old = eps, r_old = r, z_old = z;
        //////////////////////bottom left region/////////////////////
        if( nodeX0 != 0)
        {
            begin[0] = R_init[1], begin[1] = Z_init[1];
            if(mode==0)dg::stepperRK17( fieldRZYconf, begin, end, 0, y_vec[nodeX0-1], steps);
            if(mode==1)dg::stepperRK17( fieldRZYequi, begin, end, 0, y_vec[nodeX0-1], steps);
            r[nodeX0-1] = end[0], z[nodeX0-1] = end[1];
        }
        for( int i=nodeX0-2; i>=0; i--)
        {
            temp = end;
            if(mode==0)dg::stepperRK17( fieldRZYconf, temp, end, y_vec[i+1], y_vec[i], steps);
            if(mode==1)dg::stepperRK17( fieldRZYequi, temp, end, y_vec[i+1], y_vec[i], steps);
            r[i] = end[0], z[i] = end[1];
        }
        ////////////////middle region///////////////////////////
        begin[0] = R_init[0], begin[1] = Z_init[0];
        if(mode==0)dg::stepperRK17( fieldRZYconf, begin, end, 0, y_vec[nodeX0], steps);
        if(mode==1)dg::stepperRK17( fieldRZYequi, begin, end, 0, y_vec[nodeX0], steps);
        r[nodeX0] = end[0], z[nodeX0] = end[1];
        for( unsigned i=nodeX0+1; i<nodeX1; i++)
        {
            temp = end;
            if(mode==0)dg::stepperRK17( fieldRZYconf, temp, end, y_vec[i-1], y_vec[i], steps);
            if(mode==1)dg::stepperRK17( fieldRZYequi, temp, end, y_vec[i-1], y_vec[i], steps);
            r[i] = end[0], z[i] = end[1];
        }
        temp = end;
        if(mode==0)dg::stepperRK17( fieldRZYconf, temp, end, y_vec[nodeX1-1], 2.*M_PI, steps);
        if(mode==1)dg::stepperRK17( fieldRZYequi, temp, end, y_vec[nodeX1-1], 2.*M_PI, steps);
        eps = sqrt( (end[0]-R_init[0])*(end[0]-R_init[0]) + (end[1]-Z_init[0])*(end[1]-Z_init[0]));
        std::cout << "abs. error is "<<eps<<" with "<<steps<<" steps\n";
        ////////////////////bottom right region
        if( nodeX0!= 0)
        {
            begin[0] = R_init[1], begin[1] = Z_init[1];
            if(mode==0)dg::stepperRK17( fieldRZYconf, begin, end, 2.*M_PI, y_vec[nodeX1], steps);
            if(mode==1)dg::stepperRK17( fieldRZYequi, begin, end, 2.*M_PI, y_vec[nodeX1], steps);
            r[nodeX1] = end[0], z[nodeX1] = end[1];
        }
        for( unsigned i=nodeX1+1; i<y_vec.size(); i++)
        {
            temp = end;
            if(mode==0)dg::stepperRK17( fieldRZYconf, temp, end, y_vec[i-1], y_vec[i], steps);
            if(mode==1)dg::stepperRK17( fieldRZYequi, temp, end, y_vec[i-1], y_vec[i], steps);
            r[i] = end[0], z[i] = end[1];
        }
        //compute error in R,Z only
        dg::blas1::axpby( 1., r, -1., r_old, r_diff);
        dg::blas1::axpby( 1., z, -1., z_old, z_diff);
        double er = dg::blas1::dot( r_diff, r_diff);
        double ez = dg::blas1::dot( z_diff, z_diff);
        double ar = dg::blas1::dot( r, r);
        double az = dg::blas1::dot( z, z);
        eps =  sqrt( er + ez)/sqrt(ar+az);
        std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        steps*=2;
    }
    r = r_old, z = z_old;
}


} //namespace detail
}//namespace orthogonal
///@endcond

/**
 * @brief Choose points on inside or outside line 
 *
 * @ingroup generators_geo
 */
struct SimpleOrthogonalX : public aGeneratorX2d
{
    SimpleOrthogonalX(): f0_(1), firstline_(0){}
    ///psi_0 must be the closed surface, 0 the separatrix
    SimpleOrthogonalX( const BinaryFunctorsLvl2& psi, double psi_0, 
            double xX, double yX, double x0, double y0, int firstline =0): psi_(psi)
    {
        firstline_ = firstline;
        orthogonal::detail::Fpsi fpsi(psi_, x0, y0, firstline);
        double R0, Z0; 
        f0_ = fpsi.construct_f( psi_0, R0, Z0);
        zeta0_=f0_*psi_0;
        dg::geo::orthogonal::detail::InitialX initX(psi_, xX, yX);
        initX.find_initial(psi_0, R0_, Z0_);
    }
    SimpleOrthogonalX* clone()const{return new SimpleOrthogonalX(*this);}
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
         thrust::host_vector<double>& etaY) const
    {

        thrust::host_vector<double> r_init, z_init;
        orthogonal::detail::computeX_rzy( psi_, eta1d, nodeX0, nodeX1, r_init, z_init, R0_, Z0_, f0_, firstline_);
        dg::geo::orthogonal::detail::Nemov nemov(psi_, f0_, firstline_);
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
    double do_zeta0(double fx) const { return zeta0_; }
    double do_zeta1(double fx) const { return -fx/(1.-fx)*zeta0_;}
    double do_eta0(double fy) const { return -2.*M_PI*fy/(1.-2.*fy); }
    double do_eta1(double fy) const { return 2.*M_PI*(1.+fy/(1.-2.*fy));}
    BinaryFunctorsLvl2 psi_;
    double R0_[2], Z0_[2];
    double zeta0_, f0_;
    int firstline_;
};

/**
 * @brief Choose points on separatrix 
 *
 * @ingroup generators_geo
 */
struct SeparatrixOrthogonal : public aGeneratorX2d
{
    /**
     * @brief Construct 
     *
     * @param psi
     * @param psi_0
     * @param xX the X-point
     * @param yX the X-point
     * @param x0
     * @param y0
     * @param firstline =0 means conformal, =1 means equalarc discretization
     */
    SeparatrixOrthogonal( const BinaryFunctorsLvl2& psi, double psi_0, //psi_0 must be the closed surface, 0 the separatrix
            double xX, double yX, double x0, double y0, int firstline ):
        psi_(psi),
        sep_( psi, xX, yX, x0, y0, firstline)
    {
        firstline_ = firstline;
        f0_ = sep_.get_f();
        psi_0_=psi_0;
    }
    SeparatrixOrthogonal* clone()const{return new SeparatrixOrthogonal(*this);}
    private:
    bool isConformal()const{return false;}
    bool do_isOrthogonal()const{return true;}
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
        dg::geo::orthogonal::detail::Nemov nemov(psi_, f0_, firstline_);

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
        orthogonal::detail::construct_rz(nemov, 0., zeta1dI, r_initC, z_initC, xIC, yIC, hIC);
        orthogonal::detail::construct_rz(nemov, 0., zeta1dO, r_initC, z_initC, xOC, yOC, hOC);
        orthogonal::detail::construct_rz(nemov, 0., zeta1dI, r_initF, z_initF, xIF, yIF, hIF);
        orthogonal::detail::construct_rz(nemov, 0., zeta1dO, r_initF, z_initF, xOF, yOF, hOF);
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
            zetaX[idx] = f0_*psipX;
            zetaY[idx] = f0_*psipY;
            etaX[idx] = -h[idx]*psipY;
            etaY[idx] = +h[idx]*psipX;
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
    BinaryFunctorsLvl2 psi_;
    dg::geo::detail::SeparatriX sep_;
};

// /**
//* @brief Integrates the equations for a field line and 1/B
// */ 
//struct XField
//{
//    XField( dg::geo::solovev::GeomParameters gp,const dg::GridX2d& gXY, const thrust::host_vector<double>& g):
//        gp_(gp),
//        psipR_(gp), psipZ_(gp),
//        ipol_(gp), invB_(gp), gXY_(gXY), g_(dg::create::forward_transform(g, gXY)) 
//    { 
//        solovev::HessianRZtau hessianRZtau(gp);
//        R_X = gp.R_0-1.1*gp.triangularity*gp.a;
//        Z_X = -1.1*gp.elongation*gp.a;
//        thrust::host_vector<double> X(2,0), XN(X);
//        X[0] = R_X, X[1] = Z_X;
//        for( unsigned i=0; i<3; i++)
//        {
//            hessianRZtau.newton_iteration( X, XN);
//            XN.swap(X);
//        }
//        R_X = X[0], Z_X = X[1];
//    
//    }
//
//    /**
//     * @brief \f[ \frac{d \hat{R} }{ d \varphi}  = \frac{\hat{R}}{\hat{I}} \frac{\partial\hat{\psi}_p}{\partial \hat{Z}}, \hspace {3 mm}
//     \frac{d \hat{Z} }{ d \varphi}  =- \frac{\hat{R}}{\hat{I}} \frac{\partial \hat{\psi}_p}{\partial \hat{R}} , \hspace {3 mm}
//     \frac{d \hat{l} }{ d \varphi}  =\frac{\hat{R}^2 \hat{B}}{\hat{I}  \hat{R}_0}  \f]
//     */ 
//    void operator()( const dg::HVec& y, dg::HVec& yp)
//    {
//        //x,y,s,R,Z
//        double psipR = psipR_(y[3],y[4]), psipZ = psipZ_(y[3],y[4]), ipol = ipol_( y[3],y[4]);
//        double xs = y[0],ys=y[1];
//        if( y[4] > Z_X) //oberhalb vom X-Punkt
//            gXY_.shift_topologic( y[0], M_PI, xs,ys);
//        else 
//        {
//            if( y[1] > M_PI) //Startpunkt vermutlich in der rechten Hälfte
//                gXY_.shift_topologic( y[0], gXY_.y1()-1e-10, xs,ys);
//            else
//                gXY_.shift_topologic( y[0], gXY_.y0()+1e-10, xs,ys);
//        }
//        if( !gXY_.contains(xs,ys))
//        {
//            if(y[0] > R_X) ys = gXY_.y1()-1e-10;
//            else           ys = gXY_.y0()+1e-10;
//        }
//        double g = dg::interpolate( xs,  ys, g_, gXY_);
//        yp[0] =  0;
//        yp[1] =  y[3]*g*(psipR*psipR+psipZ*psipZ)/ipol;
//        yp[2] =  y[3]*y[3]/invB_(y[3],y[4])/ipol/gp_.R_0; //ds/dphi =  R^2 B/I/R_0_hat
//        yp[3] =  y[3]*psipZ/ipol;              //dR/dphi =  R/I mod::Psip_Z
//        yp[4] = -y[3]*psipR/ipol;             //dZ/dphi = -R/I mod::Psip_R
//
//    }
//    /**
//     * @brief \f[   \frac{1}{\hat{B}} = 
//      \frac{\hat{R}}{\hat{R}_0}\frac{1}{ \sqrt{ \hat{I}^2  + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{R}}\right)^2
//      + \left(\frac{\partial \hat{\psi}_p }{ \partial \hat{Z}}\right)^2}}  \f]
//     */ 
//    double operator()( double R, double Z) const { return invB_(R,Z); }
//    /**
//     * @brief == operator()(R,Z)
//     */ 
//    double operator()( double R, double Z, double phi) const { return invB_(R,Z,phi); }
//    
//    private:
//    dg::geo::solovev::GeomParameters gp_;
//    dg::geo::solovev::mod::PsipR  psipR_;
//    dg::geo::solovev::mod::PsipZ  psipZ_;
//    dg::geo::Ipol   ipol_;
//    dg::geo::InvB   invB_;
//    const dg::GridX2d gXY_;
//    thrust::host_vector<double> g_;
//    double R_X, Z_X;
//   
//};
//

}//namespace geo
}//namespace dg

