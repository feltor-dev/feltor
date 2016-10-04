#pragma once

#include "dg/backend/gridX.h"
#include "dg/backend/interpolationX.cuh"
#include "dg/backend/evaluationX.cuh"
#include "dg/backend/weightsX.cuh"
#include "dg/runge_kutta.h"
#include "utilitiesX.h"

#include "orthogonal.h"



namespace dg
{
namespace orthogonal
{

namespace detail
{

//find points on the perp line through the X-point
template< class Psi, class PsiX, class PsiY>
struct InitialX
{

    InitialX( Psi psi, PsiX psiX, PsiY psiY, double xX, double yX): 
        psip_(psi), fieldRZtau_(psiX, psiY), 
        xpointer_(psi, psiX, psiY, xX, yX, 1e-4)
    {
        //constructor finds four points around X-point and integrates them a bit away from it
        solovev::FieldRZtau<PsiX, PsiY> fieldRZtau_(psiX, psiY);
        thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin), end_old(end);
        double eps[] = {1e-11, 1e-12, 1e-11, 1e-12};
        for( unsigned i=0; i<4; i++)
        {
            xpointer_.set_quadrant( i);
            double x_min = -1e-4, x_max = 1e-4;
            dg::bisection1d( xpointer_, x_min, x_max, eps[i]);
            xpointer_.point( R_i_[i], Z_i_[i], (x_min+x_max)/2.);
            //std::cout << "Found initial point: "<<R_i_[i]<<" "<<Z_i_[i]<<" "<<psip_(R_i_[i], Z_i_[i])<<"\n";
            thrust::host_vector<double> begin(2), end(2), end_old(2);
            begin[0] = R_i_[i], begin[1] = Z_i_[i];
            double eps = 1e10, eps_old = 2e10;
            unsigned N=10;
            double psi0 = psip_(begin[0], begin[1]), psi1 = 1e3*psi0; 
            while( (eps < eps_old || eps > 1e-5 ) && eps > 1e-9)
            {
                eps_old = eps; end_old = end;
                N*=2; dg::stepperRK6( fieldRZtau_, begin, end, psi0, psi1, N); //lower order integrator is better for difficult field

                eps = sqrt( (end[0]-end_old[0])*(end[0]-end_old[0]) + (end[1]-end_old[1])*(end[1]-end_old[1]));
                if( isnan(eps)) { eps = eps_old/2.; end = end_old; }
                //std::cout << " for N "<< N<<" eps is "<<eps<<"\n";
            }
            R_i_[i] = end_old[0], Z_i_[i] = end_old[1];
            begin[0] = R_i_[i], begin[1] = Z_i_[i];
            eps = 1e10, eps_old = 2e10; N=10;
            psi0 = psip_(begin[0], begin[1]), psi1 = -0.01; 
            if( i==0||i==2)psi1*=-1.;
            while( (eps < eps_old || eps > 1e-5 ) && eps > 1e-9)
            {
                eps_old = eps; end_old = end;
                N*=2; dg::stepperRK6( fieldRZtau_, begin, end, psi0, psi1, N); //lower order integrator is better for difficult field

                eps = sqrt( (end[0]-end_old[0])*(end[0]-end_old[0]) + (end[1]-end_old[1])*(end[1]-end_old[1]));
                if( isnan(eps)) { eps = eps_old/2.; end = end_old; }
                //std::cout << " for N "<< N<<" eps is "<<eps<<"\n";
            }
            R_i_[i] = end_old[0], Z_i_[i] = end_old[1];
            std::cout << "Quadrant "<<i<<" Found initial point: "<<R_i_[i]<<" "<<Z_i_[i]<<" "<<psip_(R_i_[i], Z_i_[i])<<"\n";

        }
    }
    /**
     * @brief for a given psi finds the two points that lie on psi = const and the perpendicular line through the X-point
     *
     * @param psi psi \neq 0
     * @param R_0 array of size 2 (write-only)
     * @param Z_0 array of size 2 (write-only)
     */
    void find_initial( double psi, double* R_0, double* Z_0) 
    {
        thrust::host_vector<double> begin( 2, 0), end( begin), end_old(begin); 
        for( unsigned i=0; i<2; i++)
        {
            if(psi<0)
            {
                begin[0] = R_i_[2*i+1], begin[1] = Z_i_[2*i+1]; end = begin;
            }
            else
            {
                begin[0] = R_i_[2*i], begin[1] = Z_i_[2*i]; end = begin;
            }
            unsigned steps = 1;
            double eps = 1e10, eps_old=2e10;
            while( (eps < eps_old||eps > 1e-7) && eps > 1e-11)
            {
                eps_old = eps; end_old = end;
                dg::stepperRK17( fieldRZtau_, begin, end, psip_(begin[0], begin[1]), psi, steps);
                eps = sqrt( (end[0]-end_old[0])*(end[0]- end_old[0]) + (end[1]-end_old[1])*(end[1]-end_old[1]));
                //std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
                if( isnan(eps)) { eps = eps_old/2.; end = end_old; }
                steps*=2;
            }
            std::cout << "Found initial point "<<end_old[0]<<" "<<end_old[1]<<"\n";
            if( psi<0)
            {
                R_0[i] = R_i_[2*i+1] = begin[0] = end_old[0], Z_i_[2*i+1] = Z_0[i] = begin[1] = end_old[1];
            }
            else
            {
                R_0[i] = R_i_[2*i] = begin[0] = end_old[0], Z_i_[2*i] = Z_0[i] = begin[1] = end_old[1];
            }

        }
    }


    private:
    Psi psip_;
    const solovev::FieldRZtau<PsiX, PsiY> fieldRZtau_;
    dg::detail::XCross<Psi, PsiX, PsiY> xpointer_;
    double R_i_[4], Z_i_[4];

};

//compute the vector of r and z - values that form one psi surface
//assumes y_0 = 0
template <class PsiX, class PsiY>
void computeX_rzy( PsiX psiX, PsiY psiY, 
        const thrust::host_vector<double>& y_vec, 
        const unsigned nodeX0, const unsigned nodeX1,
        thrust::host_vector<double>& r, //output r - values
        thrust::host_vector<double>& z, //output z - values
        double* R_init, double* Z_init,  //2 input coords on perp line
        double f_psi,  //input f
        int mode ) 
{
    thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old);
    thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size());
    thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
    begin[0] = R_init[0], begin[1] = Z_init[0];
    solovev::ribeiro::FieldRZY<PsiX, PsiY> fieldRZYconf(psiX, psiY);
    solovev::equalarc::FieldRZY<PsiX, PsiY> fieldRZYequi(psiX, psiY);
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

template< class Psi, class PsiX, class PsiY, class LaplacePsi>
struct SimpleOrthogonalX
{
    SimpleOrthogonalX(): f0_(1), firstline_(0){}
    SimpleOrthogonalX( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double psi_0, //psi_0 must be the closed surface, 0 the separatrix
            double xX, double yX, double x0, double y0, int firstline =0):
        psiX_(psiX), psiY_(psiY), laplacePsi_(laplacePsi)
    {
        firstline_ = firstline;
        orthogonal::detail::Fpsi<Psi, PsiX, PsiY> fpsi(psi, psiX, psiY, x0, y0, firstline);
        double R0, Z0; 
        f0_ = fpsi.construct_f( psi_0, R0, Z0);
        zeta0_=f0_*psi_0;
        dg::orthogonal::detail::InitialX<Psi, PsiX, PsiY> initX(psi, psiX, psiY, xX, yX);
        initX.find_initial(psi_0, R0_, Z0_);
    }
    bool isConformal()const{return false;}
    bool isOrthogonal()const{return true;}
    double f0() const{return f0_;}
    void operator()( //this one doesn't know if the separatrix comes to lie on a cell boundary or not
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         const unsigned nodeX0, const unsigned nodeX1,
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) 
    {

        thrust::host_vector<double> r_init, z_init;
        orthogonal::detail::computeX_rzy( psiX_, psiY_, eta1d, nodeX0, nodeX1, r_init, z_init, R0_, Z0_, f0_, firstline_);
        orthogonal::detail::Nemov<PsiX, PsiY, LaplacePsi> nemov(psiX_, psiY_, laplacePsi_, f0_, firstline_);
        thrust::host_vector<double> h;
        orthogonal::detail::construct_rz(nemov, zeta0_, zeta1d, r_init, z_init, x, y, h);
        unsigned size = x.size();
        zetaX.resize(size), zetaY.resize(size), 
        etaX.resize(size), etaY.resize(size);
        for( unsigned idx=0; idx<size; idx++)
        {
            double psipR = psiX_(x[idx], y[idx]);
            double psipZ = psiY_(x[idx], y[idx]);
            zetaX[idx] = f0_*psipR;
            zetaY[idx] = f0_*psipZ;
            etaX[idx] = -h[idx]*psipZ;
            etaY[idx] = +h[idx]*psipR;
        }
    }
    double laplace(double x, double y) {return f0_*laplacePsi_(x,y);}
    private:
    PsiX psiX_;
    PsiY psiY_;
    LaplacePsi laplacePsi_;
    double R0_[2], Z0_[2];
    double zeta0_, f0_;
    int firstline_;
};

template< class Psi, class PsiX, class PsiY, class LaplacePsi>
struct SeparatrixOrthogonal
{
    typedef dg::OrthogonalTag metric_category;
    SeparatrixOrthogonal( Psi psi, PsiX psiX, PsiY psiY, LaplacePsi laplacePsi, double psi_0, //psi_0 must be the closed surface, 0 the separatrix
            double xX, double yX, double x0, double y0, int firstline ):
        psiX_(psiX), psiY_(psiY), laplacePsi_(laplacePsi),
        sep_( psi, psiX, psiY, xX, yX, x0, y0, firstline)
    {
        firstline_ = firstline;
        f0_ = sep_.get_f();
    }
    bool isConformal()const{return false;}
    bool isOrthogonal()const{return true;}
    double f0() const{return sep_.get_f();}
    void operator()(  //this one doesn't know if the separatrix comes to lie on a cell boundary or not
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         const unsigned nodeX0, const unsigned nodeX1, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) 
    {

        thrust::host_vector<double> r_init, z_init;
        sep_.compute_rzy( eta1d, nodeX0, nodeX1, r_init, z_init);
        orthogonal::detail::Nemov<PsiX, PsiY, LaplacePsi> nemov(psiX_, psiY_, laplacePsi_, f0_, firstline_);

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
            double psipX = psiX_(x[idx], y[idx]);
            double psipY = psiY_(x[idx], y[idx]);
            zetaX[idx] = f0_*psipX;
            zetaY[idx] = f0_*psipY;
            etaX[idx] = -h[idx]*psipY;
            etaY[idx] = +h[idx]*psipX;
        }
    }
    double laplace(double x, double y) {return f0_*laplacePsi_(x,y);}
    private:
    double R0_[2], Z0_[2];
    double f0_;
    int firstline_;
    PsiX psiX_;
    PsiY psiY_;
    LaplacePsi laplacePsi_;
    dg::detail::SeparatriX<Psi, PsiX, PsiY> sep_;
};

namespace orthogonal
{

template< class container>
struct GridX2d; 

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct GridX3d : public dg::GridX3d
{
    typedef dg::OrthogonalTag metric_category;
    typedef GridX2d<container> perpendicular_grid;

    template< class Generator>
    GridX3d( Generator generator, double psi_0, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx, dg::bc bcy):
        dg::GridX3d( 0,1, -2.*M_PI*fy/(1.-2.*fy), 2.*M_PI*(1.+fy/(1.-2.*fy)), 0., 2*M_PI, fx, fy, n, Nx, Ny, Nz, bcx, bcy, dg::PER),
        r_(this->size()), z_(r_), xr_(r_), xz_(r_), yr_(r_), yz_(r_)
    {
        assert( generator.isOrthogonal());
        construct( generator, psi_0, fx, n, Nx, Ny);
    }

    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}
    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& g_pp()const{return g_pp_;}
    const container& vol()const{return vol_;}
    const container& perpVol()const{return vol2d_;}
    perpendicular_grid perp_grid() const { return orthogonal::GridX2d<container>(*this);}
    private:
    template<class Generator>
    void construct( Generator generator, double psi_0, double fx, unsigned n, unsigned Nx, unsigned Ny )
    {
        const double x_0 = generator.f0()*psi_0;
        const double x_1 = -fx/(1.-fx)*x_0;
        init_X_boundaries( x_0, x_1);
        dg::Grid1d<double> gX1d( this->x0(), this->x1(), n, Nx, dg::DIR);
        thrust::host_vector<double> x_vec = dg::evaluate( dg::cooX1d, gX1d);
        dg::GridX1d gY1d( -this->fy()*2.*M_PI/(1.-2.*this->fy()), 2*M_PI+this->fy()*2.*M_PI/(1.-2.*this->fy()), this->fy(), this->n(), this->Ny(), dg::DIR);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::cooX1d, gY1d);
        thrust::host_vector<double> rvec, zvec, yrvec, yzvec, xrvec, xzvec;
        generator( x_vec, y_vec, gY1d.n()*gY1d.outer_N(), gY1d.n()*(gY1d.inner_N()+gY1d.outer_N()), rvec, zvec, xrvec, xzvec, yrvec, yzvec);

        unsigned Mx = this->n()*this->Nx(), My = this->n()*this->Ny();
        //now lift to 3D grid
        for( unsigned k=0; k<this->Nz(); k++)
            for( unsigned i=0; i<Mx*My; i++)
            {
                r_[k*Mx*My+i] = rvec[i];
                z_[k*Mx*My+i] = zvec[i];
                yr_[k*Mx*My+i] = yrvec[i];
                yz_[k*Mx*My+i] = yzvec[i];
                xr_[k*Mx*My+i] = xrvec[i];
                xz_[k*Mx*My+i] = xzvec[i];
            }
        construct_metric();
    }
    //compute metric elements from xr, xz, yr, yz, r and z
    void construct_metric()
    {
        //std::cout << "CONSTRUCTING METRIC\n";
        thrust::host_vector<double> tempxx( r_), tempxy(r_), tempyy(r_), tempvol(r_);
        for( unsigned idx=0; idx<this->size(); idx++)
        {
            tempxx[idx] = (xr_[idx]*xr_[idx]+xz_[idx]*xz_[idx]);
            tempxy[idx] = (yr_[idx]*xr_[idx]+yz_[idx]*xz_[idx]);
            tempyy[idx] = (yr_[idx]*yr_[idx]+yz_[idx]*yz_[idx]);
            tempvol[idx] = r_[idx]/sqrt(tempxx[idx]*tempyy[idx]-tempxy[idx]*tempxy[idx]);
        }
        g_xx_=tempxx, g_xy_=tempxy, g_yy_=tempyy, vol_=tempvol;
        dg::blas1::pointwiseDivide( tempvol, r_, tempvol);
        vol2d_ = tempvol;
        thrust::host_vector<double> ones = dg::evaluate( dg::one, *this);
        dg::blas1::pointwiseDivide( ones, r_, tempxx);
        dg::blas1::pointwiseDivide( tempxx, r_, tempxx); //1/R^2
        g_pp_=tempxx;
    }
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, g_pp_, vol_, vol2d_;
};

/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
template< class container>
struct GridX2d : public dg::GridX2d
{
    typedef dg::OrthogonalTag metric_category;
    template<class Generator>
    GridX2d(Generator generator, double psi_0, double fx, double fy, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx, dg::bc bcy):
        dg::GridX2d( 0, 1,-fy*2.*M_PI/(1.-2.*fy), 2*M_PI+fy*2.*M_PI/(1.-2.*fy), fx, fy, n, Nx, Ny, bcx, bcy)
    {
        orthogonal::GridX3d<container> g( generator, psi_0, fx,fy, n,Nx,Ny,1,bcx,bcy);
        init_X_boundaries( g.x0(),g.x1());
        r_=g.r(), z_=g.z(), xr_=g.xr(), xz_=g.xz(), yr_=g.yr(), yz_=g.yz();
        g_xx_=g.g_xx(), g_xy_=g.g_xy(), g_yy_=g.g_yy();
        vol2d_=g.perpVol();
    }
    GridX2d( const GridX3d<container>& g):
        dg::GridX2d( g.x0(), g.x1(), g.y0(), g.y1(), g.fx(), g.fy(), g.n(), g.Nx(), g.Ny(), g.bcx(), g.bcy())
    {
        unsigned s = this->size();
        r_.resize( s), z_.resize(s), xr_.resize(s), xz_.resize(s), yr_.resize(s), yz_.resize(s);
        g_xx_.resize( s), g_xy_.resize(s), g_yy_.resize(s), vol2d_.resize(s);
        for( unsigned i=0; i<s; i++)
        { r_[i]=g.r()[i], z_[i]=g.z()[i], xr_[i]=g.xr()[i], xz_[i]=g.xz()[i], yr_[i]=g.yr()[i], yz_[i]=g.yz()[i];}
        thrust::copy( g.g_xx().begin(), g.g_xx().begin()+s, g_xx_.begin());
        thrust::copy( g.g_xy().begin(), g.g_xy().begin()+s, g_xy_.begin());
        thrust::copy( g.g_yy().begin(), g.g_yy().begin()+s, g_yy_.begin());
        thrust::copy( g.perpVol().begin(), g.perpVol().begin()+s, vol2d_.begin());
    }
    const thrust::host_vector<double>& r()const{return r_;}
    const thrust::host_vector<double>& z()const{return z_;}
    const thrust::host_vector<double>& xr()const{return xr_;}
    const thrust::host_vector<double>& yr()const{return yr_;}
    const thrust::host_vector<double>& xz()const{return xz_;}
    const thrust::host_vector<double>& yz()const{return yz_;}

    const container& g_xx()const{return g_xx_;}
    const container& g_yy()const{return g_yy_;}
    const container& g_xy()const{return g_xy_;}
    const container& vol()const{return vol2d_;}
    const container& perpVol()const{return vol2d_;}
    private:
    thrust::host_vector<double> r_, z_, xr_, xz_, yr_, yz_;
    container g_xx_, g_xy_, g_yy_, vol2d_;
};

/**
 * @brief Integrates the equations for a field line and 1/B
 */ 
//struct XField
//{
//    XField( solovev::GeomParameters gp,const dg::GridX2d& gXY, const thrust::host_vector<double>& g):
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
//    solovev::GeomParameters gp_;
//    solovev::mod::PsipR  psipR_;
//    solovev::mod::PsipZ  psipZ_;
//    solovev::Ipol   ipol_;
//    solovev::InvB   invB_;
//    const dg::GridX2d gXY_;
//    thrust::host_vector<double> g_;
//    double R_X, Z_X;
//   
//};
//
}//namespace solovev

}//namespace dg

