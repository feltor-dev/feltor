#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/functions.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/operator.h"
#include "dg/backend/derivatives.h"
#include "dg/functors.h"
#include "dg/runge_kutta.h"
#include "dg/nullstelle.h"
#include "dg/geometry/geometry.h"
#include "generator.h"
#include "utilities.h"


namespace dg
{
namespace geo
{
///@cond
namespace orthogonal
{

namespace detail
{

//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as 
//good as it can, i.e. until machine precision is reached
struct Fpsi
{
    
    //firstline = 0 -> conformal, firstline = 1 -> equalarc
    Fpsi( const BinaryFunctorsLvl1& psi, double x0, double y0, int firstline): 
        psip_(psi), fieldRZYTconf_(psi, x0, y0),fieldRZYTequl_(psi, x0, y0), fieldRZtau_(psi)
    {
        X_init = x0, Y_init = y0;
        while( fabs( psi.dfx()(X_init, Y_init)) <= 1e-10 && fabs( psi.dfy()( X_init, Y_init)) <= 1e-10)
            X_init +=  1.; 
        firstline_ = firstline;
    }
    //finds the starting points for the integration in y direction
    void find_initial( double psi, double& R_0, double& Z_0) 
    {
        unsigned N = 50;
        thrust::host_vector<double> begin2d( 2, 0), end2d( begin2d), end2d_old(begin2d); 
        begin2d[0] = end2d[0] = end2d_old[0] = X_init;
        begin2d[1] = end2d[1] = end2d_old[1] = Y_init;
        double eps = 1e10, eps_old = 2e10;
        while( (eps < eps_old || eps > 1e-7) && eps > 1e-14)
        {
            eps_old = eps; end2d_old = end2d;
            N*=2; dg::stepperRK17( fieldRZtau_, begin2d, end2d, psip_.f()(X_init, Y_init), psi, N);
            eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) + (end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
        }
        X_init = R_0 = end2d_old[0], Y_init = Z_0 = end2d_old[1];
        //std::cout << "In init function error: psi(R,Z)-psi0: "<<psip_(X_init, Y_init)-psi<<"\n";
    }

    //compute f for a given psi between psi0 and psi1
    double construct_f( double psi, double& R_0, double& Z_0) 
    {
        find_initial( psi, R_0, Z_0);
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        begin[0] = R_0, begin[1] = Z_0;
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 50;
        while( (eps < eps_old || eps > 1e-7)&& eps > 1e-14)
        {
            eps_old = eps, end_old = end; N*=2; 
            if( firstline_ == 0)
                dg::stepperRK17( fieldRZYTconf_, begin, end, 0., 2*M_PI, N);
            if( firstline_ == 1)
                dg::stepperRK17( fieldRZYTequl_, begin, end, 0., 2*M_PI, N);
            eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
        }
        //std::cout << "\t error "<<eps<<" with "<<N<<" steps\t";
        //std::cout <<end_old[2] << " "<<end[2] << "error in y is "<<y_eps<<"\n";
        double f_psi = 2.*M_PI/end_old[2];
        return f_psi;
    }
    double operator()( double psi)
    {
        double R_0, Z_0; 
        return construct_f( psi, R_0, Z_0);
    }

    private:
    int firstline_;
    double X_init, Y_init;
    BinaryFunctorsLvl1 psip_;
    dg::geo::ribeiro::FieldRZYT fieldRZYTconf_;
    dg::geo::equalarc::FieldRZYT fieldRZYTequl_;
    dg::geo::FieldRZtau fieldRZtau_;

};

//compute the vector of r and z - values that form one psi surface
//assumes y_0 = 0
void compute_rzy( const BinaryFunctorsLvl1& psi, const thrust::host_vector<double>& y_vec,
        thrust::host_vector<double>& r, 
        thrust::host_vector<double>& z, 
        double R_0, double Z_0, double f_psi, int mode ) 
{

    thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old);
    thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size());
    thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
    begin[0] = R_0, begin[1] = Z_0;
    //std::cout <<f_psi<<" "<<" "<< begin[0] << " "<<begin[1]<<"\t";
    dg::geo::ribeiro::FieldRZY fieldRZYconf(psi);
    dg::geo::equalarc::FieldRZY fieldRZYequi(psi);
    fieldRZYconf.set_f(f_psi);
    fieldRZYequi.set_f(f_psi);
    unsigned steps = 1;
    double eps = 1e10, eps_old=2e10;
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-14)
    {
        //begin is left const
        eps_old = eps, r_old = r, z_old = z;
        if(mode==0)dg::stepperRK17( fieldRZYconf, begin, end, 0, y_vec[0], steps);
        if(mode==1)dg::stepperRK17( fieldRZYequi, begin, end, 0, y_vec[0], steps);
        r[0] = end[0], z[0] = end[1];
        for( unsigned i=1; i<y_vec.size(); i++)
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
        //std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        //temp = end; 
        //if(mode==0)dg::stepperRK17( fieldRZYconf, temp, end, y_vec[y_vec.size()-1], 2.*M_PI, steps);
        //if(mode==1)dg::stepperRK17( fieldRZYequi, temp, end, y_vec[y_vec.size()-1], 2.*M_PI, steps);
        //std::cout << "abs. error is "<<sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]))<<"\n";
        steps*=2;
    }
    r = r_old, z = z_old;

}

//This struct computes -2pi/f with a fixed number of steps for all psi
//and provides the Nemov algorithm for orthogonal grid
struct Nemov
{
    Nemov( const BinaryFunctorsLvl2 psi, double f0, int mode):
        f0_(f0), mode_(mode),
        psip_(psi) { }
    void initialize( 
        const thrust::host_vector<double>& r_init, //1d intial values
        const thrust::host_vector<double>& z_init, //1d intial values
        thrust::host_vector<double>& h_init) //,
    //    thrust::host_vector<double>& hr_init,
    //    thrust::host_vector<double>& hz_init)
    {
        unsigned size = r_init.size(); 
        h_init.resize( size);//, hr_init.resize( size), hz_init.resize( size);
        for( unsigned i=0; i<size; i++)
        {
            if(mode_ == 0)
                h_init[i] = f0_;
            if(mode_ == 1)
            {
                double psipR = psip_.dfx()(r_init[i], z_init[i]), 
                       psipZ = psip_.dfy()(r_init[i], z_init[i]);
                double psip2 = (psipR*psipR+psipZ*psipZ);
                h_init[i]  = f0_/sqrt(psip2); //equalarc
            }
            //double laplace = psipRR_(r_init[i], z_init[i]) + 
                             //psipZZ_(r_init[i], z_init[i]);
            //hr_init[i] = -f0_*laplace/psip2*psipR;
            //hz_init[i] = -f0_*laplace/psip2*psipZ;
        }
    }

    void operator()(const std::vector<thrust::host_vector<double> >& y, std::vector<thrust::host_vector<double> >& yp) 
    { 
        //y[0] = R, y[1] = Z, y[2] = h, y[3] = hr, y[4] = hz
        unsigned size = y[0].size();
        double psipR, psipZ, psip2;
        for( unsigned i=0; i<size; i++)
        {
            psipR = psip_.dfx()(y[0][i], y[1][i]), psipZ = psip_.dfy()(y[0][i], y[1][i]);
            //psipRR = psipRR_(y[0][i], y[1][i]), psipRZ = psipRZ_(y[0][i], y[1][i]), psipZZ = psipZZ_(y[0][i], y[1][i]);
            psip2 = f0_*(psipR*psipR+psipZ*psipZ);
            yp[0][i] = psipR/psip2;
            yp[1][i] = psipZ/psip2;
            yp[2][i] = y[2][i]*( - psip_.dfxx()(y[0][i], y[1][i]) - psip_.dfyy()(y[0][i], y[1][i]) )/psip2;
            //yp[3][i] = ( -(2.*psipRR+psipZZ)*y[3][i] - psipRZ*y[4][i] - laplacePsipR_(y[0][i], y[1][i])*y[2][i])/psip2;
            //yp[4][i] = ( -psipRZ*y[3][i] - (2.*psipZZ+psipRR)*y[4][i] - laplacePsipZ_(y[0][i], y[1][i])*y[2][i])/psip2;
        }
    }
    private:
    double f0_;
    int mode_;
    BinaryFunctorsLvl2 psip_;
};

template<class Nemov>
void construct_rz( Nemov nemov, 
        double x_0, //the x value that corresponds to the first psi surface
        const thrust::host_vector<double>& x_vec,  //1d x values
        const thrust::host_vector<double>& r_init, //1d intial values of the first psi surface
        const thrust::host_vector<double>& z_init, //1d intial values of the first psi surface
        thrust::host_vector<double>& r, 
        thrust::host_vector<double>& z, 
        thrust::host_vector<double>& h//,
        //thrust::host_vector<double>& hr,
        //thrust::host_vector<double>& hz
    )
{
    unsigned N = 1;
    double eps = 1e10, eps_old=2e10;
    std::vector<thrust::host_vector<double> > begin(3); //begin(5);
    thrust::host_vector<double> h_init( r_init.size(), 0.);
    //thrust::host_vector<double> h_init, hr_init, hz_init;
    nemov.initialize( r_init, z_init, h_init);//, hr_init, hz_init);
    begin[0] = r_init, begin[1] = z_init, 
    begin[2] = h_init; //begin[3] = hr_init, begin[4] = hz_init;
    //now we have the starting values 
    std::vector<thrust::host_vector<double> > end(begin), temp(begin);
    unsigned sizeX = x_vec.size(), sizeY = r_init.size();
    unsigned size2d = x_vec.size()*r_init.size();
    r.resize(size2d), z.resize(size2d), h.resize(size2d); //hr.resize(size2d), hz.resize(size2d);
    double x0=x_0, x1 = x_vec[0];
    thrust::host_vector<double> r_old(r), r_diff( r), z_old(z), z_diff(z);
    while( (eps < eps_old || eps > 1e-6) && eps > 1e-13)
    {
        r_old = r, z_old = z; eps_old = eps; 
        temp = begin;
        //////////////////////////////////////////////////
        for( unsigned i=0; i<sizeX; i++)
        {
            x0 = i==0?x_0:x_vec[i-1], x1 = x_vec[i];
            //////////////////////////////////////////////////
            dg::stepperRK17( nemov, temp, end, x0, x1, N);
            for( unsigned j=0; j<sizeY; j++)
            {
                unsigned idx = j*sizeX+i;
                 r[idx] = end[0][j],  z[idx] = end[1][j];
                //hr[idx] = end[3][j], hz[idx] = end[4][j];
                 h[idx] = end[2][j]; 
            }
            //////////////////////////////////////////////////
            temp = end;
        }
        dg::blas1::axpby( 1., r, -1., r_old, r_diff);
        dg::blas1::axpby( 1., z, -1., z_old, z_diff);
        dg::blas1::pointwiseDot( r_diff, r_diff, r_diff);
        dg::blas1::pointwiseDot( 1., z_diff, z_diff, 1., r_diff);
        eps = sqrt( dg::blas1::dot( r_diff, r_diff)/sizeX/sizeY); //should be relative to the interpoint distances
        //std::cout << "Effective Absolute diff error is "<<eps<<" with "<<N<<" steps\n"; 
        N*=2;
    }

}

} //namespace detail

}//namespace orthogonal
///@endcond


/**
 * @brief Generate a simple orthogonal grid 
 *
 * Psi is the radial coordinate and you can choose various discretizations of the first line
 * @ingroup generators_geo
 */
struct SimpleOrthogonal : public aGenerator2d
{
    /**
     * @brief Construct a simple orthogonal grid 
     *
     * @param psi \f$\psi(x,y)\f$ is the flux function and its derivatives in Cartesian coordinates (x,y)
     * @param psi_0 first boundary 
     * @param psi_1 second boundary
     * @param x0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param y0 a point in the inside of the ring bounded by psi0 (shouldn't be the O-point)
     * @param firstline This parameter indicates the adaption type used to create the orthogonal grid: 0 is no adaption, 1 is an equalarc adaption
     */
    SimpleOrthogonal(const BinaryFunctorsLvl2& psi, double psi_0, double psi_1, double x0, double y0, int firstline =0):
        psi_(psi)
    {
        assert( psi_1 != psi_0);
        firstline_ = firstline;
        orthogonal::detail::Fpsi fpsi(psi, x0, y0, firstline);
        f0_ = fabs( fpsi.construct_f( psi_0, R0_, Z0_));
        if( psi_1 < psi_0) f0_*=-1;
        lz_ =  f0_*(psi_1-psi_0);
    }

    /**
     * @brief The grid constant
     *
     * @return f_0 is the grid constant  s.a. width() )
     */
    double f0() const{return f0_;}
    virtual SimpleOrthogonal* clone() const{return new SimpleOrthogonal(*this);}

    private:
     // length of zeta-domain (f0*(psi_1-psi_0))
    virtual double do_width() const{return lz_;}
    virtual double do_height() const{return 2.*M_PI;}
    virtual bool do_isOrthogonal() const{return true;}
    virtual void do_generate( 
         const thrust::host_vector<double>& zeta1d, 
         const thrust::host_vector<double>& eta1d, 
         thrust::host_vector<double>& x, 
         thrust::host_vector<double>& y, 
         thrust::host_vector<double>& zetaX, 
         thrust::host_vector<double>& zetaY, 
         thrust::host_vector<double>& etaX, 
         thrust::host_vector<double>& etaY) const
    {
        thrust::host_vector<double> r_init, z_init;
        orthogonal::detail::compute_rzy( psi_, eta1d, r_init, z_init, R0_, Z0_, f0_, firstline_);
        orthogonal::detail::Nemov nemov(psi_, f0_, firstline_);
        thrust::host_vector<double> h;
        orthogonal::detail::construct_rz(nemov, 0., zeta1d, r_init, z_init, x, y, h);
        unsigned size = x.size();
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
    BinaryFunctorsLvl2 psi_;
    double f0_, lz_, R0_, Z0_;
    int firstline_;
};

}//namespace geo
}//namespace dg
