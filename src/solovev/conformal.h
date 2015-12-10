#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/functions.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/operator.h"
#include "dg/backend/derivatives.h"
#include "dg/functors.h"
#include "dg/runge_kutta.h"
#include "dg/nullstelle.h"
#include "geometry.h"



namespace solovev
{

namespace detail
{



//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as 
//good as it can, i.e. until machine precision is reached
struct Fpsi
{
    Fpsi( const GeomParameters& gp, double psi_0): 
        gp_(gp), fieldRZYT_(gp), fieldRZtau_(gp), psi_0(psi_0) 
    {
        /**
         * @brief Find R such that \f$ \psi_p(R,0) = psi_0\f$
         *
         * Searches the range R_0 to R_0 + 2*gp.a
         * @param gp The geometry parameters
         * @param psi_0 The intended value for psi_p
         *
         * @return the value for R
         */
        solovev::Psip psip( gp);
        double min = gp.R_0, max = gp.R_0+2*gp.a, middle;
        double value_middle, value_max=psip(gp.R_0+2*gp.a, 0)-psi_0, value_min=psip(gp.R_0, 0) - psi_0;
        if( value_max*value_min>=0)
            throw dg::KeineNST_1D( min, max);
        double eps=max-min, eps_old = 2*eps;
        unsigned number =0;
        while( eps<eps_old)
        {
            eps_old = eps;
            value_middle = psip( middle = (min+max)/2., 0) - psi_0;
            if( value_middle == 0)              {max = min = middle; break;}
            else if( value_middle*value_max >0) max = middle;
            else                                min = middle;
            eps = max-min; number++;
        }
        //std::cout << eps<<" with "<<number<<" steps\n";
        R_init = (min+max)/2;
    }
    //finds the starting points for the integration in y direction
    void find_initial( double psi, double& R_0, double& Z_0) const
    {
        unsigned N = 50;
        thrust::host_vector<double> begin2d( 2, 0), end2d( begin2d), end2d_old(begin2d); 
        begin2d[0] = end2d[0] = end2d_old[0] = R_init;
        //std::cout << "In init function\n";
        double eps = 1e10, eps_old = 2e10;
        while( eps < eps_old && N<1e6 && eps > 1e-15)
        {
            //remember old values
            eps_old = eps;
            end2d_old = end2d;
            //compute new values
            N*=2;
            dg::stepperRK17( fieldRZtau_, begin2d, end2d, psi_0, psi, N);
            eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) + (end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
        }
        R_0 = end2d_old[0], Z_0 = end2d_old[1];
    }

    //compute f for a given psi between psi0 and psi1
    double construct_f( double psi, double& R_0, double& Z_0) const
    {
        find_initial( psi, R_0, Z_0);
        //std::cout << "Begin error "<<eps_old<<" with "<<N<<" steps\n";
        //std::cout << "In Stepper function:\n";
        //double y_old=0;
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        begin[0] = R_0, begin[1] = Z_0;
        //std::cout << begin[0]<<" "<<begin[1]<<" "<<begin[2]<<"\n";
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 50;
        //double y_eps;
        while( eps < eps_old && N < 1e6)
        {
            //remember old values
            eps_old = eps, end_old = end; //y_old = end[2];
            //compute new values
            N*=2;
            dg::stepperRK17( fieldRZYT_, begin, end, 0., 2*M_PI, N);
            eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
            //y_eps = sqrt( (y_old - end[2])*(y_old-end[2]));
            //std::cout << "\t error "<<eps<<" with "<<N<<" steps\t";
            //std::cout <<"error in y is "<<y_eps<<"\n";
        }
        double f_psi = 2.*M_PI/end_old[2];
        return f_psi;
    }
    double operator()( double psi)const
    {
        double R_0, Z_0; 
        return construct_f( psi, R_0, Z_0);
    }

    /**
     * @brief This function computes the integral x_1 = \int_{\psi_0}^{\psi_1} f(\psi) d\psi to machine precision
     *
     * @param gp The geometry parameters
     * @param psi_0 lower boundary 
     * @param psi_1 upper boundary 
     *
     * @return x1
     */
    double find_x1( double psi_1 ) const
    {
        unsigned P=3;
        double x1 = 0, x1_old = 0;
        double eps=1e10, eps_old=2e10;
        std::cout << "In x1 function\n";
        while(eps < eps_old && P < 20 && eps > 1e-15)
        {
            eps_old = eps; 
            x1_old = x1;

            P+=1;
            dg::Grid1d<double> grid( psi_0, psi_1, P, 1);
            thrust::host_vector<double> psi_vec = dg::evaluate( dg::coo1, grid);
            thrust::host_vector<double> f_vec(grid.size(), 0);
            thrust::host_vector<double> w1d = dg::create::weights(grid);
            for( unsigned i=0; i<psi_vec.size(); i++)
            {
                f_vec[i] = this->operator()( psi_vec[i]);
            }
            x1 = dg::blas1::dot( f_vec, w1d);

            eps = fabs(x1 - x1_old);
            std::cout << "X1 = "<<-x1<<" error "<<eps<<" with "<<P<<" polynomials\n";
        }
        return -x1_old;

    }

    //compute the vector of r and z - values that form one psi surface
    double compute_rzy( double psi, unsigned n, unsigned N, thrust::host_vector<double>& r, thrust::host_vector<double>& z, double& R_0, double& Z_0) const
    {
        dg::Grid1d<double> g1d( 0, 2*M_PI, n, N, dg::PER);
        thrust::host_vector<double> y_vec = dg::evaluate( dg::coo1, g1d);
        thrust::host_vector<double> r_old(n*N, 0), r_diff( r_old);
        thrust::host_vector<double> z_old(n*N, 0), z_diff( z_old);
        const thrust::host_vector<double> w1d = dg::create::weights( g1d);
        r.resize( n*N), z.resize(n*N);

        thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
        double f_psi = construct_f( psi, begin[0], begin[1]);
        R_0 = begin[0], Z_0 = begin[1];
        //std::cout <<f_psi<<" "<< psi_x[j] <<" "<< begin[0] << " "<<begin[1]<<"\t";
        FieldRZY fieldRZY(gp_);
        fieldRZY.set_f(f_psi);
        unsigned steps = 1;
        double eps = 1e10, eps_old=2e10;
        while( eps < eps_old)
        {
            eps_old = eps, r_old = r, z_old = z;
            dg::stepperRK17( fieldRZY, begin, end, 0, y_vec[0], steps);
            r[0] = end[0]; z[0] = end[1];
            //std::cout <<end[0]<<" "<< end[1] <<"\n";
            for( unsigned i=1; i<n*N; i++)
            {
                temp = end;
                dg::stepperRK17( fieldRZY, temp, end, y_vec[i-1], y_vec[i], steps);
                r[i] = end[0]; z[i] = end[1];
            }
            dg::blas1::axpby( 1., r, -1., r_old, r_diff);
            dg::blas1::axpby( 1., z, -1., z_old, z_diff);
            double er = dg::blas2::dot( r_diff, w1d, r_diff);
            double ez = dg::blas2::dot( z_diff, w1d, z_diff);
            eps =  sqrt( er + ez);
            std::cout << "error is "<<eps<<" with "<<steps<<" steps\n";
            steps*=2;
        }
        r = r_old, z = z_old;
        return f_psi;

    }
    private:
    const GeomParameters gp_;
    const FieldRZYT fieldRZYT_;
    const FieldRZtau fieldRZtau_;
    double R_init;
    const double psi_0;

};

//This struct computes -2pi/f with a fixed number of steps for all psi
struct FieldFinv
{
    FieldFinv( const GeomParameters& gp, double psi_0, unsigned N_steps = 500): 
        psi_0(psi_0), 
        fpsi_(gp, psi_0), fieldRZYT_(gp) 
            { }
    void operator()(const thrust::host_vector<double>& psi, thrust::host_vector<double>& fpsiM) const 
    { 
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        fpsi_.find_initial( psi[0], begin[0], begin[1]);
        //eps = 1e10, eps_old = 2e10;
        //N = 10;
        //double y_old;
        //while( eps < eps_old && N < 1e6)
        //{
        //    //remember old values
        //    eps_old = eps, end_old = end, y_old = end[2];
        //    //compute new values
        //    N*=2;
        //    dg::stepperRK17( fieldRZYT_, begin, end, 0., 2*M_PI, N);
        //    //eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
        //    eps = fabs( (y_old - end[2]));
        //    //std::cout << "F error "<<eps<<" with "<<N<<" steps\n";
        //    //std::cout <<"error in y is "<<y_eps<<"\n";
        //}

        //std::cout << begin[0]<<" "<<begin[1]<<" "<<begin[2]<<"\n";
        dg::stepperRK17( fieldRZYT_, begin, end, 0., 2*M_PI, 500);
        //eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
        fpsiM[0] = - end[2]/2./M_PI;
        //std::cout <<"fpsiMinverse is "<<fpsiM[0]<<" "<<-1./fpsi_(psi[0])<<" "<<eps<<"\n";
    }
    private:
    double psi_0;
    Fpsi fpsi_;
    FieldRZYT fieldRZYT_;
    thrust::host_vector<double> fpsi_neg_inv;
    unsigned N_steps;
};





struct Naive
{
    Naive( const dg::Grid2d<double>& g2d): dx_(dg::create::pidxpj(g2d.n())), dy_(dx_)
    {
        dg::Operator<double> tx( dg::create::pipj_inv(g2d.n())), ty(tx);
        dg::Operator<double> forward( g2d.dlt().forward()); 
        dg::Operator<double> backward( g2d.dlt().backward());
        tx*= 2./g2d.hx();
        ty*= 2./g2d.hy();
        dx_ = backward*tx*dx_*forward;
        dy_ = backward*ty*dy_*forward;
        Nx = g2d.Nx();
        Ny = g2d.Ny();
        n = g2d.n();
    }
    void dx( const thrust::host_vector<double>& in, thrust::host_vector<double>& out)
    {
        for( unsigned i=0; i<Ny*n; i++)
            for( unsigned j=0; j<Nx; j++)
                for( unsigned k=0; k<n; k++)
                {
                    out[i*Nx*n + j*n +k] = 0;
                    for( unsigned l=0; l<n; l++)
                        out[i*Nx*n + j*n +k] += dx_(k,l)*in[i*Nx*n+j*n+l];
                }
    }
    void dy( const thrust::host_vector<double>& in, thrust::host_vector<double>& out)
    {
        for( unsigned i=0; i<Ny; i++)
            for( unsigned k=0; k<n; k++)
                for( unsigned j=0; j<Nx*n; j++)
                {
                    out[i*Nx*n*n + k*Nx*n +j] = 0;
                    for( unsigned l=0; l<n; l++)
                        out[i*Nx*n*n + k*Nx*n +j] += dy_(k,l)*in[i*Nx*n*n+l*Nx*n+j];
                }
    }

    private:
    dg::Operator<double> dx_, dy_;
    unsigned Nx, Ny, n;

};


} //namespace detail


/**
 * @brief A three-dimensional grid based on "almost-conformal" coordinates by Ribeiro and Scott 2010
 */
struct ConformalRingGrid
{

    /**
     * @brief Construct 
     *
     * @param gp The geometric parameters define the magnetic field
     * @param psi_0 lower boundary for psi
     * @param psi_1 upper boundary for psi
     * @param n The dG number of polynomials
     * @param Nx The number of points in x-direction
     * @param Ny The number of points in y-direction
     * @param Nz The number of points in z-direction
     * @param bcx The boundary condition in x (y,z are periodic)
     */
    ConformalRingGrid( GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, unsigned Nz, dg::bc bcx): 
        g3d_( 0, detail::Fpsi(gp, psi_0).find_x1( psi_1), 0., 2*M_PI, 0., 2.*M_PI, n, Nx, Ny, Nz, bcx, dg::PER, dg::PER, dg::cartesian)
    { 
        //compute psi(x) for a grid on x 
        detail::FieldFinv fpsiM_(gp, psi_0, psi_1);
        dg::Grid1d<double> g1d_( g3d_.x0(), g3d_.x1(), g3d_.n(), g3d_.Nx(), g3d_.bcx());
        thrust::host_vector<double> x_vec = dg::evaluate( dg::coo1, g1d_);
        thrust::host_vector<double> psi_x(g3d_.n()*g3d_.Nx(), 0), psi_old(psi_x), psi_diff( psi_old);
        thrust::host_vector<double> w1d = dg::create::weights( g1d_);
        thrust::host_vector<double> begin(1,psi_0), end(begin), temp(begin);
        unsigned N = 1;
        double eps = 1e10; //eps_old=2e10;
        std::cout << "In psi function:\n";
        double x0=g3d_.x0(), x1 = x_vec[1];
        //while( eps <  eps_old && N < 1e6)
        while( eps >  1e-10 && N < 1e6)
        {
            //eps_old = eps;
            psi_old = psi_x; 
            x0 = 0, x1 = x_vec[0];

            dg::stepperRK6( fpsiM_, begin, end, x0, x1, N);
            psi_x[0] = end[0]; 
            for( unsigned i=1; i<g1d_.size(); i++)
            {
                temp = end;
                x0 = x_vec[i-1], x1 = x_vec[i];
                dg::stepperRK6( fpsiM_, temp, end, x0, x1, N);
                psi_x[i] = end[0];
            }
            temp = end;
            dg::stepperRK6(fpsiM_, temp, end, x1, g3d_.x1(),N);
            eps = fabs( end[0]-psi_1); 
            std::cout << "Effective Psi error is "<<eps<<" with "<<N<<" steps\n";
            N*=2;
        }
        f_x_.resize( psi_x.size());
        detail::Fpsi fpsi( gp, psi_0);
        //first compute boundary points in x
        double R_0, Z_0;
        fpsi.compute_rzy( psi_0, g3d_.n(), g3d_.Ny(), r_0y, z_0y, R_0, Z_0);
        fpsi.compute_rzy( psi_1, g3d_.n(), g3d_.Ny(), r_1y, z_1y, R_0, Z_0);
        construct_rz( fpsi, psi_x);
        construct_metric();
    }
    const thrust::host_vector<double> r()const{return r_;}
    const thrust::host_vector<double> z()const{return z_;}
    const thrust::host_vector<double> f_x()const{return f_x_;}
    const thrust::host_vector<double> g_xx()const{return g_xx_;}
    const thrust::host_vector<double> g_yy()const{return g_yy_;}
    const thrust::host_vector<double> g_xy()const{return g_xy_;}
    const thrust::host_vector<double> g_pp()const{return g_pp_;}
    const thrust::host_vector<double> vol()const{return vol_;}
    const dg::Grid3d<double>& grid() const{return g3d_;}
    private:
    void construct_rz( const detail::Fpsi& fpsi, thrust::host_vector<double>& psi_x) 
    {
        //construct f_x, r and z and the boundaries in y 
        r_.resize(g3d_.size()), z_.resize(g3d_.size());
        r_x0.resize( psi_x.size()), z_x0.resize( psi_x.size());
        const thrust::host_vector<double> w3d = dg::create::weights( g3d_);

        std::cout << "In grid function:\n";
        unsigned Nx = g3d_.n()*g3d_.Nx(), Ny = g3d_.n()*g3d_.Ny();
        for( unsigned i=0; i<Nx; i++)
        {
            thrust::host_vector<double> ry, zy;
            f_x_[i] = fpsi.compute_rzy( psi_x[i], g3d_.n(), g3d_.Ny(), ry, zy, r_x0[i], z_x0[i]);
            for( unsigned j=0; j<Ny; j++)
                r_[j*Nx+i] = ry[j], z_[j*Nx+i] = zy[j];
        }
        r_x1 = r_x0, z_x1 = z_x0; //periodic boundaries
        //now lift to 3D grid
        for( unsigned k=1; k<g3d_.Nz(); k++)
            for( unsigned i=0; i<Nx*Ny; i++)
            {
                r_[k*Nx*Ny+i] = r_[(k-1)*Nx*Ny+i];
                z_[k*Nx*Ny+i] = z_[(k-1)*Nx*Ny+i];
            }
    }

    void construct_metric( ) 
    {
        g_xx_.resize(g3d_.size()), g_xy_.resize( g3d_.size()), g_yy_.resize( g3d_.size()), g_pp_.resize( g3d_.size()), vol_.resize( g3d_.size());
        thrust::host_vector<double> r_x( r_), r_y(r_), z_x(r_), z_y(r_);
        thrust::host_vector<double> temp0( r_), temp1(r_);
        dg::EllSparseBlockMat dx = dg::create::dx( g3d_, dg::DIR, dg::centered);
        dg::EllSparseBlockMat dy = dg::create::dy( g3d_, dg::PER, dg::centered);
        //First lift the boundaries to a 3D grid
        thrust::host_vector<double> r_0( g3d_.size()), z_0(r_0), r_1(r_0), z_1(z_0);
        thrust::host_vector<double> r_tilde( r_0), z_tilde(r_0);
        thrust::host_vector<double> r_bar( r_0), z_bar(r_0), dx_r_bar(r_bar), dx_z_bar( r_bar);
        thrust::host_vector<double> x = dg::evaluate( dg::coo1, g3d_);
        unsigned Nx = g3d_.n()*g3d_.Nx(), Ny = g3d_.n()*g3d_.Ny();
        for( unsigned k=0; k<g3d_.Nz(); k++)
            for( unsigned i=0; i<Ny; i++)
                for( unsigned j=0; j<Nx; j++)
                {
                    r_0[k*Ny*Nx + i*Nx + j] = r_0y[i];
                    r_1[k*Ny*Nx + i*Nx + j] = r_1y[i];
                    z_0[k*Ny*Nx + i*Nx + j] = z_0y[i];
                    z_1[k*Ny*Nx + i*Nx + j] = z_1y[i];
                }
        //now compute \bar r = (R_1-R_0)/x1 * x + R_0
        dg::blas1::axpby( 1./g3d_.x1(), r_1, -1./g3d_.x1(), r_0, dx_r_bar);
        dg::blas1::pointwiseDot( x, dx_r_bar, temp0);
        dg::blas1::axpby( 1., temp0, 1., r_0, r_bar);
        dg::blas1::axpby( 1./g3d_.x1(), z_1, -1./g3d_.x1(), z_0, dx_z_bar);
        dg::blas1::pointwiseDot( x, dx_z_bar, temp0);
        dg::blas1::axpby( 1., temp0, 1., z_0, z_bar);
        //now compute \tilde r = r - \bar r
        dg::blas1::axpby( 1., r_ , -1., r_bar, r_tilde);
        dg::blas1::axpby( 1., z_ , -1., z_bar, z_tilde);
        //now compute derivatives
        dg::blas2::symv( dx, r_tilde, r_x);
        dg::blas1::axpby( 1., r_x, 1., dx_r_bar, r_x);
        dg::blas2::symv( dx, z_tilde, z_x);
        dg::blas1::axpby( 1., z_x, 1., dx_z_bar, z_x);
        dg::blas2::symv( dy, r_, r_y);
        dg::blas2::symv( dy, z_, z_y);
        //Now compute the linear interpolation of the boundaries
        dg::blas1::pointwiseDot( r_x, r_x, temp0);
        dg::blas1::pointwiseDot( z_x, z_x, temp1);
        dg::blas1::axpby( 1., temp0, 1., temp1, g_xx_);
        dg::blas1::pointwiseDot( r_x, r_y, temp0);
        dg::blas1::pointwiseDot( z_x, z_y, temp1);
        dg::blas1::axpby( 1., temp0, 1., temp1, g_xy_);
        dg::blas1::pointwiseDot( r_y, r_y, temp0);
        dg::blas1::pointwiseDot( z_y, z_y, temp1);
        dg::blas1::axpby( 1., temp0, 1., temp1, g_yy_);
        dg::blas1::pointwiseDot( g_xx_, g_yy_, temp0);
        dg::blas1::pointwiseDot( g_xy_, g_xy_, temp1);
        dg::blas1::axpby( 1., temp0, -1., temp1, vol_); //determinant
        //now invert to get contravariant elements
        dg::blas1::pointwiseDivide( g_xx_, vol_, g_xx_);
        dg::blas1::pointwiseDivide( g_xy_, vol_, g_xy_);
        dg::blas1::pointwiseDivide( g_yy_, vol_, g_yy_);
        g_xx_.swap( g_yy_);
        dg::blas1::scal( g_xy_, -1.);
        //compute real volume form
        dg::blas1::transform( vol_, vol_, dg::SQRT<double>());
        dg::blas1::pointwiseDot( r_, vol_, vol_);
        thrust::host_vector<double> ones = dg::evaluate( dg::one, g3d_);
        dg::blas1::pointwiseDivide( ones, r_, temp0);
        dg::blas1::pointwiseDivide( temp0, r_, g_pp_); //1/R^2

    }
    const dg::Grid3d<double> g3d_;
    thrust::host_vector<double> f_x_; //1d vector
    thrust::host_vector<double> r_, z_; //3d vector
    thrust::host_vector<double> g_xx_, g_xy_, g_yy_, g_pp_, vol_;
    
    //The following points might also be useful for external grid generation
    thrust::host_vector<double> r_0y, r_1y, z_0y, z_1y; //boundary points in x
    thrust::host_vector<double> r_x0, r_x1, z_x0, z_x1; //boundary points in y

};


}//namespace solovev
namespace dg{

    /**
     * @brief This function pulls back a function defined in cylindrical coordinates R,Z,\phi to the conformal coordinates x,y,\phi
     *
     * i.e. F(x,y,\phi) = f(R(x,y), Z(x,y), \phi)
     * @tparam TernaryOp The function object 
     * @param f The function defined on R,Z,\phi
     * @param g The grid
     *
     * @return A set of points representing F(x,y,\phi)
     */
template< class TernaryOp>
thrust::host_vector<double> pullback( TernaryOp f, const solovev::ConformalRingGrid& g)
{
    thrust::host_vector<double> vec( g.grid().size());
    unsigned size2d = g.grid().n()*g.grid().n()*g.grid().Nx()*g.grid().Ny();
    Grid1d<double> gz( g.grid().z0(), g.grid().z1(), 1, g.grid().Nz());
    thrust::host_vector<double> absz = create::abscissas( gz);
    for( unsigned k=0; k<g.grid().Nz(); k++)
        for( unsigned i=0; i<size2d; i++)
            vec[k*size2d+i] = f( g.r()[k*size2d+i], g.z()[k*size2d+i], absz[k]);
    return vec;
}
///@cond
thrust::host_vector<double> pullback( double (f)(double,double,double), const solovev::ConformalRingGrid& g)
{
    return pullback<double(double, double, double)>( f, g);
}
///@endcond
namespace create{

/**
 * @brief Create weights on a conformal grid
 *
 * The weights are the volume form times the weights on x,y,\phi
 * @param g The grid
 *
 * @return The weights
 */
thrust::host_vector<double> weights( const solovev::ConformalRingGrid& g)
{
    thrust::host_vector<double> vec = dg::create::weights( g.grid());
    for( unsigned i=0; i<vec.size(); i++)
        vec[i] *= g.vol()[i];
    return vec;
}

}//namespace create
}//namespace dg
