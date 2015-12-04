#pragma once

#include "dg/backend/grid.h"
#include "dg/backend/functions.h"
#include "dg/backend/interpolation.cuh"
#include "dg/runge_kutta.h"
#include "dg/nullstelle.h"
#include "geometry.h"



namespace solovev
{

namespace detail
{


/**
 * @brief Find R such that \f$ \psi_p(R,0) = psi_0\f$
 *
 * @param gp
 * @param psi_0
 *
 * @return 
 */
double find_initial_R( const GeomParameters& gp, double psi_0)
{
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
    return (min+max)/2;
}

struct Fpsi
{

    Fpsi( const GeomParameters& gp, double psi_0): 
        fieldRZYT_(gp), fieldRZtau_(gp), 
        R_init( detail::find_initial_R( gp, psi_0)), psi_0(psi_0) 
    { }

    //compute f for a given psi between psi0 and psi1
    double construct_f( double psi, double& R_0, double& Z_0) const
    {
        unsigned N = 50;
        thrust::host_vector<double> begin2d( 2, 0), end2d( begin2d), end2d_old(begin2d); 
        begin2d[0] = end2d[0] = end2d_old[0] = R_init;
        //std::cout << "In init function\n";
        double eps = 1e10, eps_old = 2e10;
        while( eps < eps_old && N<1e6)
        {
            //remember old values
            eps_old = eps;
            end2d_old = end2d;
            //compute new values
            N*=2;
            dg::stepperRK4( fieldRZtau_, begin2d, end2d, psi_0, psi, N);
            eps = sqrt( (end2d[0]-end2d_old[0])*(end2d[0]-end2d_old[0]) + (end2d[1]-end2d_old[1])*(end2d[1]-end2d_old[1]));
        }
        //std::cout << "Begin error "<<eps_old<<" with "<<N<<" steps\n";
        //std::cout << "In Stepper function:\n";
        //double y_old=0;
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        R_0 = begin[0] = end2d_old[0], Z_0 = begin[1] = end2d_old[1];
        eps = 1e10, eps_old = 2e10;
        N = 50;
        //double y_eps;
        while( eps < eps_old && N < 1e6)
        {
            //remember old values
            eps_old = eps, end_old = end; //y_old = end[2];
            //compute new values
            N*=2;
            dg::stepperRK4( fieldRZYT_, begin, end, 0., 2*M_PI, N);
            eps = sqrt( (end[0]-begin[0])*(end[0]-begin[0]) + (end[1]-begin[1])*(end[1]-begin[1]));
            //y_eps = sqrt( (y_old - end[2])*(y_old-end[2]));
            //std::cout << "\t error "<<eps<<" with "<<N<<" steps\t";
            //std::cout <<"error in y is "<<y_eps<<"\n";
        }
        double f_psi = 2.*M_PI/end_old[2];

        return f_psi;
    }
    double operator()( double psi)const{double R_0, Z_0; return construct_f( psi, R_0, Z_0);}

    private:
    const FieldRZYT fieldRZYT_;
    const FieldRZtau fieldRZtau_;
    const double R_init;
    const double psi_0;

};

struct FieldFinv
{
    FieldFinv( const GeomParameters& gp, double psi_0, double psi_1):psi_0(psi_0), psi_1(psi_1), fpsi_(gp, psi_0){
        //Fpsi fpsi(gp, psi_0);
        //P_=2;
        //double x1 = 0, x1_old = 0;
        //double eps=1e10, eps_old=2e10;
        //std::cout << "In Inverse function\n";
        //thrust::host_vector<double> psi_vec;
        //while(eps < eps_old && P_ < 20)
        //{
        //    eps_old = eps; 
        //    x1_old = x1;

        //    P_+=1;
        //    dg::Grid1d<double> grid( psi_0, psi_1, P_, 1);
        //    psi_vec = dg::evaluate( dg::coo1, grid);
        //    fpsi_neg_inv.resize( grid.size(), 0);
        //    thrust::host_vector<double> w1d = dg::create::weights(grid);
        //    for( unsigned i=0; i<psi_vec.size(); i++)
        //    {
        //        fpsi_neg_inv[i] = -1./fpsi( psi_vec[i]);
        //    }
        //    x1 = dg::blas1::dot( fpsi_neg_inv, w1d);

        //    eps = fabs(x1 - x1_old);
        //    std::cout << "F1 = "<<x1<<" error "<<eps<<" with "<<P_<<" polynomials\n";
        //}
        ////take the optimum
        //P_-=1;
        //dg::Grid1d<double> grid( psi_0, psi_1, P_, 1);
        //psi_vec = dg::evaluate( dg::coo1, grid);
        //fpsi_neg_inv.resize( grid.size(), 0);
        //thrust::host_vector<double> w1d = dg::create::weights(grid);
        //for( unsigned i=0; i<psi_vec.size(); i++)
        //{
        //    fpsi_neg_inv[i] = -1./fpsi( psi_vec[i]);
        //}
    }
    inline void operator()(const thrust::host_vector<double>& psi, thrust::host_vector<double>& fpsiM) const { 
        ////determine normalized psi
        ////std::cout << "psi "<<psi[0]<<" "<<psi_0<<" "<<psi_1<<"\n";
        //double psi_in = psi[0];
        //if(psi[0] < psi_0) psi_in = psi_0;
        //if(psi[0] > psi_1) psi_in = psi_1;
        //double psin = 2*(psi_in-psi_0)/(psi_1-psi_0)-1.;
        ////compute p_i(psi_n) for i=0,...,P-1
        //std::vector<double> coeffs = dg::create::detail::coefficients( psin, P_); 
        //double sum=0;
        //for(unsigned i=0; i<P_;i++)
        //{
        //    sum += coeffs[i]*fpsi_neg_inv[i];
        //}
        //fpsiM[0] = sum;
        fpsiM[0] = -1./fpsi_(psi[0]);
    }
    private:
    double psi_0, psi_1;
    Fpsi fpsi_;
    thrust::host_vector<double> fpsi_neg_inv;
    unsigned P_;
};

double find_x1( const GeomParameters& gp, double psi_0, double psi_1 )
{
    Fpsi fpsi(gp, psi_0);
    unsigned P=3;
    double x1 = 0, x1_old = 0;
    double eps=1e10, eps_old=2e10;
    std::cout << "In x1 function\n";
    while(eps < eps_old && P < 20)
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
            f_vec[i] = fpsi( psi_vec[i]);
        }
        x1 = dg::blas1::dot( f_vec, w1d);

        eps = fabs(x1 - x1_old);
        std::cout << "X1 = "<<-x1<<" error "<<eps<<" with "<<P<<" polynomials\n";
    }
    return -x1_old;

}

} //namespace detail


struct ConformalRingGrid
{

    ConformalRingGrid( GeomParameters gp, double psi_0, double psi_1, unsigned n, unsigned Nx, unsigned Ny, dg::bc bcx): 
        g2d_( 0, detail::find_x1(gp, psi_0, psi_1), 0, 2*M_PI, n, Nx, Ny, bcx, dg::PER),
        psi_0(psi_0), psi_1(psi_1),
        gp_(gp), 
        psi_x(g2d_.n()*g2d_.Nx(),0){ }

    //compute psi for a grid on x 
    void construct_psi( ) 
    {
        detail::FieldFinv fpsiM_(gp_, psi_0, psi_1);
        //assert( psi.size() == g2d_.n()*g2d_.Nx());
        dg::Grid1d<double> g1d_( g2d_.x0(), g2d_.x1(), g2d_.n(), g2d_.Nx(), g2d_.bcx());
        thrust::host_vector<double> x_vec = dg::evaluate( dg::coo1, g1d_);
        thrust::host_vector<double> psi_old(g2d_.n()*g2d_.Nx(), 0), psi_diff( psi_old);
        thrust::host_vector<double> w1d = dg::create::weights( g1d_);
        thrust::host_vector<double> begin(1,psi_0), end(begin), temp(begin);
        unsigned N = 1;
        double eps = 1e10, eps_old=2e10;
        std::cout << "In psi function:\n";
        double x0=g2d_.x0(), x1 = x_vec[1];
        //while( eps <  eps_old && N < 1e6)
        while( eps >  1e-10 && N < 1e6)
        {
            eps_old = eps;
            psi_old = psi_x; 
            x0 = 0, x1 = x_vec[0];

            dg::stepperRK4( fpsiM_, begin, end, x0, x1, N);
            psi_x[0] = end[0]; 
            for( unsigned i=1; i<g1d_.size(); i++)
            {
                temp = end;
                x0 = x_vec[i-1], x1 = x_vec[i];
                dg::stepperRK4( fpsiM_, temp, end, x0, x1, N);
                psi_x[i] = end[0];
            }
            dg::blas1::axpby( 1., psi_x, -1., psi_old, psi_diff);
            double epsi = dg::blas2::dot( psi_diff, w1d, psi_diff);
            eps =  sqrt( epsi);
            std::cout << "Psi error is "<<eps<<" with "<<N<<" steps\n";
            N*=2;
        }
        //psi_x = psi_old;
    }
    void construct_rz( thrust::host_vector<double>& r, thrust::host_vector<double>& z) const
    {
        assert( r.size() == g2d_.size() && z.size() == g2d_.size());
        const unsigned Nx = g2d_.n()*g2d_.Nx();
        thrust::host_vector<double> y_vec = dg::evaluate( dg::coo2, g2d_);
        thrust::host_vector<double> r_old(g2d_.size(), 0), r_diff( r_old);
        thrust::host_vector<double> z_old(g2d_.size(), 0), z_diff( z_old);
        const thrust::host_vector<double> w2d = dg::create::weights( g2d_);
        unsigned N = 1;
        double eps = 1e10, eps_old=2e10;
        std::cout << "In grid function:\n";
        while( eps <  eps_old && N < 1e6)
        {
            eps_old = eps;
            r_old = r; z_old = z;
            N*=2;
            for( unsigned j=0; j<Nx; j++)
            {
                thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin);
                FieldRZY fieldRZY(gp_);
                detail::Fpsi fpsi( gp_, psi_0);
                double f_psi = fpsi.construct_f( psi_x[j], begin[0], begin[1]);
                //std::cout <<f_psi<<" "<< psi_x[j] <<" "<< begin[0] << " "<<begin[1]<<"\t";
                fieldRZY.set_f(f_psi);
                double y0 = 0, y1 = y_vec[0]; 
                dg::stepperRK4( fieldRZY, begin, end, y0, y1, N);
                r[0+j] = end[0]; z[0+j] = end[1];
                //std::cout <<end[0]<<" "<< end[1] <<"\n";
                for( unsigned i=1; i<g2d_.n()*g2d_.Ny(); i++)
                {
                    temp = end;
                    y0 = y_vec[(i-1)*Nx+j], y1 = y_vec[i*Nx+j];
                    dg::stepperRK4( fieldRZY, temp, end, y0, y1, N);
                    r[i*Nx+j] = end[0]; z[i*Nx+j] = end[1];
                    //std::cout << y0<<" "<<y1<<" "<<temp[0]<<" "<<temp[1]<<" "<<end[0]<<" "<<end[1]<<"\n";
                }
                //std::cout << r[j] <<" "<< z[j] << " "<<r[Nx + j]<<" "<<z[Nx + j]<<"\n";
            }
            dg::blas1::axpby( 1., r, -1., r_old, r_diff);
            dg::blas1::axpby( 1., z, -1., z_old, z_diff);
            double er = dg::blas2::dot( r_diff, w2d, r_diff);
            double ez = dg::blas2::dot( z_diff, w2d, z_diff);
            eps =  sqrt( er + ez);
            std::cout << "error is "<<eps<<" with "<<N<<" steps\n";
        }
        r = r_old, z = z_old;
    }

    const dg::Grid2d<double>& grid() const{return g2d_;}




    private:
    const dg::Grid2d<double> g2d_;
    const double psi_0, psi_1;
    const GeomParameters gp_;
    thrust::host_vector<double> psi_x;

};

}//namespace solovev
