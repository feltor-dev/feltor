#pragma once

///@cond
namespace dg
{

namespace detail
{

//compute psi(x) and f(x) for given discretization of x and a fpsiMinv functor
//doesn't integrate over the x-point
//returns psi_1
template <class FieldFinv>
void construct_psi_values( FieldFinv fpsiMinv, 
        const double psi_0, const double psi_1, const double x_0, const thrust::host_vector<double>& x_vec, const double x_1,
        thrust::host_vector<double>& psi_x, 
        thrust::host_vector<double>& f_x_)
{
    f_x_.resize( x_vec.size()), psi_x.resize( x_vec.size());
    thrust::host_vector<double> begin(1,psi_0), end(begin), temp(begin);
    unsigned N = 1;
    double eps = 1e10, eps_old=2e10;
    //std::cout << "In psi function:\n";
    double x0=x_0, x1 = psi_1>psi_0? x_vec[0]:-x_vec[0];
    while( (eps <  eps_old || eps > 1e-8) && eps > 1e-14) //1e-8 < eps < 1e-14
    {
        eps_old = eps;
        x0 = x_0, x1 = x_vec[0];
        if( psi_1<psi_0) x1*=-1;
        dg::stepperRK17( fpsiMinv, begin, end, x0, x1, N);
        psi_x[0] = end[0]; fpsiMinv(end,temp); f_x_[0] = temp[0];
        for( unsigned i=1; i<x_vec.size(); i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            if( psi_1<psi_0) x0*=-1, x1*=-1;
            dg::stepperRK17( fpsiMinv, temp, end, x0, x1, N);
            psi_x[i] = end[0]; fpsiMinv(end,temp); f_x_[i] = temp[0];
        }
        temp = end;
        dg::stepperRK17(fpsiMinv, temp, end, x1, psi_1>psi_0?x_1:-x_1,N);
        double psi_1_numerical = end[0];
        eps = fabs( psi_1_numerical-psi_1); 
        //std::cout << "Effective Psi error is "<<eps<<" with "<<N<<" steps\n"; 
        N*=2;
    }

}

//compute the vector of r and z - values that form one psi surface
//assumes that the initial line is perpendicular 
template <class Fpsi, class FieldRZYRYZY>
void compute_rzy(Fpsi fpsi, FieldRZYRYZY fieldRZYRYZY, 
        double psi, const thrust::host_vector<double>& y_vec, 
        thrust::host_vector<double>& r, 
        thrust::host_vector<double>& z, 
        thrust::host_vector<double>& yr, 
        thrust::host_vector<double>& yz,  
        thrust::host_vector<double>& xr, 
        thrust::host_vector<double>& xz,  
        double& R_0, double& Z_0, double& f, double& fp ) 
{
    thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old), yr_old(r_old), xr_old(r_old);
    thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old), yz_old(r_old), xz_old(z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size()), yr.resize(y_vec.size()), yz.resize(y_vec.size()), xr.resize(y_vec.size()), xz.resize(y_vec.size());

    //now compute f and starting values 
    thrust::host_vector<double> begin( 4, 0), end(begin), temp(begin);
    const double f_psi = fpsi.construct_f( psi, begin[0], begin[1]);
    fieldRZYRYZY.set_f(f_psi);
    double fprime = fpsi.f_prime( psi);
    fieldRZYRYZY.set_fp(fprime);
    fieldRZYRYZY.initialize( begin[0], begin[1], begin[2], begin[3]);
    R_0 = begin[0], Z_0 = begin[1];
    //std::cout <<f_psi<<" "<<" "<< begin[0] << " "<<begin[1]<<"\t";
    unsigned steps = 1;
    double eps = 1e10, eps_old=2e10;
    while( eps < eps_old)
    {
        //begin is left const
        eps_old = eps, r_old = r, z_old = z, yr_old = yr, yz_old = yz, xr_old = xr, xz_old = xz;
        dg::stepperRK17( fieldRZYRYZY, begin, end, 0, y_vec[0], steps);
        r[0] = end[0], z[0] = end[1], yr[0] = end[2], yz[0] = end[3];
        fieldRZYRYZY.derive( r[0], z[0], xr[0], xz[0]);
        //std::cout <<end[0]<<" "<< end[1] <<"\n";
        for( unsigned i=1; i<y_vec.size(); i++)
        {
            temp = end;
            dg::stepperRK17( fieldRZYRYZY, temp, end, y_vec[i-1], y_vec[i], steps);
            r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
            fieldRZYRYZY.derive( r[i], z[i], xr[i], xz[i]);
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
        steps*=2;
    }
    r = r_old, z = z_old, yr = yr_old, yz = yz_old, xr = xr_old, xz = xz_old;
    f = f_psi;

}

} //namespace detail


} //namespace dg
///@endcond

