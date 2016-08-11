#pragma once

namespace dg
{

namespace detail
{

//compute psi(x) and f(x) for given discretization of x and a fpsiMinv functor
//doesn't integrate over the x-point
//returns psi_1
template <class FieldFinv>
void construct_psi_values( FieldFinv fpsiMinv, const solovev::GeomParameters& gp, 
        const double psi_0, const double psi_1, const double x_0, const thrust::host_vector<double>& x_vec, const double x_1,
        thrust::host_vector<double>& psi_x, 
        thrust::host_vector<double>& f_x_)
{
    f_x_.resize( x_vec.size()), psi_x.resize( x_vec.size());
    //thrust::host_vector<double> w1d = dg::create::weights( g1d_);
    thrust::host_vector<double> begin(1,psi_0), end(begin), temp(begin);
    unsigned N = 1;
    double eps = 1e10, eps_old=2e10;
    //std::cout << "In psi function:\n";
    double x0=x_0, x1 = x_vec[0];
    //while( eps <  eps_old && N < 1e6)
    while( fabs(eps - eps_old) >  1e-10 && N < 1e6)
    {
        eps_old = eps;
        //psi_old = psi_x; 
        x0 = x_0, x1 = x_vec[0];

        dg::stepperRK6( fpsiMinv, begin, end, x0, x1, N);
        psi_x[0] = end[0]; fpsiMinv(end,temp); f_x_[0] = temp[0];
        for( unsigned i=1; i<x_vec.size(); i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            dg::stepperRK6( fpsiMinv, temp, end, x0, x1, N);
            psi_x[i] = end[0]; fpsiMinv(end,temp); f_x_[i] = temp[0];
        }
        temp = end;
        dg::stepperRK6(fpsiMinv, temp, end, x1, x_1,N);
        //double psi_1_numerical = psi_0 + dg::blas1::dot( f_x_, w1d);
        double psi_1_numerical = end[0];
        eps = fabs( psi_1_numerical-psi_1); 
        //std::cout << "Effective absolute Psi error is "<<psi_1_numerical-psi_1<<" with "<<N<<" steps\n"; 
        //std::cout << "Effective relative Psi error is "<<fabs(eps-eps_old)<<" with "<<N<<" steps\n"; 
        N*=2;
    }

}

} //namespace detail
} //namespace dg

