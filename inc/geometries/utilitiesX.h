#pragma once

namespace dg
{

namespace detail
{

    /**
     * @brief This struct finds and stores the X-point and can act in a root finding routine to find points on the perpendicular line through the X-point 
     */
struct XPointer
{
    XPointer( const solovev::GeomParameters& gp, double distance=1): fieldRZtau_(gp), psip_(gp), dist_(distance){
        solovev::HessianRZtau hessianRZtau(gp);
        R_X = gp.R_0-1.1*gp.triangularity*gp.a;
        Z_X = -1.1*gp.elongation*gp.a;
        thrust::host_vector<double> X(2,0), XN(X);
        X[0] = R_X, X[1] = Z_X;
        for( unsigned i=0; i<3; i++)
        {
            hessianRZtau.newton_iteration( X, XN);
            XN.swap(X);
        }
        R_X = X[0], Z_X = X[1];
        std::cout << "X-point set at "<<R_X<<" "<<Z_X<<"\n";
        R_i[0] = R_X + dist_, Z_i[0] = Z_X;
        R_i[1] = R_X    , Z_i[1] = Z_X + dist_;
        R_i[2] = R_X - dist_, Z_i[2] = Z_X;
        R_i[3] = R_X    , Z_i[3] = Z_X - dist_;
    }
    void set_quadrant( int quad){quad_ = quad;}
    double operator()( double x) const
    {
        thrust::host_vector<double> begin(2), end(2), end_old(2);
        begin[0] = R_i[quad_], begin[1] = Z_i[quad_];
        double eps = 1e10, eps_old = 2e10;
        unsigned N=10;
        if( quad_ == 0 || quad_ == 2) { begin[1] += x;}
        else if( quad_ == 1 || quad_ == 3) { begin[0] += x;}

        double psi0 = psip_(begin[0], begin[1]);
        while( (eps < eps_old || eps > 1e-4 ) && eps > 1e-7)
        {
            eps_old = eps; end_old = end;
            N*=2; 
            dg::stepperRK17( fieldRZtau_, begin, end, psi0, 0, N);

            eps = sqrt( (end[0]-end_old[0])*(end[0]-end_old[0]) + (end[1]-end_old[1])*(end[1]-end_old[1]));
            if( isnan(eps)) { eps = eps_old/2.; end = end_old; }
        }
        if( quad_ == 0 || quad_ == 2){ return end_old[1] - Z_X;}
        return end_old[0] - R_X;
    }
    void point( double& R, double& Z, double x)
    {
        if( quad_ == 0 || quad_ == 2){ R = R_i[quad_], Z= Z_i[quad_] +x;}
        else if (quad_ == 1 || quad_ == 3) { R = R_i[quad_] + x, Z = Z_i[quad_];}
    }

    private:
    int quad_;
    solovev::FieldRZtau fieldRZtau_;
    solovev::Psip psip_;
    double R_X, Z_X;
    double R_i[4], Z_i[4];
    double dist_;
};

//compute psi(x) and f(x) for given discretization of x and a fpsiMinv functor
//doesn't integrate over the x-point
//returns psi_1
template <class XFieldFinv>
double construct_psi_values( XFieldFinv fpsiMinv, const solovev::GeomParameters& gp, 
        const double psi_0, const double x_0, const thrust::host_vector<double>& x_vec, const double x_1, unsigned idxX,
        thrust::host_vector<double>& psi_x, 
        thrust::host_vector<double>& f_x_)
{
    f_x_.resize( x_vec.size()), psi_x.resize( x_vec.size());
    thrust::host_vector<double> psi_old(psi_x), psi_diff( psi_old);
    //thrust::host_vector<double> w1d = dg::create::weights( g1d_);
    unsigned N = 1;
    std::cout << "In psi function:\n";
    //double x0=this->x0(), x1 = x_vec[0];
    double x0, x1;
    //const unsigned idxX = inner_Nx()*this->n();
    const double psi_const = fpsiMinv.find_psi( x_vec[idxX]);
    double psi_1_numerical;
    double eps = 1e10;//, eps_old=2e10;
    //while( eps <  eps_old && N < 1e6)
    while( eps >  1e-8 && N < 1e6 )
    {
       // eps_old = eps; 
        psi_old = psi_x; 
        x0 = x_0, x1 = x_vec[0];

        thrust::host_vector<double> begin(1,psi_0), end(begin), temp(begin);
        dg::stepperRK6( fpsiMinv, begin, end, x0, x1, N);
        psi_x[0] = end[0]; fpsiMinv(end,temp); f_x_[0] = temp[0];
        for( unsigned i=1; i<idxX; i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            dg::stepperRK6( fpsiMinv, temp, end, x0, x1, N);
            psi_x[i] = end[0]; fpsiMinv(end,temp); f_x_[i] = temp[0];
            //std::cout << "FOUND PSI "<<end[0]<<"\n";
        }
        end[0] = psi_const;
        //std::cout << "FOUND PSI "<<end[0]<<"\n";
        psi_x[idxX] = end[0]; fpsiMinv(end,temp); f_x_[idxX] = temp[0];
        for( unsigned i=idxX+1; i<x_vec.size(); i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            dg::stepperRK6( fpsiMinv, temp, end, x0, x1, N);
            psi_x[i] = end[0]; fpsiMinv(end,temp); f_x_[i] = temp[0];
            //std::cout << "FOUND PSI "<<end[0]<<"\n";
        }
        temp = end;
        dg::stepperRK6(fpsiMinv, temp, end, x1, x_1,N);
        psi_1_numerical = end[0];
        dg::blas1::axpby( 1., psi_x, -1., psi_old, psi_diff);
        //eps = sqrt( dg::blas2::dot( psi_diff, w1d, psi_diff)/ dg::blas2::dot( psi_x, w1d, psi_x));
        eps = sqrt( dg::blas1::dot( psi_diff, psi_diff)/ dg::blas1::dot( psi_x, psi_x));
        //psi_1_numerical_ = psi_0 + dg::blas1::dot( f_x_, w1d);

        //eps = fabs( psi_1_numerical-psi_1); 
        //std::cout << "Effective absolute Psi error is "<<psi_1_numerical-psi_1<<" with "<<N<<" steps\n"; 
        std::cout << "Effective Psi error is "<<eps<<" with "<<N<<" steps\n"; 
        //std::cout << "psi 1               is "<<psi_1_numerical_<<"\n"; 
        N*=2;
    }
    return psi_1_numerical;
}
} //namespace detail
} //namespace dg

