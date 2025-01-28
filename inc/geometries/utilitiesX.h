#pragma once

#include "dg/algorithm.h"
#include "utilities.h"

namespace dg
{
namespace geo
{

///@cond
namespace detail{
struct Monitor : public aCylindricalFunctor<Monitor>
{
    //computes a + eps * b
    Monitor( double value, double eps_value, double R_X, double Z_X, double sigmaR, double sigmaZ):
        m_value(value), m_eps_value(eps_value),
        m_cauchy(R_X, Z_X, sigmaR, sigmaZ, 1){}
    double do_compute( double x, double y)const
    {
        return m_value+m_cauchy(x,y)*m_eps_value;
    }
    private:
    double m_value, m_eps_value;
    dg::Cauchy m_cauchy;

};
struct DivMonitor : public aCylindricalFunctor<DivMonitor>
{
    //computes a*epsX + b*epsY
    DivMonitor( double valueX, double valueY, double R_X, double Z_X, double sigmaR, double sigmaZ):
        m_valueX(valueX), m_valueY(valueY),
        m_cauchy(R_X, Z_X, sigmaR, sigmaZ, 1){}
    double do_compute( double x, double y)const
    {
        return m_valueX*m_cauchy.dx(x,y)+m_valueY*m_cauchy.dy(x,y);
    }
    private:
    double m_valueX, m_valueY;
    dg::Cauchy m_cauchy;

};
}//namespace detail
///@endcond

/**
 * @brief construct a monitor metric in which the Laplacian vanishes at the X-point
 *
 * calls the \c findXpoint function to find the X-point
 * @param psi the flux functions
 * @param R_X start value on input, X-point on output
 * @param Z_X start value on input, X-point on output
 * @param radiusX size of bump in x direction
 * @param radiusY size of bump in y-direction
 *
 * @return a metric tensor and its derivatives
 */
inline CylindricalSymmTensorLvl1 make_Xbump_monitor( const CylindricalFunctorsLvl2& psi, double& R_X, double& Z_X, double radiusX, double radiusY)
{
    findXpoint( psi, R_X, Z_X);
    double x = R_X, y = Z_X;
    double psixy    = psi.dfxy()(x,y), psixx = psi.dfxx()(x,y), psiyy = psi.dfyy()(x,y);
    double sumpsi   = psixx + psiyy;
    double diffpsi  = psixx - psiyy;
    double alpha    = (psixy*psixy - psixx*psiyy)*(diffpsi*diffpsi + 4.*psixy*psixy);

    double gxx = (-psiyy*diffpsi + 2.*psixy*psixy)/sqrt(alpha);
    double gyy = ( psixx*diffpsi + 2.*psixy*psixy)/sqrt(alpha);
    double gxy = (               -   sumpsi*psixy)/sqrt(alpha);
    detail::Monitor xx(1, gxx-1, x,y, radiusX, radiusY);
    detail::Monitor xy(0, gxy-0, x,y, radiusX, radiusY);
    detail::Monitor yy(1, gyy-1, x,y, radiusX, radiusY);
    detail::DivMonitor divX(gxx-1, gxy, x,y, radiusX, radiusY);
    detail::DivMonitor divY(gxy, gyy-1, x,y, radiusX, radiusY);
    CylindricalSymmTensorLvl1 chi( xx, xy, yy, divX, divY);
    //double laplace = psi.dfxx()(x,y)*chi.xx()(x,y)+2.*psi.dfxy()(x,y)*chi.xy()(x,y)+psi.dfyy()(x,y)*chi.yy()(x,y)
    //        + chi.divX()(x,y)*psi.dfx()(x,y) + chi.divY()(x,y)*psi.dfy()(x,y);
    //std::cout << "Laplace at X-point "<<laplace<<std::endl;

    return chi;
}
/**
 * @brief construct a monitor metric in which the Laplacian vanishes at the X-point
 *
 * calls the \c findXpoint function to find the X-point
 * @param psi the flux functions
 * @param R_X start value on input, X-point on output
 * @param Z_X start value on input, X-point on output
 *
 * @return a metric tensor and its derivatives
 */
inline CylindricalSymmTensorLvl1 make_Xconst_monitor( const CylindricalFunctorsLvl2& psi, double& R_X, double& Z_X)
{
    findXpoint( psi, R_X, Z_X);
    double x = R_X, y = Z_X;
    double psixy    = psi.dfxy()(x,y), psixx = psi.dfxx()(x,y), psiyy = psi.dfyy()(x,y);
    double sumpsi   = psixx + psiyy;
    double diffpsi  = psixx - psiyy;
    double alpha    = (psixy*psixy - psixx*psiyy)*(diffpsi*diffpsi + 4.*psixy*psixy);

    double gxx = (-psiyy*diffpsi + 2.*psixy*psixy)/sqrt(alpha);
    double gyy = ( psixx*diffpsi + 2.*psixy*psixy)/sqrt(alpha);
    double gxy = (               -   sumpsi*psixy)/sqrt(alpha);
    Constant xx(gxx);
    Constant xy(gxy);
    Constant yy(gyy);
    Constant divX(0);
    Constant divY(0);
    CylindricalSymmTensorLvl1 chi( xx, xy, yy, divX, divY);
    //std::cout << "px  "<<psi.dfx()(x,y)<<" py "<<psi.dfy()(x,y)<<std::endl;
    //std::cout << "gxx "<<chi.xx()(x,y)<<" gxy "<< chi.xy()(x,y)<<" gyy "<<chi.yy()(x,y)<<std::endl;
    //std::cout << "pxx "<<psixx<<" pxy "<< psixy<<" pyy "<<psiyy<<std::endl;
    //double laplace = psi.dfxx()(x,y)*chi.xx()(x,y)+2.*psi.dfxy()(x,y)*chi.xy()(x,y)+psi.dfyy()(x,y)*chi.yy()(x,y);
    //        //+ chi.divX()(x,y)*psi.dfx()(x,y) + chi.divY()(x,y)*psi.dfy()(x,y);
    //std::cout << "Laplace at X-point "<<laplace<<std::endl;
    return chi;
}


///@cond
namespace detail
{


/**
 * @brief This struct finds and stores the X-point and can act in a root finding routine to find points on the perpendicular line through the X-point
 */
struct XCross
{
    XCross( const CylindricalFunctorsLvl1& psi, double R_X, double Z_X, double distance=1): fieldRZtau_(psi), psip_(psi), dist_(distance)
    {
        R_X_ = R_X, Z_X_ = Z_X;
        //std::cout << "X-point set at "<<R_X_<<" "<<Z_X_<<"\n";
        R_i[0] = R_X_ + dist_, Z_i[0] = Z_X_;
        R_i[1] = R_X_    , Z_i[1] = Z_X_ + dist_;
        R_i[2] = R_X_ - dist_, Z_i[2] = Z_X_;
        R_i[3] = R_X_    , Z_i[3] = Z_X_ - dist_;
    }
    /**
     * @brief Set the quadrant in which operator() searches for perpendicular line
     *
     * @param quad 0 ( R_X + 1), 1 ( Z_X + 1), 2 ( R_X -1 ), 3 ( Z_X - 1)
     */
    void set_quadrant( int quad){quad_ = quad;}
    double operator()( double x) const
    {
        std::array<double,2> begin, end, end_old;
        begin[0] = R_i[quad_], begin[1] = Z_i[quad_];
        double eps = 1e10, eps_old = 2e10;
        unsigned N=10;
        if( quad_ == 0 || quad_ == 2) { begin[1] += x;}
        else if( quad_ == 1 || quad_ == 3) { begin[0] += x;}

        double psi0 = psip_.f()(begin[0], begin[1]);
        while( (eps < eps_old || eps > 1e-4 ) && eps > 1e-10)
        {
            eps_old = eps; end_old = end;
            N*=2;
            using Vec = std::array<double,2>;
            dg::SinglestepTimeloop<Vec>( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                begin), fieldRZtau_).integrate_steps( psi0, begin, 0, end, N);

            eps = sqrt( (end[0]-end_old[0])*(end[0]-end_old[0]) +
                    (end[1]-end_old[1])*(end[1]-end_old[1]));
            if( std::isnan(eps)) { eps = eps_old/2.; end = end_old; }
        }
        if( quad_ == 0 || quad_ == 2){ return end_old[1] - Z_X_;}
        return end_old[0] - R_X_;
    }
    /**
     * @brief This is to determine the actual point coordinates from the root-finding
     *
     * @param R
     * @param Z
     * @param x
     */
    void point( double& R, double& Z, double x)
    {
        if( quad_ == 0 || quad_ == 2){ R = R_i[quad_], Z= Z_i[quad_] +x;}
        else if (quad_ == 1 || quad_ == 3) { R = R_i[quad_] + x, Z = Z_i[quad_];}
    }

    private:
    int quad_;
    dg::geo::FieldRZtau fieldRZtau_;
    CylindricalFunctorsLvl1 psip_;
    double R_X_, Z_X_;
    double R_i[4], Z_i[4];
    double dist_;
};

//compute the vector of r and z - values that form one psi surface
//assumes y_0 = 0
template <class FpsiX, class FieldRZYRYZY>
void computeX_rzy(FpsiX fpsi, FieldRZYRYZY fieldRZYRYZY,
        double psi, const thrust::host_vector<double>& y_vec,
        const unsigned nodeX0, const unsigned nodeX1,
        thrust::host_vector<double>& r, //output r - values
        thrust::host_vector<double>& z, //output z - values
        thrust::host_vector<double>& yr,
        thrust::host_vector<double>& yz,
        thrust::host_vector<double>& xr,
        thrust::host_vector<double>& xz,
        double* R_0, double* Z_0,  //2 output coords on perp line
        double& f_psi,  //output f
        bool verbose = false
        )
{
    thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old), yr_old(r_old), xr_old(r_old);
    thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old), yz_old(r_old), xz_old(z_old);
    r.resize( y_vec.size()), z.resize(y_vec.size()), yr.resize(y_vec.size()), yz.resize(y_vec.size()), xr.resize(y_vec.size()), xz.resize(y_vec.size());
    //now compute f and starting values
    std::array<double, 4> begin{ {0,0,0,0} }, end(begin), temp(begin);
    const double fprime = fpsi.f_prime( psi);
    f_psi = fpsi.construct_f(psi, R_0, Z_0);
    fieldRZYRYZY.set_f(f_psi);
    fieldRZYRYZY.set_fp(fprime);
    using Vec = std::array<double,4>;
    dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>( "Feagin-17-8-10",
                begin), fieldRZYRYZY); // stores reference to fieldRZYRZYZ
    unsigned steps = 1; double eps = 1e10, eps_old=2e10;
    while( (eps < eps_old||eps > 1e-7) && eps > 1e-11)
    {
        eps_old = eps, r_old = r, z_old = z, yr_old = yr, yz_old = yz, xr_old = xr, xz_old = xz;
        ////////////////////////bottom left region/////////////////////
        if( nodeX0 != 0)
        {
            if(psi<0)begin[0] = R_0[1], begin[1] = Z_0[1];
            else     begin[0] = R_0[0], begin[1] = Z_0[0];
            fieldRZYRYZY.initialize( begin[0], begin[1], begin[2], begin[3]);
            unsigned i=nodeX0-1;
            odeint.integrate_steps( 0, begin, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
            fieldRZYRYZY.derive(r[i], z[i], xr[i], xz[i]);
        }
        for( int i=nodeX0-2; i>=0; i--)
        {
            temp = end;
            odeint.integrate_steps( y_vec[i+1],temp, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
            fieldRZYRYZY.derive(r[i], z[i], xr[i], xz[i]);
        }
        ////////////////middle region///////////////////////////
        begin[0] = R_0[0], begin[1] = Z_0[0];
        fieldRZYRYZY.initialize( begin[0], begin[1], begin[2], begin[3]);
        unsigned i=nodeX0;
        odeint.integrate_steps( 0, begin, y_vec[i], end, steps);
        r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
        fieldRZYRYZY.derive(r[i], z[i], xr[i], xz[i]);
        for( unsigned i=nodeX0+1; i<nodeX1; i++)
        {
            temp = end;
            odeint.integrate_steps( y_vec[i-1], temp, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
            fieldRZYRYZY.derive(r[i], z[i], xr[i], xz[i]);
        }
        temp = end;
        odeint.integrate_steps( y_vec[nodeX1-1], temp, 2.*M_PI, end, steps);
        if( psi <0)
            eps = sqrt( (end[0]-R_0[0])*(end[0]-R_0[0]) + (end[1]-Z_0[0])*(end[1]-Z_0[0]));
        else
            eps = sqrt( (end[0]-R_0[1])*(end[0]-R_0[1]) + (end[1]-Z_0[1])*(end[1]-Z_0[1]));
        if(verbose)std::cout << "abs. error is "<<eps<<" with "<<steps<<" steps\n";
        ////////////////////bottom right region
        if( nodeX0 != 0)
        {
            begin[0] = R_0[1], begin[1] = Z_0[1];
            fieldRZYRYZY.initialize( begin[0], begin[1], begin[2], begin[3]);
            unsigned i=nodeX1;
            odeint.integrate_steps( 2.*M_PI, begin, y_vec[i], end, steps);
            r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
            fieldRZYRYZY.derive(r[i], z[i], xr[i], xz[i]);
        }
        for( unsigned i=nodeX1+1; i<y_vec.size(); i++)
        {
            temp = end;
            odeint.integrate_steps( y_vec[i-1], temp, y_vec[i] ,end, steps);
            r[i] = end[0], z[i] = end[1], yr[i] = end[2], yz[i] = end[3];
            fieldRZYRYZY.derive(r[i], z[i], xr[i], xz[i]);
        }
        //compute error in R,Z only
        dg::blas1::axpby( 1., r, -1., r_old, r_diff);
        dg::blas1::axpby( 1., z, -1., z_old, z_diff);
        double er = dg::blas1::dot( r_diff, r_diff);
        double ez = dg::blas1::dot( z_diff, z_diff);
        double ar = dg::blas1::dot( r, r);
        double az = dg::blas1::dot( z, z);
        eps =  sqrt( er + ez)/sqrt(ar+az);
        if(verbose)std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
        if( std::isnan(eps)) { eps = eps_old/2.; }
        steps*=2;
    }
    r = r_old, z = z_old, yr = yr_old, yz = yz_old, xr = xr_old, xz = xz_old;
}


//compute psi(x) and f(x) for given discretization of x and a fpsiMinv functor
//doesn't integrate over the x-point
//returns psi_1
template <class XFieldFinv>
double construct_psi_values( XFieldFinv fpsiMinv,
        const double psi_0, const double x_0, const thrust::host_vector<double>& x_vec, const double x_1, unsigned idxX, //idxX is the number of x_vec[i] < 0
        thrust::host_vector<double>& psi_x, bool verbose = false )
{
    psi_x.resize( x_vec.size());
    thrust::host_vector<double> psi_old(psi_x), psi_diff( psi_old);
    unsigned N = 1;
    //std::cout << "In psi function:\n";
    double x0, x1;
    const double psi_const = fpsiMinv.find_psi( x_vec[idxX]);
    double psi_1_numerical=0;
    double eps = 1e10, eps_old=2e10;
    dg::SinglestepTimeloop<double> odeint(
            dg::RungeKutta<double>("Feagin-17-8-10", 0.), fpsiMinv);
    while( (eps <  eps_old || eps > 1e-8) && eps > 1e-11) //1e-8 < eps < 1e-14
    {
        eps_old = eps;
        psi_old = psi_x;
        x0 = x_0, x1 = x_vec[0];

        double begin(psi_0), end(begin), temp(begin);
        odeint.integrate_steps( x0, begin, x1, end, N);
        psi_x[0] = end; fpsiMinv(0.,end,temp);
        for( unsigned i=1; i<idxX; i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            odeint.integrate_steps( x0, temp, x1, end, N);
            psi_x[i] = end; fpsiMinv(0.,end,temp);
            //std::cout << "FOUND PSI "<<end[0]<<"\n";
        }
        end = psi_const;
        //std::cout << "FOUND PSI "<<end[0]<<"\n";
        psi_x[idxX] = end; fpsiMinv(0.,end,temp);
        for( unsigned i=idxX+1; i<x_vec.size(); i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            odeint.integrate_steps( x0, temp, x1, end, N);
            psi_x[i] = end; fpsiMinv(0.,end,temp);
            //std::cout << "FOUND PSI "<<end[0]<<"\n";
        }
        temp = end;
        odeint.integrate_steps(x1, temp, x_1, end,N);
        psi_1_numerical = end;
        dg::blas1::axpby( 1., psi_x, -1., psi_old, psi_diff);
        //eps = sqrt( dg::blas2::dot( psi_diff, w1d, psi_diff)/ dg::blas2::dot( psi_x, w1d, psi_x));
        eps = sqrt( dg::blas1::dot( psi_diff, psi_diff)/ dg::blas1::dot( psi_x, psi_x));

        if(verbose)std::cout << "Effective Psi error is "<<eps<<" with "<<N<<" steps\n";
        //std::cout << "psi 1               is "<<psi_1_numerical<<std::endl;
        N*=2;
    }
    return psi_1_numerical;
}



//!ATTENTION: choosing h on separatrix is a mistake if LaplacePsi does not vanish at X-point
struct PsipSep
{
    PsipSep( const CylindricalFunctor& psi): psip_(psi), Z_(0){}
    void set_Z( double z){ Z_=z;}
    double operator()(double R) { return psip_(R, Z_);}
    private:
    CylindricalFunctor psip_;
    double Z_;
};

//!ATTENTION: choosing h on separatrix is a mistake if LaplacePsi does not vanish at X-point
//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as
//good as it can, i.e. until machine precision is reached (like FpsiX just for separatrix)
struct SeparatriX
{
    SeparatriX( const CylindricalFunctorsLvl1& psi, const CylindricalSymmTensorLvl1& chi, double xX, double yX, double x0, double y0, int firstline, bool verbose=false):
        mode_(firstline),
        fieldRZYequi_(psi, chi), fieldRZYTequi_(psi, x0, y0, chi), fieldRZYZequi_(psi, chi),
        fieldRZYconf_(psi, chi), fieldRZYTconf_(psi, x0, y0, chi), fieldRZYZconf_(psi, chi), m_verbose( verbose)
    {
        //find four points on the separatrix and construct y coordinate at those points and at last construct f
        //////////////////////////////////////////////
        double R_X = xX; double Z_X = yX;
        double boxR = 0.05, boxZ = 0.01; // determine bisection boundaries
        std::array<double,4> R_min={R_X, R_X*(1.0-boxR), R_X*(1.0-boxR), R_X};
        std::array<double,4> R_max={R_X*(1.0+boxR), R_X, R_X, R_X*(1.0+boxR)};
        std::array<double,4> Z_0={Z_X+boxZ*R_X, Z_X+boxZ*R_X, Z_X-boxZ*R_X, Z_X-boxZ*R_X};
        PsipSep psip_sep( psi.f());
        for( unsigned i=0; i<4; i++)
        {
            psip_sep.set_Z( Z_0[i]);
            dg::bisection1d( psip_sep, R_min[i], R_max[i], 1e-13);
            R_i[i] = (R_min[i]+R_max[i])/2., Z_i[i] = Z_0[i];
            if(m_verbose)std::cout << "Found "<<i+1<<"st point "<<R_i[i]
                                   <<" "<<Z_i[i]<<"\n";
        }
        //now measure y distance to X-point
        std::array<double, 3> begin2d{ {0,0,0} }, end2d(begin2d);
        for( int i=0; i<4; i++)
        {
            unsigned N = 1;
            begin2d[0] = end2d[0] = R_i[i];
            begin2d[1] = end2d[1] = Z_i[i];
            begin2d[2] = end2d[2] = 0.;
            double eps = 1e10, eps_old = 2e10;
            //double y=0, y_old=0;
            //difference to X-point isn't much better than 1e-5
            while( (eps < eps_old || eps > 5e-5))
            {
                eps_old = eps; N*=2;// y_old=y;
                using Vec = std::array<double,3>;
                dg::SinglestepTimeloop<Vec> odeint;
                if( mode_ == 0)
                    odeint.construct( dg::RungeKutta<Vec>( "Fehlberg-6-4-5",
                                begin2d), fieldRZYZconf_);
                if(mode_==1)
                    odeint.construct( dg::RungeKutta<Vec>( "Fehlberg-6-4-5",
                                begin2d), fieldRZYZequi_);
                odeint.integrate_steps( Z_i[i], begin2d,
                                Z_X, end2d, N);
                //y=end2d[2];
                //eps = fabs((y-y_old)/y_old);
                eps = sqrt( (end2d[0]-R_X)*(end2d[0]-R_X))/R_X;
                if( std::isnan(eps)) { eps = eps_old/2.; }
                //std::cout << "Found y_i["<<i<<"]: "<<y<<" with eps = "<<eps<<" and "<<N<<" steps and diff "<<fabs(end2d[0]-R_X)/R_X<<"\n";
            }
            //remember last call
            y_i[i] = end2d[2];
            if( i==0 || i == 2)
                y_i[i] *= -1;//these were integrated against y direction
            if(m_verbose)std::cout << "Found |y_i["<<i<<"]|: "<<y_i[i]<<" with eps = "<<eps<<" and "<<N<<" steps and diff "<<fabs(end2d[0]-R_X)/R_X<<"\n";
        }

        f_psi_ = construct_f( );
        y_i[0]*=f_psi_, y_i[1]*=f_psi_, y_i[2]*=f_psi_, y_i[3]*=f_psi_;
        fieldRZYequi_.set_f(f_psi_);
        fieldRZYconf_.set_f(f_psi_);
    }

    double get_f( ) const{return f_psi_;}

    //compute the vector of r and z - values that form the separatrix
    void compute_rzy( const thrust::host_vector<double>& y_vec,
            const unsigned nodeX0, const unsigned nodeX1,
            thrust::host_vector<double>& r, //same size as y_vec on output
            thrust::host_vector<double>& z ) const
    {
        ///////////////////////////find y coordinate line//////////////
        std::array<double, 2> begin{ {0,0} }, end(begin), temp(begin);
        thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old);
        thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old);
        r.resize( y_vec.size()), z.resize(y_vec.size());
        unsigned steps = 1; double eps = 1e10, eps_old=2e10;
        using Vec = std::array<double,2>;
        dg::SinglestepTimeloop<Vec> odeint;
        if( mode_ == 0)
            odeint.construct( dg::RungeKutta<Vec>("Feagin-17-8-10", begin),
                    fieldRZYconf_);
        if( mode_ == 1)
            odeint.construct( dg::RungeKutta<Vec>("Feagin-17-8-10", begin),
                    fieldRZYequi_);
        while( (eps < eps_old||eps > 1e-7) && eps > 1e-11)
        {
            eps_old = eps, r_old = r, z_old = z;
            //////////////////////bottom right region/////////////////////
            if( nodeX0 != 0) //integrate to start point
            {
                begin[0] = R_i[3], begin[1] = Z_i[3];
                odeint.integrate_steps( -y_i[3], begin, y_vec[nodeX0-1], end,
                        N_steps_);
                r[nodeX0-1] = end[0], z[nodeX0-1] = end[1];
            }
            for( int i=nodeX0-2; i>=0; i--)
            {
                temp = end;
                odeint.integrate_steps( y_vec[i+1], temp, y_vec[i], end,steps);
                r[i] = end[0], z[i] = end[1];
            }
            ////////////////middle region///////////////////////////
            begin[0] = R_i[0], begin[1] = Z_i[0];
            odeint.integrate_steps( y_i[0], begin, y_vec[nodeX0], end, N_steps_);
            r[nodeX0] = end[0], z[nodeX0] = end[1];
            for( unsigned i=nodeX0+1; i<nodeX1; i++)
            {
                temp = end;
                odeint.integrate_steps( y_vec[i-1], temp, y_vec[i], end, steps);
                r[i] = end[0], z[i] = end[1];
            }
            temp = end;
            odeint.integrate_steps( y_vec[nodeX1-1], temp, 2.*M_PI-y_i[1], end,
                    N_steps_);
            eps = sqrt( (end[0]-R_i[1])*(end[0]-R_i[1]) +
                    (end[1]-Z_i[1])*(end[1]-Z_i[1]));
            //std::cout << "abs. error is "<<eps<<" with "<<steps<<" steps\n";
            ////////////////////bottom left region

            if( nodeX0!= 0)
            {
                begin[0] = R_i[2], begin[1] = Z_i[2];
                odeint.integrate_steps( 2.*M_PI+y_i[2], begin, y_vec[nodeX1],
                        end, N_steps_);
                r[nodeX1] = end[0], z[nodeX1] = end[1];
            }
            for( unsigned i=nodeX1+1; i<y_vec.size(); i++)
            {
                temp = end;
                odeint.integrate_steps( y_vec[i-1], temp, y_vec[i], end, steps);
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
            if(m_verbose)std::cout << "rel. Separatrix error is "<<eps<<" with "<<steps<<" steps\n";
            steps*=2;
        }
        r = r_old, z = z_old;
    }
    private:
    //compute f for psi=0
    double construct_f( )
    {
        if(m_verbose)std::cout << "In construct f function!\n";

        std::array<double, 3> begin{ {0,0,0} }, end(begin), end_old(begin);
        begin[0] = R_i[0], begin[1] = Z_i[0];
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 32;
        using Vec = std::array<double,3>;
        dg::SinglestepTimeloop<Vec> odeintT, odeintZ;
        if( mode_ == 0)
        {
            odeintT.construct( dg::RungeKutta<Vec>(
                        "Feagin-17-8-10", begin), fieldRZYTconf_);
            odeintZ.construct( dg::RungeKutta<Vec>(
                        "Feagin-17-8-10", begin), fieldRZYZconf_);
        }
        if( mode_ == 1)
        {
            odeintT.construct( dg::RungeKutta<Vec>(
                        "Feagin-17-8-10", begin), fieldRZYTequi_);
            odeintZ.construct( dg::RungeKutta<Vec>(
                        "Feagin-17-8-10", begin), fieldRZYZequi_);
        }
        while( (eps < eps_old || eps > 1e-7) && N < 1e6)
        {
            eps_old = eps, end_old = end;
            N*=2;
            odeintZ.integrate_steps( begin[1], begin,  0., end, N);
            std::array<double,3> temp(end);
            odeintT.integrate_steps( 0., temp, M_PI, end, N);
            temp = end;
            odeintZ.integrate_steps( temp[1], temp, Z_i[1], end, N);
            eps = sqrt( (end[0]-R_i[1])*(end[0]-R_i[1]) + (end[1]-Z_i[1])*(end[1]-Z_i[1]));
            //std::cout << "Found end[2] = "<< end_old[2]<<" with eps = "<<eps<<"\n";
            if( std::isnan(eps)) { eps = eps_old/2.; end = end_old; }
        }
        N_steps_=N;
        if(m_verbose)std::cout << "Found end[2] = "<< end_old[2]<<" with eps = "<<eps<<"\n";
        if(m_verbose)std::cout << "Found f = "<< 2.*M_PI/(y_i[0]+end_old[2]+y_i[1])<<" with eps = "<<eps<<"\n";
        f_psi_ = 2.*M_PI/(y_i[0]+end_old[2]+y_i[1]);
        return f_psi_;
    }
    int mode_;
    dg::geo::equalarc::FieldRZY  fieldRZYequi_;
    dg::geo::equalarc::FieldRZYT fieldRZYTequi_;
    dg::geo::equalarc::FieldRZYZ fieldRZYZequi_;
    dg::geo::ribeiro::FieldRZY   fieldRZYconf_;
    dg::geo::ribeiro::FieldRZYT  fieldRZYTconf_;
    dg::geo::ribeiro::FieldRZYZ  fieldRZYZconf_;
    unsigned N_steps_;
    double R_i[4], Z_i[4], y_i[4];
    double f_psi_;
    bool m_verbose;

};
} //namespace detail
namespace orthogonal
{
namespace detail
{
//find points on the perp line through the X-point
struct InitialX
{

    InitialX( const CylindricalFunctorsLvl1& psi, double xX, double yX, bool verbose = false):
        psip_(psi), fieldRZtau_(psi),
        xpointer_(psi, xX, yX, 1e-4), m_verbose( verbose)
    {
        //constructor finds four points around X-point and integrates them a bit away from it
        dg::geo::FieldRZtau fieldRZtau_(psi);
        using Vec = std::array<double,2>;
        dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>(
                    "Fehlberg-6-4-5", Vec{0.,0.}), fieldRZtau_);
        double eps[] = {1e-11, 1e-12, 1e-11, 1e-12};
        for( unsigned i=0; i<4; i++)
        {
            xpointer_.set_quadrant( i);
            double x_min = -1e-4, x_max = 1e-4;
            dg::bisection1d( xpointer_, x_min, x_max, eps[i]);
            xpointer_.point( R_i_[i], Z_i_[i], (x_min+x_max)/2.);
            //std::cout << "Found initial point: "<<R_i_[i]<<" "<<Z_i_[i]<<" "<<psip_(R_i_[i], Z_i_[i])<<"\n";
            std::array<double,2> begin, end, end_old;
            begin[0] = R_i_[i], begin[1] = Z_i_[i];
            double eps = 1e10, eps_old = 2e10;
            unsigned N=10;
            double psi0 = psip_.f()(begin[0], begin[1]), psi1 = 1e3*psi0;
            while( (eps < eps_old || eps > 1e-5 ) && eps > 1e-9)
            {
                eps_old = eps; end_old = end;
                N*=2;
                odeint.integrate_steps( psi0, begin, psi1, end, N); //lower order integrator is better for difficult field

                eps = sqrt( (end[0]-end_old[0])*(end[0]-end_old[0]) + (end[1]-end_old[1])*(end[1]-end_old[1]));
                if( std::isnan(eps)) { eps = eps_old/2.; end = end_old; }
                //std::cout << " for N "<< N<<" eps is "<<eps<<"\n";
            }
            R_i_[i] = end_old[0], Z_i_[i] = end_old[1];
            begin[0] = R_i_[i], begin[1] = Z_i_[i];
            eps = 1e10, eps_old = 2e10; N=10;
            psi0 = psip_.f()(begin[0], begin[1]), psi1 = -0.01;
            if( i==0||i==2)psi1*=-1.;
            while( (eps < eps_old || eps > 1e-5 ) && eps > 1e-9)
            {
                eps_old = eps; end_old = end;
                N*=2;
                odeint.integrate_steps( psi0, begin, psi1, end, N); //lower order integrator is better for difficult field

                eps = sqrt( (end[0]-end_old[0])*(end[0]-end_old[0]) + (end[1]-end_old[1])*(end[1]-end_old[1]));
                if( std::isnan(eps)) { eps = eps_old/2.; end = end_old; }
                //std::cout << " for N "<< N<<" eps is "<<eps<<"\n";
            }
            R_i_[i] = end_old[0], Z_i_[i] = end_old[1];
            if(m_verbose)std::cout << "Quadrant "<<i<<" Found initial point: "<<R_i_[i]<<" "<<Z_i_[i]<<" "<<psip_.f()(R_i_[i], Z_i_[i])<<"\n";

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
        std::array<double, 2> begin{ {0,0} }, end(begin), end_old(begin);
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
            using Vec = std::array<double,2>;
            dg::SinglestepTimeloop<Vec> odeint( dg::RungeKutta<Vec>(
                    "Fehlberg-6-4-5", Vec{0.,0.}), fieldRZtau_);
            while( (eps < eps_old||eps > 1e-7) && eps > 1e-11)
            {
                eps_old = eps; end_old = end;
                odeint.integrate_steps( psip_.f()(begin[0], begin[1]), begin,
                        psi, end, steps);
                eps = sqrt( (end[0]-end_old[0])*(end[0]- end_old[0]) +
                        (end[1]-end_old[1])*(end[1]-end_old[1]));
                //std::cout << "rel. error is "<<eps<<" with "<<steps<<" steps\n";
                if( std::isnan(eps)) { eps = eps_old/2.; end = end_old; }
                steps*=2;
            }
            //std::cout << "Found initial point "<<end_old[0]<<" "<<end_old[1]<<"\n";
            if( psi<0)
            {
                R_0[i] = R_i_[2*i+1] = begin[0] = end_old[0], Z_i_[2*i+1] =
                    Z_0[i] = begin[1] = end_old[1];
            }
            else
            {
                R_0[i] = R_i_[2*i] = begin[0] = end_old[0], Z_i_[2*i] = Z_0[i]
                    = begin[1] = end_old[1];
            }

        }
    }


    private:
    CylindricalFunctorsLvl1 psip_;
    const dg::geo::FieldRZtau fieldRZtau_;
    dg::geo::detail::XCross xpointer_;
    double R_i_[4], Z_i_[4];
    bool m_verbose;

};
}//namespace detail
}//namespace orthogonal
///@endcond
} //namespace geo
} //namespace dg

