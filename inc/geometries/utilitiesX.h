#pragma once
#include "dg/nullstelle.h"

namespace dg
{

namespace detail
{

/**
 * @brief This struct finds and stores the X-point and can act in a root finding routine to find points on the perpendicular line through the X-point 
 */
template<class Psi, class PsiR, class PsiZ>
struct XPointer
{
    
    XPointer( Psi psi, PsiR psiR, PsiZ psiZ, double R_X, double Z_X, double distance=1): fieldRZtau_(psiR, psiZ), psip_(psi), dist_(distance)
    {
        //solovev::HessianRZtau hessianRZtau(gp);
        //R_X_ = gp.R_0-1.1*gp.triangularity*gp.a;
        //Z_X_ = -1.1*gp.elongation*gp.a;
        //thrust::host_vector<double> X(2,0), XN(X);
        //X[0] = R_X_, X[1] = Z_X_;
        //for( unsigned i=0; i<2; i++)
        //{
        //    hessianRZtau.newton_iteration( X, XN);
        //    XN.swap(X);
        //}
        //std::cout << "X-point error  "<<R_X_-X[0]<<" "<<Z_X_-X[1]<<"\n";
        //R_X_ = X[0], Z_X_ = X[1];
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
        thrust::host_vector<double> begin(2), end(2), end_old(2);
        begin[0] = R_i[quad_], begin[1] = Z_i[quad_];
        double eps = 1e10, eps_old = 2e10;
        unsigned N=10;
        if( quad_ == 0 || quad_ == 2) { begin[1] += x;}
        else if( quad_ == 1 || quad_ == 3) { begin[0] += x;}

        double psi0 = psip_(begin[0], begin[1]);
        while( (eps < eps_old || eps > 1e-4 ) && eps > 1e-10)
        {
            eps_old = eps; end_old = end;
            N*=2; 
            dg::stepperRK17( fieldRZtau_, begin, end, psi0, 0, N);

            eps = sqrt( (end[0]-end_old[0])*(end[0]-end_old[0]) + (end[1]-end_old[1])*(end[1]-end_old[1]));
            if( isnan(eps)) { eps = eps_old/2.; end = end_old; }
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
    solovev::FieldRZtau<PsiR, PsiZ> fieldRZtau_;
    Psi psip_;
    double R_X_, Z_X_;
    double R_i[4], Z_i[4];
    double dist_;
};

//compute psi(x) and f(x) for given discretization of x and a fpsiMinv functor
//doesn't integrate over the x-point
//returns psi_1
template <class XFieldFinv>
double construct_psi_values( XFieldFinv fpsiMinv, 
        const double psi_0, const double x_0, const thrust::host_vector<double>& x_vec, const double x_1, unsigned idxX, //idxX is the index of the Xpoint
        thrust::host_vector<double>& psi_x, 
        thrust::host_vector<double>& f_x_)
{
    f_x_.resize( x_vec.size()), psi_x.resize( x_vec.size());
    thrust::host_vector<double> psi_old(psi_x), psi_diff( psi_old);
    unsigned N = 1;
    //std::cout << "In psi function:\n";
    double x0, x1;
    const double psi_const = fpsiMinv.find_psi( x_vec[idxX]);
    double psi_1_numerical=0;
    double eps = 1e10, eps_old=2e10;
    while( (eps <  eps_old || eps > 1e-8) && eps > 1e-11) //1e-8 < eps < 1e-14
    {
        eps_old = eps; 
        psi_old = psi_x; 
        x0 = x_0, x1 = x_vec[0];

        thrust::host_vector<double> begin(1,psi_0), end(begin), temp(begin);
        dg::stepperRK17( fpsiMinv, begin, end, x0, x1, N);
        psi_x[0] = end[0]; fpsiMinv(end,temp); f_x_[0] = temp[0];
        for( unsigned i=1; i<idxX; i++)
        {
            temp = end;
            x0 = x_vec[i-1], x1 = x_vec[i];
            dg::stepperRK17( fpsiMinv, temp, end, x0, x1, N);
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
            dg::stepperRK17( fpsiMinv, temp, end, x0, x1, N);
            psi_x[i] = end[0]; fpsiMinv(end,temp); f_x_[i] = temp[0];
            //std::cout << "FOUND PSI "<<end[0]<<"\n";
        }
        temp = end;
        dg::stepperRK17(fpsiMinv, temp, end, x1, x_1,N);
        psi_1_numerical = end[0];
        dg::blas1::axpby( 1., psi_x, -1., psi_old, psi_diff);
        //eps = sqrt( dg::blas2::dot( psi_diff, w1d, psi_diff)/ dg::blas2::dot( psi_x, w1d, psi_x));
        eps = sqrt( dg::blas1::dot( psi_diff, psi_diff)/ dg::blas1::dot( psi_x, psi_x));
        //psi_1_numerical_ = psi_0 + dg::blas1::dot( f_x_, w1d);

        //std::cout << "Effective Psi error is "<<eps<<" with "<<N<<" steps\n"; 
        //std::cout << "psi 1               is "<<psi_1_numerical_<<"\n"; 
        N*=2;
    }
    return psi_1_numerical;
}



//!ATTENTION: choosing h on separatrix is a mistake if LaplacePsi does not vanish at X-point
template< class Psi>
struct PsipSep
{
    PsipSep( Psi psi): psip_(psi), Z_(0){}
    void set_Z( double z){ Z_=z;}
    double operator()(double R) { return psip_(R, Z_);}
    private:
    double Z_;
    Psi psip_;
};

//!ATTENTION: choosing h on separatrix is a mistake if LaplacePsi does not vanish at X-point
//This leightweights struct and its methods finds the initial R and Z values and the coresponding f(\psi) as 
//good as it can, i.e. until machine precision is reached (like FpsiX just for separatrix)
template< class Psi, class PsiX, class PsiY>
struct SeparatriX
{
    SeparatriX( Psi psi, PsiX psiX, PsiY psiY, double xX, double yX, double x0, double y0, int firstline): 
        mode_(firstline),
        fieldRZYequi_(psiX, psiY), fieldRZYTequi_(psiX, psiY, x0, y0), fieldRZYZequi_(psiX, psiY),
        fieldRZYconf_(psiX, psiY), fieldRZYTconf_(psiX, psiY, x0, y0), fieldRZYZconf_(psiX, psiY)
    {
        //find four points on the separatrix and construct y coordinate at those points and at last construct f 
        //////////////////////////////////////////////
        double R_X = xX; double Z_X = yX;
        PsipSep<Psi> psip_sep( psi);
        psip_sep.set_Z( Z_X + 1.);
        double R_min = R_X, R_max = R_X + 10;
        dg::bisection1d( psip_sep, R_min, R_max, 1e-13);
        R_i[0] = (R_min+R_max)/2., Z_i[0] = Z_X+1.;
        R_min = R_X-10, R_max = R_X;
        dg::bisection1d( psip_sep, R_min, R_max, 1e-13);
        R_i[1] = (R_min+R_max)/2., Z_i[1] = Z_X+1.;
        psip_sep.set_Z( Z_X - 1.);
        R_min = R_X-10, R_max = R_X;
        dg::bisection1d( psip_sep, R_min, R_max, 1e-13);
        R_i[2] = (R_min+R_max)/2., Z_i[2] = Z_X-1.;
        R_min = R_X, R_max = R_X+10;
        dg::bisection1d( psip_sep, R_min, R_max, 1e-13);
        R_i[3] = (R_min+R_max)/2., Z_i[3] = Z_X-1.;
        //std::cout << "Found 3rd point "<<R_i[3]<<" "<<Z_i[3]<<"\n";
        //now measure y distance to X-point
        thrust::host_vector<double> begin2d( 3, 0), end2d( begin2d);
        for( int i=0; i<4; i++)
        {
            unsigned N = 1;
            begin2d[0] = end2d[0] = R_i[i];
            begin2d[1] = end2d[1] = Z_i[i];
            begin2d[2] = end2d[2] = 0.;
            double eps = 1e10, eps_old = 2e10;
            double y=0, y_old=0;
            //difference to X-point isn't much better than 1e-5
            while( (eps < eps_old || eps > 5e-5))
            {
                eps_old = eps; N*=2; y_old=y;
                if(mode_==0)dg::stepperRK6( fieldRZYZconf_, begin2d, end2d, Z_i[i], Z_X, N);
                if(mode_==1)dg::stepperRK6( fieldRZYZequi_, begin2d, end2d, Z_i[i], Z_X, N);
                y=end2d[2];
                eps = fabs((y-y_old)/y_old);
                eps = sqrt( (end2d[0]-R_X)*(end2d[0]-R_X))/R_X;
                //std::cout << "Found y_i["<<i<<"]: "<<y<<" with eps = "<<eps<<" and "<<N<<" steps and diff "<<fabs(end2d[0]-R_X)/R_X<<"\n";
            }
            //remember last call
            y_i[i] = end2d[2]; 
            std::cout << "Found y_i["<<i<<"]: "<<y<<" with eps = "<<eps<<" and "<<N<<" steps and diff "<<fabs(end2d[0]-R_X)/R_X<<"\n";
        }
        y_i[0]*=-1; y_i[2]*=-1; //these were integrated against y direction

        f_psi_ = construct_f( );
        y_i[0]*=f_psi_, y_i[1]*=f_psi_, y_i[2]*=f_psi_, y_i[3]*=f_psi_;
        fieldRZYequi_.set_f(f_psi_);
        fieldRZYconf_.set_f(f_psi_);
    }

    double get_f( ) const{return f_psi_;}

    //compute the vector of r and z - values that form the separatrix
    void compute_rzy( const thrust::host_vector<double>& y_vec, 
            const unsigned nodeX0, const unsigned nodeX1,
            thrust::host_vector<double>& r, 
            thrust::host_vector<double>& z ) 
    {
        ///////////////////////////find y coordinate line//////////////
        thrust::host_vector<double> begin( 2, 0), end(begin), temp(begin), end_old(end);
        thrust::host_vector<double> r_old(y_vec.size(), 0), r_diff( r_old);
        thrust::host_vector<double> z_old(y_vec.size(), 0), z_diff( z_old);
        r.resize( y_vec.size()), z.resize(y_vec.size());
        unsigned steps = 1; double eps = 1e10, eps_old=2e10;
        while( (eps < eps_old||eps > 1e-7) && eps > 1e-11)
        {
            eps_old = eps, r_old = r, z_old = z;
            //////////////////////bottom right region/////////////////////
            if( nodeX0 != 0) //integrate to start point
            {
                begin[0] = R_i[3], begin[1] = Z_i[3];
                if(mode_==0)dg::stepperRK17( fieldRZYconf_, begin, end, -y_i[3], y_vec[nodeX0-1], N_steps_);
                if(mode_==1)dg::stepperRK17( fieldRZYequi_, begin, end, -y_i[3], y_vec[nodeX0-1], N_steps_);
                r[nodeX0-1] = end[0], z[nodeX0-1] = end[1];
            }
            for( int i=nodeX0-2; i>=0; i--)
            {
                temp = end;
                if(mode_==0)dg::stepperRK17( fieldRZYconf_, temp, end, y_vec[i+1], y_vec[i], steps);
                if(mode_==1)dg::stepperRK17( fieldRZYequi_, temp, end, y_vec[i+1], y_vec[i], steps);
                r[i] = end[0], z[i] = end[1];
            }
            ////////////////middle region///////////////////////////
            begin[0] = R_i[0], begin[1] = Z_i[0];
            if(mode_==0)dg::stepperRK17( fieldRZYconf_, begin, end, y_i[0], y_vec[nodeX0], N_steps_);
            if(mode_==1)dg::stepperRK17( fieldRZYequi_, begin, end, y_i[0], y_vec[nodeX0], N_steps_);
            r[nodeX0] = end[0], z[nodeX0] = end[1];
            for( unsigned i=nodeX0+1; i<nodeX1; i++)
            {
                temp = end;
                if(mode_==0)dg::stepperRK17( fieldRZYconf_, temp, end, y_vec[i-1], y_vec[i], steps);
                if(mode_==1)dg::stepperRK17( fieldRZYequi_, temp, end, y_vec[i-1], y_vec[i], steps);
                r[i] = end[0], z[i] = end[1];
            }
            temp = end;
            if(mode_==0)dg::stepperRK17( fieldRZYconf_, temp, end, y_vec[nodeX1-1], 2.*M_PI-y_i[1], N_steps_);
            if(mode_==1)dg::stepperRK17( fieldRZYequi_, temp, end, y_vec[nodeX1-1], 2.*M_PI-y_i[1], N_steps_);
            eps = sqrt( (end[0]-R_i[1])*(end[0]-R_i[1]) + (end[1]-Z_i[1])*(end[1]-Z_i[1]));
            //std::cout << "abs. error is "<<eps<<" with "<<steps<<" steps\n";
            ////////////////////bottom left region

            if( nodeX0!= 0)
            {
                begin[0] = R_i[2], begin[1] = Z_i[2];
                if(mode_==0)dg::stepperRK17( fieldRZYconf_, begin, end, 2.*M_PI+y_i[2], y_vec[nodeX1], N_steps_);
                if(mode_==1)dg::stepperRK17( fieldRZYequi_, begin, end, 2.*M_PI+y_i[2], y_vec[nodeX1], N_steps_);
                r[nodeX1] = end[0], z[nodeX1] = end[1];
            }
            for( unsigned i=nodeX1+1; i<y_vec.size(); i++)
            {
                temp = end;
                if(mode_==0)dg::stepperRK17( fieldRZYconf_, temp, end, y_vec[i-1], y_vec[i], steps);
                if(mode_==1)dg::stepperRK17( fieldRZYequi_, temp, end, y_vec[i-1], y_vec[i], steps);
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
            std::cout << "rel. Separatrix error is "<<eps<<" with "<<steps<<" steps\n";
            steps*=2;
        }
        r = r_old, z = z_old;
    }
    private:
    //compute f for psi=0
    double construct_f( ) 
    {
        std::cout << "In construct f function!\n";
        
        thrust::host_vector<double> begin( 3, 0), end(begin), end_old(begin);
        begin[0] = R_i[0], begin[1] = Z_i[0];
        double eps = 1e10, eps_old = 2e10;
        unsigned N = 32; 
        while( (eps < eps_old || eps > 1e-7) && N < 1e6)
        {
            eps_old = eps, end_old = end; 
            N*=2; 
            if(mode_==0)
            {
                dg::stepperRK17( fieldRZYZconf_, begin, end, begin[1], 0., N);
                thrust::host_vector<double> temp(end);
                dg::stepperRK17( fieldRZYTconf_, temp, end, 0., M_PI, N);
                temp = end; 
                dg::stepperRK17( fieldRZYZconf_, temp, end, temp[1], Z_i[1], N);
            }
            if(mode_==1)
            {
                dg::stepperRK17( fieldRZYZequi_, begin, end, begin[1], 0., N);
                thrust::host_vector<double> temp(end);
                dg::stepperRK17( fieldRZYTequi_, temp, end, 0., M_PI, N);
                temp = end; 
                dg::stepperRK17( fieldRZYZequi_, temp, end, temp[1], Z_i[1], N);
            }
            eps = sqrt( (end[0]-R_i[1])*(end[0]-R_i[1]) + (end[1]-Z_i[1])*(end[1]-Z_i[1]));
            //std::cout << "Found end[2] = "<< end_old[2]<<" with eps = "<<eps<<"\n";
            if( isnan(eps)) { eps = eps_old/2.; end = end_old; }
        }
        N_steps_=N;
        std::cout << "Found end[2] = "<< end_old[2]<<" with eps = "<<eps<<"\n";
        std::cout << "Found f = "<< 2.*M_PI/(y_i[0]+end_old[2]+y_i[1])<<" with eps = "<<eps<<"\n";
        f_psi_ = 2.*M_PI/(y_i[0]+end_old[2]+y_i[1]);
        return f_psi_;
    }
    int mode_;
    solovev::equalarc::FieldRZY<PsiX, PsiY>   fieldRZYequi_;
    solovev::equalarc::FieldRZYT<PsiX, PsiY> fieldRZYTequi_;
    solovev::equalarc::FieldRZYZ<PsiX, PsiY> fieldRZYZequi_;
    solovev::ribeiro::FieldRZY<PsiX, PsiY>    fieldRZYconf_;
    solovev::ribeiro::FieldRZYT<PsiX, PsiY>  fieldRZYTconf_;
    solovev::ribeiro::FieldRZYZ<PsiX, PsiY>  fieldRZYZconf_;
    unsigned N_steps_;
    double R_i[4], Z_i[4], y_i[4];
    double f_psi_;

};
} //namespace detail
} //namespace dg

