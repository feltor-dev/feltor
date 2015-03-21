#pragma once

#include <complex>
#include <cassert>

#include "toefl/toefl.h"
#include "blueprint.h"
#include "equations.h"

namespace toefl{

    ///@addtogroup solvers
    ///@{
    /**
     * @brief Blas1 function
     *
     * computes \f[ y = \alpha x + \beta y \f]
     * @param alpha a scalar
     * @param x Matrix 
     * @param beta a scalar
     * @param y Matrix (read & write) containing result on output
     */
void axpby(double alpha,  const Matrix<double, TL_DFT>& x, double beta, Matrix<double, TL_DFT>& y)
{
    for( unsigned i=0; i<x.rows(); i++)
        for( unsigned j=0; j<x.cols(); j++)
            y(i,j) = alpha*x(i,j)+beta*y(i,j);
}

/**
 * @brief Blas1 dot product
 *
 * Computes \f[\sum_{ij} m_{1ij}m_{2ij}\f]
 * @param m1 
 * @param m2
 *
 * @return Dot product
 */
double dot( const Matrix<double, TL_DFT>& m1, const Matrix<double, TL_DFT>& m2)
{
    double sum = 0;
#pragma omp parallel for reduction(+: sum)
    for( unsigned i=0; i<m1.rows(); i++)
        for( unsigned j=0; j<m1.cols(); j++)
            sum+= m1(i,j)*m2(i,j);
    return sum;

}

/**
 * @brief Compute dot product
 *
 * Computes \f[\sum_{i} v_{1i}v^*_{2i}\f]
 * where * is the complex conjugate 
 *
 * @param v1
 * @param v2
 *
 * @return 
 */
double dot( const std::vector<std::complex<double> >& v1, const std::vector<std::complex<double> >& v2)
{
    assert( v1.size() == v2.size());
    unsigned cols = v1.size();
    std::complex<double> sum=0;
    sum += v1[0]*conj( v2[0]);
    for( unsigned j=1; j<cols/2; j++)
        sum += 2.*v1[j]*conj( v2[j]);
    if( cols%2)
        sum += v1[cols/2]*conj(v2[cols/2]);
    else
        sum += 2.*v1[cols/2]*conj( v2[cols/2]);
    //if( imag(sum) > 1e-12) std::cerr<<sum << "WARNING: Imag not zero!\n";
    return real( sum);
}

/**
 * @brief Compute dot product
 *
 * Computes \f[\sum_{i} v_{1i}v^_{2i}\f]
 *
 * @param v1 real vector
 * @param v2 real vector
 *
 * @return  dot product
 */
double dot( const std::vector<double>& v1, const std::vector<double>& v2)
{
    assert( v1.size() == v2.size());
    double sum=0; 
    for( unsigned i=0; i<v1.size(); i++)
        sum += v1[i]*v2[i];
    return sum;
}

std::vector<std::complex<double> > extract_sum_y( const Matrix<std::complex<double> >& in)
{
    std::vector<std::complex<double> > out( in.cols());
    for( unsigned j=0; j<in.cols(); j++)
        out[j] = in(0,j);
    return out;
}

std::vector<double> extract_sum_y( const Matrix<double, TL_DFT>& in)
{
    std::vector<double> out( in.cols(), 0);
    for( unsigned i=0; i<in.rows(); i++)
        for( unsigned j=0; j<in.cols(); j++)
            out[j] += in(i,j);
    return out;
}

//remove the real average in y-direction
void remove_average_y( const Matrix<double, TL_DFT>& in, Matrix<double, TL_DFT>& m)
{
    m = in;
    std::vector<double> average = extract_sum_y( in);
#pragma omp parallel for
    for( unsigned i=0; i<in.rows(); i++)
        for( unsigned j=0; j<in.cols(); j++)
            m(i,j) -= average[j]/in.rows();
}

/**
 * @brief Compute y derivative using spectral method
 *
 * @param in
 * @param out
 * @param h grid constant
 */
void dy( const Matrix<double, TL_DFT>& in, Matrix<double, TL_DFT>& out, double h)
{
    assert( &in != &out);
    unsigned rows = in.rows(); 
    for( unsigned j=0; j<in.cols(); j++)
        out(0,j) = (in(1,j) - in(rows-1, j))/2./h;
#pragma omp parallel for
    for( unsigned i=1; i<in.rows()-1; i++)
    {
        for( unsigned j=0; j<in.cols(); j++)
            out(i,j) = (in(i+1,j) - in(i-1, j))/2./h;
    }
    for( unsigned j=0; j<in.cols(); j++)
        out(rows-1,j) = (in(0,j) - in(rows-2, j))/2./h;
}

/**
 * @brief Compute Energetics of INNTO code
 *
 * @tparam n the number of equations
 */
template<size_t n>
struct Energetics
{
    typedef Matrix<double, TL_DFT> Matrix_Type;
    typedef std::complex<double> complex;
    Energetics( const Blueprint& bp):
        rows( bp.algorithmic().ny ), cols( bp.algorithmic().nx ),
        crows( rows), ccols( cols/2+1),
        diff_coeff( rows, cols),
        a_mu_gamma0_coeff( MatrixArray< double, TL_NONE, n-1>::construct( crows, ccols)),
        dens_( MatrixArray<double, TL_DFT,n>::construct( rows, cols)),
        phi_( dens_),
        cdens_( MatrixArray<complex, TL_NONE, n>::construct( crows, ccols)), 
        cphi_(cdens_), 
        blue(bp), phys( bp.physical()), bound( bp.boundary()), alg( bp.algorithmic()),
        dft_dft( rows, cols, FFTW_MEASURE),
        a_{-1. , phys.a[0], phys.a[1]}, tau_{-1. , phys.tau[0], phys.tau[1]},
        g_{phys.g_e, phys.g[0], phys.g[1]}
    {
        double laplace;
        Poisson p( phys);
        int ik;
        const double kxmin2 = 2.*2.*M_PI*M_PI/(double)(bound.lx*bound.lx),
                     kymin2 = 2.*2.*M_PI*M_PI/(double)(bound.ly*bound.ly);
        for( unsigned i = 0; i<crows; i++)
            for( unsigned j = 0; j<ccols; j++)
            {
                ik = (i>rows/2) ? (i-rows) : i; //integer division rounded down
                laplace = - kxmin2*(double)(j*j) - kymin2*(double)(ik*ik);
                diff_coeff(i,j) = -phys.nu*pow(-laplace,2);
                if( n==2)
                {
                    a_mu_gamma0_coeff[0](i,j) = -phys.a[0]*phys.mu[0]*laplace*p.gamma0_i( laplace);
                }
                else if( n==3)
                {
                    a_mu_gamma0_coeff[0](i,j) = -phys.a[0]*phys.mu[0]*laplace*p.gamma0_i( laplace);
                    a_mu_gamma0_coeff[1](i,j) = -phys.a[1]*phys.mu[1]*laplace*p.gamma0_z( laplace);
                }
            }
    }
    std::vector<double> thermal_energies(const std::array<Matrix<double, TL_DFT>, n>& dens_ );
    std::vector<double> exb_energies(const Matrix<double, TL_DFT>& phi_ );
    std::vector<double> diffusion( const std::array<Matrix<double, TL_DFT>, n>& density , const std::array<Matrix<double, TL_DFT>, n>& potential);
    std::vector<double> gradient_flux( const std::array<Matrix<double, TL_DFT>, n>& density , const std::array<Matrix<double, TL_DFT>, n>& potential);
    double capital_jot( const Matrix<double, TL_DFT>& density , const Matrix<double, TL_DFT>& potential);
    double capital_a( const Matrix<double, TL_DFT>& density , const Matrix<double, TL_DFT>& potential);
    
  private:
    unsigned rows, cols;
    unsigned crows, ccols;
    Matrix<double, TL_DFT> diff_coeff;
    std::array< Matrix< double>, n-1> a_mu_gamma0_coeff;
    std::array< Matrix<double, TL_DFT>, n > dens_, phi_;
    std::array< Matrix< complex>, n> cdens_, cphi_;
    Blueprint blue;
    Physical phys;
    Boundary bound;
    Algorithmic alg;
    DFT_DFT dft_dft;
    double a_[3]; 
    double tau_[3]; 
    double g_[3]; 
};
///////////////////////////////////////////////////////
///@cond
template<size_t n>
std::vector<double> Energetics<n>::thermal_energies(const std::array<Matrix<double, TL_DFT>, n>& density )
{
    std::vector<double> energies(n);
    for( unsigned i=0; i<n; i++)
        energies[i] = 0.5*a_[i]*tau_[i]*dot( density[i], density[i])*alg.h*alg.h;
    return energies;
}

template<size_t n>
std::vector<double> Energetics<n>::exb_energies(const Matrix<double, TL_DFT>& potential )
{
    std::vector<double> energies;
    double norm = alg.h*alg.h/(double)(rows*cols); //h^2 from integration and 1/NxNy from complex scalar product 
    phi_[0] = potential;
    dft_dft.r2c( phi_[0], cphi_[0]);
    for( size_t k=1; k<n; k++)
    {
#pragma omp parallel for 
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
                cphi_[k](i,j) = (a_mu_gamma0_coeff[k-1](i,j))*cphi_[0](i,j);
        energies.push_back( 0.5*dft_dft.dot( cphi_[0], cphi_[k])*norm);
    }
    //averages
    double norm_avg = alg.h*alg.h/(double)(rows)/(double)cols; //h*h from integration 1/Ny from average and 1/Nx from complex scalar product
    std::vector<complex> sum_phi[n];
    for( unsigned i=0; i<n; i++)
        sum_phi[i] = extract_sum_y( cphi_[i]); 
    for( size_t k=1; k<n; k++)
        energies.push_back( 0.5*dot( sum_phi[0], sum_phi[k])*norm_avg);

    return energies;
}

template<size_t n>
double Energetics<n>::capital_jot( const Matrix<double, TL_DFT>& density , const Matrix<double, TL_DFT>& potential)
{
#pragma omp parallel for
    for( unsigned i=0; i<rows; i++)
        for( unsigned j=0; j<cols; j++)
            dens_[0](i,j) = potential(i,j) - density(i,j);
    if( blue.isEnabled( TL_MHW) )
        remove_average_y( dens_[0], dens_[1]);
    return -phys.d*dot( dens_[0], dens_[1])*alg.h*alg.h;
}
template<size_t n>
double Energetics<n>::capital_a( const Matrix<double, TL_DFT>& density , const Matrix<double, TL_DFT>& potential)
{
    std::vector<double> sum_dens  = extract_sum_y( density);
    std::vector<double> sum_phi = extract_sum_y( potential);
    double a = dot( sum_phi, sum_phi) - dot(sum_dens, sum_phi);
    return -phys.d*a/(double)(rows)*alg.h*alg.h;
}

/////////////////////////////////////////////////////////////////////
template<size_t n>
std::vector<double> Energetics<n>::gradient_flux( const std::array<Matrix<double, TL_DFT>, n>& density , const std::array<Matrix<double, TL_DFT>, n>& potential)
{
    std::vector<double> flux;
    for( unsigned i=0; i<n; i++)
    {
        dy( potential[i], dens_[i], alg.h); //dens_ = dy phi_
        flux.push_back( g_[i]*a_[i]*tau_[i]*dot( density[i], dens_[i])*alg.h*alg.h);
    }

//zonal flow  R
    double r=0;
    for( unsigned k=0; k<n; k++)
    {
#pragma omp parallel for
        for( unsigned i=0; i<rows;i++)
            for( unsigned j=0; j<cols;j++)
                dens_[k](i,j) *= density[k](i,j); //dy phi *density
        std::vector<double> sum = extract_sum_y( dens_[k]);
        //compute dx sum
        std::vector<double> vy(sum);
        vy[0] = (sum[1]-sum[cols-1])/2./alg.h;
        for( unsigned i=1; i<cols-1; i++)
            vy[i] = (sum[i+1]-sum[i-1])/2./alg.h;
        vy[cols-1] = (sum[0]-sum[cols-2])/2./alg.h;
        std::vector<double> sum_phi = extract_sum_y( potential[k]);
        r+= a_[k]*dot( vy, sum_phi)*alg.h*alg.h/(double)(rows);
    }
    flux.push_back(r);
    flux.push_back(0);

/*
    phi_[0] = potential[0];
    dft_dft.r2c( phi_[0], cphi_[0]);
    for( size_t k=1; k<n; k++)
    {
#pragma omp parallel for
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
                cphi_[k](i,j) = (-a_mu_gamma0_coeff[k-1](i,j))*cphi_[0](i,j)/(double)(rows*cols);
        dft_dft.c2r( cphi_[k], phi_[k]); 
#pragma omp parallel for
        for( unsigned i=0; i<rows;i++)
            for( unsigned j=0; j<cols;j++)
                dens_[k](i,j) = dens_[0](i,j)*phi_[k](i,j); //dy phi * a/tau*(1-\Gamma_0)phi

        std::vector<double> sum = extract_sum_y( dens_[k]);
        std::vector<double> sum_phi = extract_sum_y( potential[k]);
        //compute dx sum
        std::vector<double> vy(sum);
        vy[0] = (sum[1]-sum[cols-1])/2./alg.h;
        for( unsigned i=1; i<cols-1; i++)
            vy[i] = (sum[i+1]-sum[i-1])/2./alg.h;
        vy[cols-1] = (sum[0]-sum[cols-2])/2./alg.h;
        flux.push_back( dot( vy, sum_phi)*alg.h*alg.h/(double)(rows));
    }
    */
    return flux;
}

template<size_t n>
std::vector<double> Energetics<n>::diffusion( const std::array<Matrix<double, TL_DFT>, n>& density , const std::array<Matrix<double, TL_DFT>, n>& potential)
{
    //compute -nu*laplace^2 density in dens_
    dens_ = density;
#pragma omp parallel for
    for( unsigned k=0; k<n; k++)
    {
        dft_dft.r2c( dens_[k], cdens_[k]);
        for( size_t i = 0; i < crows; i++)
            for( size_t j = 0; j < ccols; j++)
                cdens_[k](i,j) = diff_coeff(i,j)*cdens_[k](i,j)/(double)(rows*cols);
        dft_dft.c2r( cdens_[k], dens_[k]);
    }
    //add up diffusion
    std::vector<double> diffusion_(2, 0.);
    for( unsigned i=0; i<n; i++)
    {
        diffusion_[0] += a_[i]*dot( potential[i], dens_[i])*alg.h*alg.h;
        diffusion_[0] += a_[i]*tau_[i]*dot( density[i], dens_[i])*alg.h*alg.h;
    }
    //compute mean diffusion
    for(unsigned k=0; k<n; k++)
    {
        std::vector<double> sum_dens = extract_sum_y( dens_[k] );
        std::vector<double> sum_phi  = extract_sum_y( potential[k] );
        diffusion_[1] += a_[k]*dot( sum_dens, sum_phi)*alg.h*alg.h/(double)(rows);
    }
    return diffusion_;
}
///@endcond
///@}
} //namespace toefl
