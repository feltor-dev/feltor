#pragma once

#include <thrust/host_vector.h>
#include "dg/topology/weights.h"
#include "magnetic_field.h"

/*!@file
 *
 * The flux surface average
 */
namespace dg
{
namespace geo
{
/**
 * @brief Delta function for poloidal flux \f$ B_Z\f$
     \f[ \delta(\psi_p(R,Z)-\psi_0) = \frac{ 1}{\sqrt{2\pi\varepsilon}} \exp\left(-\frac{(\psi_p(R,Z) - \psi_{0})^2}{2\varepsilon} \right)  \f]
     @ingroup profiles
 */
struct DeltaFunction
{
    DeltaFunction(const TokamakMagneticField& c, double epsilon,double psivalue) :
        c_(c), epsilon_(epsilon), psivalue_(psivalue){ }
    /**
    * @brief Set a new \f$ \varepsilon\f$
    *
    * @param eps new value
    */
    void setepsilon(double eps ){epsilon_ = eps;}
    /**
    * @brief Set a new \f$ \psi_0\f$
    *
    * @param psi_0 new value
    */
    void setpsi(double psi_0 ){psivalue_ = psi_0;}

    /**
     *@brief \f[ \frac{1}{\sqrt{2\pi\varepsilon}} \exp\left(-\frac{(\psi_p(R,Z) - \psi_{0})^2}{2\varepsilon} \right)  \f]
     */
    double operator()( double R, double Z) const
    {
        double psip = c_.psip()(R,Z);
        return 1./sqrt(2.*M_PI*epsilon_)*
               exp(-( (psip-psivalue_)* (psip-psivalue_))/2./epsilon_);
    }
    /**
     * @brief == operator()(R,Z)
     */
    double operator()( double R, double Z, double phi) const
    {
        return (*this)(R,Z);
    }
    private:
    TokamakMagneticField c_;
    double epsilon_;
    double psivalue_;
};

/**
 * @brief Global safety factor
\f[ \alpha(R,Z) = \frac{I_{pol}(R,Z)}{R|\nabla\psi_p|} \frac{1}{\sqrt{2\pi\varepsilon}} \exp\left(-\frac{(\psi_p(R,Z) - \psi_{0})^2}{2\varepsilon} \right)  \f]
     @ingroup profiles
 */
struct Alpha
{
    Alpha( const TokamakMagneticField& c):c_(c){}
    Alpha(const TokamakMagneticField& c, double epsilon,double psivalue) :
        c_(c), m_eps(epsilon), m_psi(psivalue){ }
    /**
    * @brief Set a new \f$ \varepsilon\f$
    *
    * @param eps new value
    */
    void setepsilon(double eps ){m_eps = eps;}
    /**
    * @brief Set a new \f$ \psi_0\f$
    *
    * @param psi_0 new value
    */
    void setpsi(double psi_0 ){m_psi = psi_0;}

    /**
    * @brief \f[ \frac{ I_{pol}(R,Z)}{R \sqrt{\nabla\psi_p}} \f]
    */
    double operator()( double R, double Z) const
    {
        double psip = c_.psip()(R,Z);
        return c_.ipol()(R,Z)/R /sqrt(2.*M_PI*m_eps)*
               exp(-( (psip-m_psi)* (psip-m_psi))/2./m_eps);
    }
    /**
     * @brief == operator()(R,Z)
     */
    double operator()( double R, double Z, double phi) const
    {
        return operator()(R,Z);
    }
    private:
    TokamakMagneticField c_;
    double m_eps, m_psi;
};

/**
 * @brief Flux surface average over quantity
 \f[ \langle f\rangle(\psi_0) = \frac{1}{A} \int dV \delta(\psi_p(R,Z)-\psi_0) |\nabla\psi_p|f(R,Z)H(R,Z) \f]

 with \f$ A = \int dV \delta(\psi_p(R,Z)-\psi_0)|\nabla\psi_p|H(R,Z)\f$
 where \c H is a weight function that can be used to e.g. cut away parts of the domain below the X-point
 * @ingroup misc_geo
 */
template <class container = thrust::host_vector<double> >
struct FluxSurfaceAverage
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param c contains psip, psipR and psipZ
     * @param f container for global safety factor
     * @param weights Weight function \c H (can be used to cut away parts of the domain e.g. below the X-point)
     * @param multiplyByGradPsi if true multiply f with GradPsi, else not
     */
    FluxSurfaceAverage(const dg::Grid2d& g2d, const TokamakMagneticField& c, const container& f, const container& weights, bool multiplyByGradPsi = true) :
    m_f(f), m_deltafog2d(f),
    m_deltaf(c, 0.,0.),
    m_w2d ( dg::create::weights( g2d)),
    m_x ( dg::evaluate( dg::cooX2d, g2d)),
    m_y ( dg::evaluate( dg::cooY2d, g2d)),
    m_weights(weights)
    {
        thrust::host_vector<double> psipRog2d  = dg::evaluate( c.psipR(), g2d);
        thrust::host_vector<double> psipZog2d  = dg::evaluate( c.psipZ(), g2d);
        double psipRmax = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  0.,     thrust::maximum<double>()  );
        //double psipRmin = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  psipRmax,thrust::minimum<double>()  );
        double psipZmax = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), 0.,      thrust::maximum<double>()  );
        //double psipZmin = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), psipZmax,thrust::minimum<double>()  );
        double deltapsi = fabs(psipZmax/g2d.Ny()/g2d.n() +psipRmax/g2d.Nx()/g2d.n());
        m_deltaf.setepsilon(deltapsi/4);
        //m_deltaf.setepsilon(deltapsi); //macht weniger Zacken
        dg::blas1::pointwiseDot( 1., psipRog2d, psipRog2d, 1., psipZog2d, psipZog2d, 0., psipRog2d);
        dg::blas1::transform( psipRog2d, psipRog2d, dg::SQRT<double>());
        dg::assign( psipRog2d, m_gradpsi);
        if(multiplyByGradPsi)
            dg::blas1::pointwiseDot( m_f, m_gradpsi, m_f);
    }

    /**
     * @brief Set the function to average
     *
     * @param f the container containing the discretized function
     * @param multiplyByGradPsi if true multiply with GradPsi, else not
     */
    void set_container( const container& f, bool multiplyByGradPsi=true){
        dg::blas1::copy( f, m_f);
        if(multiplyByGradPsi)
            dg::blas1::pointwiseDot( m_f, m_gradpsi, m_f);
    }
    /**
     * @brief Calculate the Flux Surface Average
     *
     * @param psip0 the actual psi value for q(psi)
     * @return q(psip0)
     */
    double operator()(double psip0)
    {
        m_deltaf.setpsi( psip0);
        dg::blas1::evaluate( m_deltafog2d, dg::equals(), m_deltaf, m_x, m_y);
        dg::blas1::pointwiseDot( m_deltafog2d, m_weights, m_deltafog2d);
        double psipcut = dg::blas2::dot( m_f, m_w2d, m_deltafog2d); //int deltaf psip
        double vol     = dg::blas2::dot( m_gradpsi, m_w2d, m_deltafog2d); //int deltaf
        return psipcut/vol;
    }
    private:
    container m_f, m_deltafog2d, m_gradpsi;
    geo::DeltaFunction m_deltaf;
    const container m_w2d, m_x, m_y, m_weights;
};
/**
 * @brief Class for the evaluation of the safety factor q
 * \f[ q(\psi_0) = \frac{1}{2\pi} \int dV \alpha( R,Z) H(R,Z) \f]

where \f$ \alpha\f$ is the \c dg::geo::Alpha functor and \c H is a weights function.
 * @copydoc hide_container
 * @ingroup misc_geo
 *
 */
struct SafetyFactor
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param c contains psip, psipR and psipZ
     * @param weights Weight function \c H (can be used to cut away parts of the domain e.g. below the X-point)
     */
    SafetyFactor(const dg::Grid2d& g2d, const TokamakMagneticField& c, const thrust::host_vector<double>& weights) :
    m_g2d(g2d),
    m_alpha(c,0.0,0.0),
    m_w2d ( dg::create::weights( g2d)),
    m_weights( weights)
    {
        thrust::host_vector<double> psipRog2d  = dg::evaluate( c.psipR(), g2d);
        thrust::host_vector<double> psipZog2d  = dg::evaluate( c.psipZ(), g2d);
        double psipRmax = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(), 0.,     thrust::maximum<double>()  );
        //double psipRmin = (double)thrust::reduce( psipRog2d.begin(), psipRog2d.end(),  psipRmax,thrust::minimum<double>()  );
        double psipZmax = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), 0.,      thrust::maximum<double>()  );
        //double psipZmin = (double)thrust::reduce( psipZog2d.begin(), psipZog2d.end(), psipZmax,thrust::minimum<double>()  );
        double deltapsi = fabs(psipZmax/g2d.Ny() +psipRmax/g2d.Nx());
        //m_alpha.setepsilon(deltapsi/4.);
        m_alpha.setepsilon(deltapsi/10.);
    }
    /**
     * @brief Calculate the q profile over the function f which has to be the global safety factor
     * \f[ q(\psi_0) = \frac{1}{2\pi} \int dV \alpha( R,Z) H(R,Z) \f]
     *
     * @param psip0 the actual psi value for q(psi)
     */
    double operator()(double psip0)
    {
        m_alpha.setpsi( psip0);
        m_alphaog2d = dg::evaluate( m_alpha, m_g2d);
        return dg::blas2::dot( m_alphaog2d, m_w2d, m_weights)/(2.*M_PI);
    }
    private:
    dg::Grid2d m_g2d;
    geo::Alpha m_alpha;
    const thrust::host_vector<double> m_w2d;
    thrust::host_vector<double> m_alphaog2d;
    thrust::host_vector<double> m_weights;
};

}//namespace geo

}//namespace dg
