#pragma once

#include <thrust/host_vector.h>
#include "dg/topology/weights.h"
#include "magnetic_field.h"
#include "flux.h"

/*!@file
 *
 * The flux surface average
 */
namespace dg
{
namespace geo
{

/**
 * @brief Flux surface integral of the form
 \f$ \int dR dZ f(R,Z) \delta(\psi_p(R,Z)-\psi_0) g(R,Z) \f$

 where for the width of the Gaussian shaped delta function we use the maximum of \c 0.5*h*GradPsip
     where \c h is the cell size in the grid
 * @ingroup misc_geo
 */
template <class container >
struct FluxSurfaceIntegral
{
     /**
     * @brief Construct from a grid and a magnetic field
     * f and g are default initialized to 1
     * @param g2d grid
     * @param mag contains psip, psipR and psipZ
     * @param width_factor can be used to tune the width of the numerical delta function (\c width = \c 0.5*h*GradPsi*width_factor)
     */
    FluxSurfaceIntegral(const dg::Grid2d& g2d, const TokamakMagneticField& mag, double width_factor = 1.):
            m_f(dg::evaluate(dg::one, g2d)), m_g(m_f), m_delta(m_f),
            m_psi( dg::evaluate( mag.psip(), g2d)),
            m_w2d ( dg::create::weights( g2d))
    {
        thrust::host_vector<double> psipR  = dg::evaluate( mag.psipR(), g2d);
        thrust::host_vector<double> psipZ  = dg::evaluate( mag.psipZ(), g2d);
        double psipRmax = dg::blas1::reduce( psipR, 0., dg::AbsMax<double>()  );
        double psipZmax = dg::blas1::reduce( psipZ, 0., dg::AbsMax<double>()  );
        double deltapsi = 0.5*(psipZmax*g2d.hy() +psipRmax*g2d.hx())/g2d.nx();
        m_eps = deltapsi*width_factor;
    }
    double get_deltapsi() const{return m_eps;}

    /**
     * @brief Set the left function to integrate
     *
     * @param f the container containing the discretized function
     */
    void set_left( const container& f){
        dg::blas1::copy( f, m_f);
    }
    /**
     * @brief Set the right function to integrate
     *
     * @param g the container containing the discretized function
     */
    void set_right( const container& g){
        dg::blas1::copy( g, m_g);
    }
    /**
     * @brief Calculate the Flux Surface Integral
     *
     * @param psip0 the actual psi value of the flux surface
     * @return Int_psip0 (f,g)
     */
    double operator()(double psip0)
    {
        dg::GaussianX delta( psip0, m_eps, 1./(sqrt(2.*M_PI)*m_eps));
        dg::blas1::evaluate( m_delta, dg::equals(), delta, m_psi);
        dg::blas1::pointwiseDot( 1., m_delta, m_f, m_g, 0., m_delta);
        return dg::blas1::dot( m_delta, m_w2d);
    }
    private:
    double m_eps;
    container m_f, m_g, m_delta, m_psi;
    const container m_w2d;
};

//This method for computing volumes is tested against flux-aligned grids in e.g. flux_t.cu
/**
 * @brief Flux volume integral of the form
 \f$ \int dR dZ f(R,Z) \Theta(\psi_p(R,Z)-\psi_0) g(R,Z) \f$

 where \c Theta is the Heaviside function
 * @ingroup misc_geo
 */
template<class container>
struct FluxVolumeIntegral
{
     /**
     * @brief Construct from a grid and a magnetic field
     * f and g are default initialized to 1
     * @param g2d grid
     * @param mag contains psip
     */
    template<class Geometry2d>
    FluxVolumeIntegral(const Geometry2d& g2d, const TokamakMagneticField& mag):
        m_f(dg::evaluate(dg::one, g2d)), m_g(m_f), m_heavi(m_f),
        m_psi( dg::pullback( mag.psip(), g2d)),
        m_w2d ( dg::create::volume( g2d))
    {
    }

    /**
     * @brief Set the left function to integrate
     *
     * @param f the container containing the discretized function
     */
    void set_left( const container& f){
        dg::blas1::copy( f, m_f);
    }
    /**
     * @brief Set the right function to integrate
     *
     * @param g the container containing the discretized function
     */
    void set_right( const container& g){
        dg::blas1::copy( g, m_g);
    }
    /**
     * @brief Calculate the Flux Volume Integral
     *
     * @param psip0 the actual psi value of the flux surface
     * @return Int_0^psip0 (f,g)
     */
    double operator()(double psip0)
    {
        dg::Heaviside heavi( psip0, -1);
        dg::blas1::evaluate( m_heavi, dg::equals(), heavi, m_psi);
        dg::blas1::pointwiseDot( 1., m_heavi, m_f, m_g, 0., m_heavi);
        return dg::blas1::dot( m_heavi, m_w2d);
    }
    private:
    double m_eps;
    container m_f, m_g, m_heavi, m_psi;
    const container m_w2d;
};


/**
 * @brief Flux surface average (differential volume average) over quantity
 \f$ \langle f\rangle(\psi_0) = \frac{1}{A} \int dR dZ \delta(\psi_p(R,Z)-\psi_0) f(R,Z)H(R,Z) \f$

 with \f$ A = \int dRdZ \delta(\psi_p(R,Z)-\psi_0)H(R,Z)\f$
 where \c H is a weight function that can be used to e.g. cut away parts of the domain below the X-point or contain a volume form
 * @ingroup misc_geo
 */
template <class container >
struct FluxSurfaceAverage
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param mag contains psip, psipR and psipZ
     * @param f the function to take the average over (until \c set_container() is called)
     * @param weights Weight function \c H (can be used to cut away parts of the domain e.g. below the X-point and/or contain a volume form without dg weights)
     * @param width_factor can be used to tune the width of the numerical delta function (\c width = \c h*GradPsi*width_factor)
     */
    FluxSurfaceAverage( const dg::Grid2d& g2d, const TokamakMagneticField& mag, const container& f, container weights, double width_factor = 1.) :
    m_avg( g2d,mag, width_factor), m_area( g2d, mag, width_factor)
    {
        m_avg.set_left( f);
        //    container gradpsi  = dg::evaluate( dg::geo::GradPsip( c), g2d);
        //    dg::blas1::pointwiseDot( weights, gradpsi, weights);
        m_avg.set_right( weights);
        m_area.set_right( weights);
    }

    double get_deltapsi() const{return m_avg.get_deltapsi;}

    /**
     * @brief Reset the function to average
     *
     * @param f the container containing the discretized function
     */
    void set_container( const container& f){
        m_avg.set_left( f);
    }
    /**
     * @brief Calculate the Flux Surface Average
     *
     * @param psip0 the actual psi value for q(psi)
     * @return q(psip0)
     */
    double operator()(double psip0)
    {
        return m_avg(psip0)/m_area(psip0);
    }
    private:
    FluxSurfaceIntegral<container> m_avg, m_area;
};





/**
 * @brief Class for the evaluation of the safety factor q based on a flux-surface integral
 * \f$ q(\psi_0) = \frac{1}{2\pi} \int dRdZ \frac{I(\psi_p)}{R} \delta(\psi_p - \psi_0)H(R,Z) \f$

where \c H is a weights function that can optionally be used to cut away parts of the domain e.g. below the X-point.
 * @copydoc hide_container
 * @ingroup misc_geo
 *
 */
struct SafetyFactorAverage
{
     /**
     * @brief Construct from a field and a grid
     * @param g2d 2d grid
     * @param mag contains psip, psipR and psipZ and Ipol
     * @param width_factor can be used to tune the width of the numerical delta function (\c width = \c h*GradPsi*width_factor)
     */
    SafetyFactorAverage(const dg::Grid2d& g2d, const TokamakMagneticField& mag, double width_factor = 1.) :
        m_fsi( g2d, mag, width_factor)
    {
        thrust::host_vector<double> alpha = dg::evaluate( mag.ipol(), g2d);
        thrust::host_vector<double> R = dg::evaluate( dg::cooX2d, g2d);
        dg::blas1::pointwiseDivide( alpha, R, alpha);
        m_fsi.set_left( alpha);
    }
    /// Weight function \c H (can be used to cut away parts of the domain e.g. below the X-point)
    void set_weights( const thrust::host_vector<double>& weights){
        m_fsi.set_right( weights);
    }
    /**
     * @brief Calculate the q(psip0)
     * @param psip0 the flux surface
     * @return q(psip0)
     */
    double operator()(double psip0)
    {
        return m_fsi( psip0)/(2.*M_PI);
    }
    private:
    FluxSurfaceIntegral<thrust::host_vector<double> > m_fsi;
};



/**
 * @brief Evaluation of the safety factor q based on direct integration of
 * \f$ q(\psi_0) = \frac{1}{2\pi} \int d\Theta \frac{B^\varphi}{B^\Theta} \f$

 * @attention Return value undefined if evaluated outside the closed fieldline region, but the function always returns, it won't throw an error or something
 * @copydoc hide_container
 * @ingroup misc_geo
 *
 */
struct SafetyFactor
{
    SafetyFactor( const TokamakMagneticField& mag):
        m_fpsi( mag.get_psip(), mag.get_ipol(), mag.R0(), 0.,false){}

    /**
     * @brief Calculate q(psip0)
     * @param psip0 the flux surface
     * @return q(psip0)
     */
    double operator()( double psip0)
    {
        return 1./m_fpsi( psip0);
    }
private:
    dg::geo::flux::detail::Fpsi m_fpsi;

};

}//namespace geo

}//namespace dg
