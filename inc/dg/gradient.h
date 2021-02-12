#pragma once

#include "blas.h"
#include "enums.h"
#include "backend/memory.h"
#include "topology/evaluation.h"
#include "topology/derivatives.h"
#ifdef MPI_VERSION
#include "topology/mpi_derivatives.h"
#include "topology/mpi_evaluation.h"
#endif
#include "topology/geometry.h"

/*! @file

  @brief General gradient operator
  */
namespace dg
{

/**
 * @brief A 2d gradient \f$\chi\cdot\nabla\f$ and variation \f$ \nabla\phi \cdot \chi \nabla\phi\f$ operator
 *
 * @ingroup matrixoperators
 *
 * The terms discretized are the gradient \f[\chi\cdot\nabla\f] and the variation \f[ \nabla\phi \cdot \chi \nabla\phi\f]
 * where \f$ \nabla \f$ is the two-dimensional nabla and \f$\chi\f$ is a
 * tensor (usually the metric).
 *
 * In general coordinates that means
 * \f[\chi\cdot\nabla = \left(\chi^{xx}\partial_x + \chi^{xy}\partial_y \right)
 + \left(\chi^{yx}\partial_x + \chi^{yy}\partial_y \right) \f]
 is discretized.
 Per default, \f$ \chi\f$ is the metric tensor but you can set it to any tensor
 you like.

 * @copydoc hide_geometry_matrix_container
 * @note The constructors initialize \f$ \chi=g\f$ so that a traditional
 * gradient results
 * @attention This a convenience class. It is often more
 * efficient to compute the simple derivatives of a vector yourself, because
 * you can re-use them in other places; the same goes for the storage of the
 * metric tensor, it often can be re-used at other places.  To compute the above
 * expressions you then simply use the relevant tensor functions
 * \c dg::tensor::multiply2d and \c dg::tensor::scalar_product2d
 */
template <class Geometry, class Matrix, class Container>
class Gradient
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Gradient(){}
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid, boundary conditions are taken from here
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @note chi is assumed the metric per default
     */
    Gradient( const Geometry& g, direction dir = centered):
        Gradient( g, g.bcx(), g.bcy(), dir)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * @note chi is assumed the metric per default
     */
    Gradient( const Geometry& g, bc bcx, bc bcy, direction dir = centered)
    {
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        m_chi=g.metric();
        m_tempx = m_tempy = dg::construct<Container>( dg::evaluate( dg::zero, g));
    }

    /**
    * @brief Perfect forward parameters to one of the constructors
    *
    * @tparam Params deduced by the compiler
    * @param ps parameters forwarded to constructors
    */
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Gradient( std::forward<Params>( ps)...);
    }

    ///@copydoc Gradient3d::chi()
    SparseTensor<Container>& chi( ){return m_chi;}
    ///@brief Access the Chi tensor
    const SparseTensor<Container>& chi( ) const{return m_chi;}

    /**
     * @brief \f$ \vec v=\chi \cdot\nabla f \f$
     *
     * @param f the vector to take the gradient of
     * @param vx (output) x-component
     * @param vy (output) y-component
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2>
    void gradient( const ContainerType0& f, ContainerType1& vx, ContainerType2& vy){
        dg::blas2::gemv( m_rightx, f, vx); //R_x*f
        dg::blas2::gemv( m_righty, f, vy); //R_y*f
        dg::tensor::multiply2d(1., m_chi, vx, vy, 0., vx, vy);
    }

    /**
     * @brief \f$ \vec v = \lambda \chi\cdot\nabla f + \mu \vec v \f$
     *
     * @param lambda a prefactor
     * @param f the vector to take the gradient of
     * @param mu the output prefactor
     * @param vx (inout) x-component
     * @param vy (inout) y-component
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2,
        class ContainerType3, class ContainerType4>
    void gradient(const ContainerType0& lambda, const ContainerType1& f, const
            ContainerType2& mu, ContainerType3& vx, ContainerType4& vy)
    {
        //compute gradient
        dg::blas2::gemv( m_rightx, f, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, f, m_tempy); //R_y*f
        dg::tensor::multiply2d(lambda, m_chi, m_tempx, m_tempy, mu, vx, vy);
    }
    /**
     * @brief \f$ \sigma = (\nabla\phi\cdot\chi\cdot\nabla \phi) \f$
     *
     * @param phi the vector to take the variation of
     * @param sigma (inout) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void variation(const ContainerType0& phi, ContainerType1& sigma){
        variation(1., 1., phi, 0., sigma);
    }
    /**
     * @brief \f$ \sigma = \lambda^2(\nabla\phi\cdot\chi\cdot\nabla \phi) \f$
     *
     * @param lambda input prefactor
     * @param phi the vector to take the variation of
     * @param sigma (out) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(const ContainerTypeL& lambda, const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,lambda, phi, 0., sigma);
    }
    /**
     * @brief \f$ \sigma = \alpha \lambda^2 (\nabla\phi\cdot\chi\cdot\nabla \phi) + \beta \sigma\f$
     *
     * @param alpha scalar input prefactor
     * @param lambda input prefactor
     * @param phi the vector to take the variation of
     * @param beta the output prefactor
     * @param sigma (inout) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(value_type alpha, const ContainerTypeL& lambda, const ContainerType0& phi, value_type beta, ContainerType1& sigma)
    {
        dg::blas2::gemv( m_rightx, phi, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, phi, m_tempy); //R_y*f
        dg::tensor::scalar_product2d(alpha, lambda, m_tempx, m_tempy, m_chi, lambda, m_tempx, m_tempy, beta, sigma);
    }
    private:
    Matrix m_rightx, m_righty;
    Container m_tempx, m_tempy;
    SparseTensor<Container> m_chi;
};

///@copydoc Gradient
///@ingroup matrixoperators
template <class Geometry, class Matrix, class Container>
using Gradient2d = Gradient<Geometry, Matrix, Container>;

/**
 * @brief A 3d gradient \f$\chi\cdot\nabla\f$ and variation \f$ \nabla\phi \cdot \chi \nabla\phi\f$ operator
 *
 * @ingroup matrixoperators
 *
 * The terms discretized are the gradient \f$\chi\cdot\nabla\f$ and the variation \f[ \nabla\phi \cdot \chi \nabla\phi\f]
 * where \f$ \mathbf \chi \f$ is a tensor (usually the metric).
 * In general coordinates that means
 * \f[ \chi\cdot\nabla =
 * \left(\chi^{xx}\partial_x + \chi^{xy}\partial_y + \chi^{xz}\partial_z \right)
 + \left(\chi^{yx}\partial_x + \chi^{yy}\partial_y + \chi^{yz}\partial_z \right)
 + \left(\chi^{zx}\partial_x + \chi^{zy}\partial_y + \chi^{zz}\partial_z \right)
 \f]
 Per default, \f$ \chi\f$ is the metric tensor but you can set it to any tensor
 you like.
 * @copydoc hide_geometry_matrix_container
 * @note The constructors initialize \f$ \chi=g\f$ so that a traditional gradient
 * results
 * @attention This a convenience class. It is often more
 * efficient to compute the simple derivatives of a vector yourself, because
 * you can re-use them in other places; the same goes for the storage of the
 * metric tensor, it often can be re-used at other places.  To compute the above
 * expressions you then simply use the relevant tensor functions
 * \c dg::tensor::multiply3d and \c dg::tensor::scalar_product3d
 */
template <class Geometry, class Matrix, class Container>
class Gradient3d
{
    public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    Gradient3d(){}
    /**
     * @brief Construct from Grid
     *
     * @param g The Grid; boundary conditions are taken from here
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * the direction of the z derivative is always \c dg::centered
     * @note chi is assumed the metric per default
     */
    Gradient3d( const Geometry& g, direction dir = centered):
        Gradient3d( g, g.bcx(), g.bcy(), g.bcz(), dir)
    {
    }

    /**
     * @brief Construct from grid and boundary conditions
     * @param g The Grid
     * @param bcx boundary condition in x
     * @param bcy boundary contition in y
     * @param bcz boundary contition in z
     * @param dir Direction of the right first derivative in x and y
     *  (i.e. \c dg::forward, \c dg::backward or \c dg::centered),
     * the direction of the z derivative is always \c dg::centered
     * @note chi is assumed the metric per default
     */
    Gradient3d( const Geometry& g, bc bcx, bc bcy, bc bcz, direction dir = centered)
    {
        dg::blas2::transfer( dg::create::dx( g, bcx, dir), m_rightx);
        dg::blas2::transfer( dg::create::dy( g, bcy, dir), m_righty);
        dg::blas2::transfer( dg::create::dz( g, bcz, dg::centered), m_rightz);
        m_chi=g.metric();
        m_tempx = m_tempy = m_tempz = dg::construct<Container>( dg::evaluate( dg::zero, g));
    }
    ///@copydoc Gradient::construct()
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Gradient3d( std::forward<Params>( ps)...);
    }

    ///@brief Access the Chi tensor
    SparseTensor<Container>& chi( ){return m_chi;}
    ///@brief Access the Chi tensor
    const SparseTensor<Container>& chi( ) const{return m_chi;}

    /**
     * @brief Restrict the problem to the first 2 dimensions
     *
     * This effectively makes the behaviour of dg::Gradient3d
     * identical to the dg::Gradient class.
     * @param compute_in_2d if true, the gradient and variaton functions replace all derivatives in z with 0, false reverts to the original behaviour.
     */
    void set_compute_in_2d( bool compute_in_2d ) {
        m_multiplyZ = !compute_in_2d;
    }
    /**
     * @brief \f$ \vec v=\chi \cdot\nabla f \f$
     *
     * @param f the vector to take the gradient of
     * @param vx (output) x-component
     * @param vy (output) y-component
     * @param vz (output) z-component (0, if set_compute_in_2d(true) was set)
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2, class ContainerType3>
    void gradient( const ContainerType0& f, ContainerType1& vx, ContainerType2& vy, ContainerType3& vz){
        //compute gradient
        dg::blas2::gemv( m_rightx, f, vx); //R_x*f
        dg::blas2::gemv( m_righty, f, vy); //R_y*f
        if( m_multiplyZ)
            dg::blas2::gemv( m_rightz, f, vz); //R_y*f
        else
            dg::blas1::scal( vz, 0.);
        dg::tensor::multiply3d(1., m_chi, vx, vy, vz, 0., vx, vy, vz);
    }

    /**
     * @brief \f$ \vec v = \lambda \chi\cdot\nabla f + \mu \vec v \f$
     *
     * @param lambda a prefactor
     * @param f the vector to take the gradient of
     * @param mu the output prefactor
     * @param vx (inout) x-component
     * @param vy (inout) y-component
     * @param vz (inout) z-component
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1, class ContainerType2,
        class ContainerType3, class ContainerType4, class ContainerType5>
    void gradient(const ContainerType0& lambda, const ContainerType1& f, const
            ContainerType2& mu, ContainerType3& vx, ContainerType4& vy, ContainerType5& vz)
    {
        //compute gradient
        dg::blas2::gemv( m_rightx, f, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, f, m_tempy); //R_y*f
        if( m_multiplyZ)
            dg::blas2::gemv( m_rightz, f, m_tempz); //R_y*f
        else
            dg::blas1::scal( m_tempz, 0.);
        dg::tensor::multiply3d(lambda, m_chi, m_tempx, m_tempy, m_tempz, mu, vx, vy, vz);
    }
    /**
     * @brief \f$ \sigma = (\nabla\phi\cdot\chi\cdot\nabla \phi) \f$
     *
     * @param phi the vector to take the variation of
     * @param sigma (out) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerType0, class ContainerType1>
    void variation(const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,1., phi, 0., sigma);
    }
    /**
     * @brief \f$ \sigma = \lambda^2(\nabla\phi\cdot\chi\cdot\nabla \phi) \f$
     *
     * @param lambda input prefactor
     * @param phi the vector to take the variation of
     * @param sigma (out) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(const ContainerTypeL& lambda, const ContainerType0& phi, ContainerType1& sigma){
        variation(1.,lambda, phi, 0., sigma);
    }
    /**
     * @brief \f$ \sigma = \alpha\lambda^2 (\nabla\phi\cdot\chi\cdot\nabla \phi) + \beta \sigma\f$
     *
     * @param alpha scalar input prefactor
     * @param lambda input prefactor
     * @param phi the vector to take the variation of
     * @param beta the output prefactor
     * @param sigma (inout) the variation
     * @tparam ContainerTypes must be usable with \c Container in \ref dispatch
     */
    template<class ContainerTypeL, class ContainerType0, class ContainerType1>
    void variation(value_type alpha, const ContainerTypeL& lambda, const ContainerType0& phi, value_type beta, ContainerType1& sigma)
    {
        dg::blas2::gemv( m_rightx, phi, m_tempx); //R_x*f
        dg::blas2::gemv( m_righty, phi, m_tempy); //R_y*f
        if( m_multiplyZ)
            dg::blas2::gemv( m_rightz, phi, m_tempz); //R_y*f
        else
            dg::blas1::scal( m_tempz, 0.);
        dg::tensor::scalar_product3d(alpha, lambda,  m_tempx, m_tempy, m_tempz, m_chi, lambda, m_tempx, m_tempy, m_tempz, beta, sigma);
    }
    private:
    Matrix m_rightx, m_righty, m_rightz;
    Container m_tempx, m_tempy, m_tempz;
    SparseTensor<Container> m_chi;
    bool m_multiplyZ = true;
};

} //namespace dg
