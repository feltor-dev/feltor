#pragma once
#include <functional>
#include "dg/backend/memory.h"
#include "dg/topology/geometry.h"

#include "../src/feltor/feltor.h"
#include "../src/feltor/parameters.h"
 //#include "feltor/init.h"

 namespace dg
 {
 namespace geo
 {
 ///@addtogroup fluxfunctions
 /**
  * @brief Certain functions related with using Nabla (in divergences, perpendicular gradients, vector dot nabla...)
  * @ingroup misc_geo
  */

 template<class Geometry, class Container, class Matrix>                                                                                                                            
 struct Nablas 
 {
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = get_value_type<Container>;

 /**
  * @brief Main contructor: construct from a 3D geometry and also with a Magnetic field and a bool parameter in case there is reversed field
  * @tparam Geometry
  * @param geom3d
  * @tparam g::geo::TokamakMagneticField
  * @param mag
  * @tparam  bool
  * @param reversed
  */

 Nablas(const Geometry& geom3d, dg::geo::TokamakMagneticField& mag, bool reversed): m_g(geom3d), m_mag(mag), m_reversed_field(reversed) {
     dg::blas2::transfer( dg::create::dx( m_g, dg::DIR, dg::centered), m_dR);
     dg::blas2::transfer( dg::create::dy( m_g, dg::DIR, dg::centered), m_dZ);
     m_vol=dg::tensor::volume(m_g.metric());
     m_tmp=m_tmp2=m_tmp3=m_vol;
     m_hh=m_g.metric();

     auto bhat = dg::geo::createBHat( m_mag);
     bhat = dg::geo::createEPhi(+1);
     if( m_reversed_field)
     bhat = dg::geo::createEPhi(-1);
     m_hh = dg::geo::createProjectionTensor( bhat, m_g);

     }

    /**
      * @brief Perpendicular gradient of function f (output contravariant): \f[( \boldsymbol{\nabla_\perp)f)^i = h^{ij}\partial_j f \f]
      * @param f the container containing the scalar
      * @param grad_R container containing the R component of the perpendicular gradient
      * @param grad_Z container containing the Z component of the perpendicular gradient
      */
      
    template<class Container1>
    void Grad_perp_f(const Container1& f, Container1& grad_R, Container1& grad_Z) {
        dg::blas2::symv( m_dR, f, grad_R);
        dg::blas2::symv( m_dZ, f, grad_Z); //OUTPUT: COVARIANT
        dg::tensor::multiply2d(m_metric, grad_R, grad_Z, grad_R, grad_Z); //IF ACTIVE OUTPUT: CONTRAVARIANT
 	}
 	/**
      * @brief Divergence of a perpendicular vector field (input contravariant): \f[ \boldsymbol(\nabla)\cdot\boldsymbol{v}=\frac{1}{\sqrt{g}}\partial_i(\sqrt{g}v^i) \f] only in the perpendicular plane.
      * @param v_R container containing the R component of the perpendicular gradient
      * @param v_Z container containing the Z component of the perpendicular gradient
      * @param F the container containing the divergence result
      */

 	template<class Container1>		
 	void div (const Container1& v_R, const Container1& v_Z, Container1& F){ //INPUT: CONTRAVARIANT
 	dg::blas1::pointwiseDot(v_R, m_vol, m_tmp);
 	dg::blas1::pointwiseDot(v_Z, m_vol, m_tmp2);
 	dg::blas2::symv( m_dR, m_tmp, m_tmp3); 
 	dg::blas2::symv( m_dZ, m_tmp2,F);
 	dg::blas1::axpby(1.0, m_tmp3, 1.0, F);
 	dg::blas1::pointwiseDivide(F, m_vol,F);

 }


 	/**
      * @brief Vector dot nabla f: gradient in a vector direction (covariant) of a scalar (usually the scalar being different components of a vector): \f[\boldsymbol{v}\cdot{boldsymbol{\nabla}f=v_ih^{ij}\partial_j f \f]
      *
      * @param v_R container containing the R component of the vector of the direction
      * @param v_Z container containing the Z component of the vector of the direction
      * @param f the scalar over which the derivatives are done
      * @param F the scalar output
      */

    template<class Container1>
    void v_dot_nabla_f (const Container1& v_R, const Container1& v_Z, const Container1& f, Container1& F){ //INPUT: COVARIANT
        dg::blas2::symv( m_dR, f, m_tmp);
        dg::blas2::symv( m_dZ, f, m_tmp2);
        dg::tensor::multiply2d(m_hh, m_tmp, m_tmp2, m_tmp3, F); //WE MAKE THE GRADIENT CONTRAVARIANT
        dg::blas1::pointwiseDot(1.0, v_R, m_tmp3, 1.0,  v_Z, F, 1.0, F);
    }

    template<class Container1>
    void grad_perp_f (const Container1& f, Container1& F_R, Container1& F_Z){ //INPUT: COVARIANT
        dg::blas2::symv( m_dR, f, m_tmp);
        dg::blas2::symv( m_dZ, f, m_tmp2);
        dg::tensor::multiply2d(m_hh, m_tmp, m_tmp2, F_R, F_Z); //WE MAKE THE GRADIENT CONTRAVARIANT
    }
 
    private:
    Geometry m_g;
    feltor::Parameters m_p;
    dg::geo::TokamakMagneticField m_mag;
    dg::SparseTensor<Container > m_metric, m_hh;
     bool m_reversed_field;
    Matrix m_dR;
    Matrix m_dZ;
    Container m_vol;
    Container m_tmp, m_tmp2, m_tmp3;
 };
 };//namespace geo
 }//namespace dg
