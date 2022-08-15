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


struct Grid_cutter : public aCylindricalFunctor<Grid_cutter>
{
		/**
    * @brief Cuts a 2D X-grid from a certain central poloidal position (horizontal line in the X-grid) to a range around it (a width in the y direction around the center). 
    *
    * \f[ f(zeta,eta)= \begin{cases}
	*1 \text{ if } eta_0-eta_size/2< eta < eta_0+eta_size/2 \\
	*0 \text{ else }
	*\end{cases}
	*\f]
    * 
    * 
    * @brief <tt> Grid_cutter( eta_0, eta_size) </tt>
    * @tparam double
    * @param eta_0 (center of the range you want, in radians)
    * @tparam double
    * @param eta_size (width of the poloidal range you want to cut, in degrees)
    * 
    * @note How to use it? dg::evaluate(dg::geo::Grid_cutter(eta, Range), GridX2d()); After you have it, you usually pointwise this function to a matrix of data to apply the cut to your data: dg::blas1::pointwiseDot(data, dg::evaluate(dg::geo::Grid_cutter(eta, Range), GridX2d()), cutted_data);
	**/
	

    Grid_cutter(double eta_0, double eta_size): eta0(eta_0), etasize(eta_size){} //eta_0 is in radians and eta_size is in degrees
    
    double do_compute(double zeta, double eta) const { //go over all the point in the grid to return 1 or 0
	double eta_up_lim=eta0+etasize*M_PI/(2*180); //Define the upper and down limits of the cut  !!!IF THIS COULD BE DONE OUT OF THE LOOP, IT WOULD MAKE EVERYTHING EASIER!!! NO SENSE TO DO IT IN  EVERY ITERATION.
    double eta_down_lim=eta0-etasize*M_PI/(2*180);
    
    //As the grid goes from 0 to 2pi, we need to check that the lower limit is not under 0 or the higher over 2pi.
    // If that happens, we need to translate the limits to our range and change the conditions of our loops
    if (eta_up_lim>2*M_PI) {		
		eta_up_lim+=-2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))
            return 1;
        return 0;
	}
    if (eta_down_lim<0)  {
		eta_down_lim+=2*M_PI;
        if( (eta<eta_up_lim || eta>eta_down_lim))
            return 1;
        return 0;   
	}
    else
    {
        if( eta<eta_up_lim && eta>eta_down_lim)
            return 1;
        return 0;
	}
    }
    private:
    double eta0, etasize;
}; 


struct radial_cut
{
		/**
    * @brief Takes the radial cut of a quantitity in 2D, so it shows the poloidal distribution of a quantity at a certain flux surface
    *
    * 
    * @brief <tt> radial_cut(RealCurvilinearGridX2d<double>) </tt>
    * @tparam HVec
    * @param F (function to obtain the poloidal distribution)
    * @tparam double
    * @param zeta_def (flux surface where the poloidal distribution wants to be seen)
    * 
    */
	radial_cut(RealCurvilinearGrid2d<double> gridX2d): m_g(gridX2d){} //Changed from curvilinearGridX2d because it didn't compile
	
	HVec cut(const HVec F, const double zeta_def){ //This functions takes a 2D object in the Xgrid plane at a define radial position and saves it in a 1D variable with eta dependence.
	dg::Grid1d g1d_out_eta(m_g.y0(), m_g.y1(), m_g.n(), m_g.Ny(), dg::DIR_NEU); 
	m_conv_LCFS_F=dg::evaluate( dg::zero, g1d_out_eta);
	unsigned int zeta_cut=round(((zeta_def-m_g.x0())/m_g.lx())*m_g.Nx()*m_g.n())-1;

	for (unsigned int eta=0; eta<m_g.n()*m_g.Ny(); eta++) 
	{m_conv_LCFS_F[eta]=F[eta*m_g.n()*m_g.Nx()+zeta_cut];};
	return m_conv_LCFS_F;	
	}
	
	HVec cut2(const HVec F, const double zeta_min, const double zeta_max){ //This functions takes a 2D object in the Xgrid plane and cuts it between the two introduced Zetas
	dg::Grid1d g1d_out_eta(m_g.y0(), m_g.y1(), m_g.n(), m_g.Ny(), dg::DIR_NEU); 
	m_conv_LCFS_F=dg::evaluate( dg::zero, g1d_out_eta);
	unsigned int zeta_min_in=round(((zeta_min-m_g.x0())/m_g.lx())*m_g.Nx()*m_g.n())-1;
	unsigned int zeta_max_in=round(((zeta_max-m_g.x0())/m_g.lx())*m_g.Nx()*m_g.n())-1;
	
	for (unsigned int zeta=zeta_min_in; zeta<zeta_max_in; zeta++)
	for (unsigned int eta=0; eta<m_g.n()*m_g.Ny(); eta++) 
	{m_conv_LCFS_F[eta+(zeta-zeta_min_in)*m_g.n()*m_g.Nx()]=F[eta*m_g.n()*m_g.Nx()+zeta];};
	return m_conv_LCFS_F;	
	}
	
	private:
	RealCurvilinearGrid2d<double> m_g; //Changed from curvilinearGridX2d because it didn't compile	
	HVec m_conv_LCFS_F;
	
};
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
     * @brief Alternative contructor: construct from a 3D geometry and also with a Magnetic field and the input parameters in case there is reversed field
     * @tparam Geometry
     * @param geom3d
     * @tparam g::geo::TokamakMagneticField
     * @param mag
     * @param feltor::Parameters p
     */
	
	Nablas(const Geometry& geom3d, feltor::Parameters p, dg::geo::TokamakMagneticField& mag): m_g(geom3d), m_p(p), m_mag(mag) { 
		dg::blas2::transfer( dg::create::dx( m_g, dg::DIR, dg::centered), m_dR); 
		dg::blas2::transfer( dg::create::dy( m_g, dg::DIR, dg::centered), m_dZ);
		dg::blas2::transfer( dg::create::dz( m_g, dg::DIR, dg::centered), m_dtor);
		m_vol=dg::tensor::volume(m_g.metric());
		m_weights=dg::create::volume(m_g);
		m_tmp=m_tmp2=m_tmp3=m_tmp4=m_weights;
		m_hh=m_g.metric();
		
		auto bhat = dg::geo::createBHat( m_mag);
		bhat = dg::geo::createEPhi(+1);
		m_reversed_field = false;
		if( m_mag.ipol()( m_g.x0(), m_g.y0()) < 0)
        m_reversed_field = true;
		if( p.curvmode == "true")
        bhat = dg::geo::createBHat(m_mag);
		else if( m_reversed_field)
        bhat = dg::geo::createEPhi(-1);
		m_hh = dg::geo::createProjectionTensor( bhat, m_g);
		
		} 
		

	/*
     * @brief Perpendicular gradient of function f (output contravariant)
     *
     * @param f the container containing the scalar
     * @param grad_R container containing the R component of the perpendicular gradient
     * @param grad_Z container containing the Z component of the perpendicular gradient
     
     
	template<class Container1>
	void Grad_perp_f(const Container1& f, Container1& grad_R, Container1& grad_Z) { 
	dg::blas2::symv( m_dR, f, grad_R);
	dg::blas2::symv( m_dZ, f, grad_Z); //OUTPUT: COVARIANT
	//dg::tensor::multiply2d(m_metric, grad_R, grad_Z, grad_R, grad_Z) //IF ACTIVE OUTPUT: CONTRAVARIANT
	}		
	*/
	/**
     * @brief Divergence of a perpendicular vector field (input contravariant)
     *
     * @param v_R container containing the R component of the perpendicular gradient
     * @param v_Z container containing the Z component of the perpendicular gradient
     * @param F the container containing the divergence result
     */
	
	template<class Container1>		
	void div (const Container1& v_R, const Container1& v_Z, Container1& F){ //INPUT: CONTRAVARIANT
	dg::blas1::pointwiseDivide(v_R, m_vol, m_tmp);
	dg::blas1::pointwiseDivide(v_Z, m_vol, m_tmp2); 
	dg::blas2::symv( m_dR, m_tmp, m_tmp3); 
	dg::blas2::symv( m_dZ, m_tmp2, m_tmp4);
	dg::blas1::axpby(1, m_tmp3, 1, m_tmp4);
	dg::blas1::pointwiseDot(m_vol, m_tmp4,F);	
	
}

/**
     * @brief Divergence of a vector field even with parallel direction (input contravariant) (Usefull for parallel divergences)
     *
     * @param v_R container containing the R component of the vector field
     * @param v_Z container containing the Z component of the vector field
     * @param v_tor container containing the toroidal component of the vector field
     * @param F the container containing the divergence result
     */
	
	template<class Container1>		
	void div_par (const Container1& v_R, const Container1& v_Z, const Container1& v_tor, Container1& F){ //INPUT: CONTRAVARIANT
	dg::blas1::pointwiseDivide(v_R, m_vol, m_tmp);
	dg::blas1::pointwiseDivide(v_Z, m_vol, m_tmp2); 
	dg::blas1::pointwiseDivide(v_tor, m_vol, m_tmp3);
	dg::blas1::pointwiseDivide(m_tmp3, m_vol, m_tmp3); //DIVIDED TWICE BECAUSE IT WILL COME FROM bhatgb which is divided by sqrt(g), so we need to devide twice by the volume to get the right front factor.
	dg::blas2::symv( m_dR, m_tmp, m_tmp4); 
	dg::blas2::symv( m_dZ, m_tmp2, m_tmp5);
	dg::blas2::symv( m_dtor, m_tmp3, m_tmp6);
	dg::blas1::axpbypgz(1, m_tmp4, 1, m_tmp5, 1, m_tmp6);
	dg::blas1::pointwiseDot(m_vol, m_tmp6,F);	
	
}

	/**
     * @brief Vector dot nabla f: gradient in a vector direction (covariant) of a scalar (usually the scalar being different components of a vector)
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
	dg::tensor::multiply2d(m_hh, m_tmp, m_tmp2, m_tmp3, m_tmp4); //WE MAKE THE GRADIENT CONTRAVARIANT
	dg::blas1::pointwiseDot(v_R, m_tmp3, m_tmp3);
	dg::blas1::pointwiseDot(v_Z, m_tmp4, F);
	dg::blas1::axpby(1.0, m_tmp3, 1.0, F);
	}	
	
	template<class Container1>
	void grad_perp_f (const Container1& f, Container1& F_R, Container1& F_Z){ //INPUT: COVARIANT
	dg::blas2::symv( m_dR, f, m_tmp);
	dg::blas2::symv( m_dZ, f, m_tmp2);
	dg::tensor::multiply2d(m_hh, m_tmp, m_tmp2, F_R, F_Z); //WE MAKE THE GRADIENT CONTRAVARIANT
	}	
	
	/*
     * @brief b cross v: vectorial product of vector field unitary vector and vector v (covariant) with the toroidal magnetic field approximation
     * 
     * @param v_R_o container containing the R component of the input vector
     * @param v_Z_o container containing the Z component of the input vector
     * @param v_R_f container containing the R component of the output vector (contravariant)
     * @param v_Z_f container containing the Z component of the output vector (contravariant)
     
	
	template<class Container1>
	void b_cross_v (const Container1& v_R_o, const Container1& v_Z_o, Container1& v_R_f, Container1& v_Z_f){ //INPUT: COVARIANT
	dg::tensor::multiply2d(m_hh, v_R_o, v_Z_o, m_tmp, m_tmp2); //to transform the vector from covariant to contravariant
    dg::blas1::pointwiseDot(-1, m_tmp2, m_tmp2);
    dg::blas1::pointwiseDot(m_vol, m_tmp2, v_R_f);       
	dg::blas1::pointwiseDot(m_vol, m_tmp, v_Z_f); //OUTPUT: CONTRAVARIANT
	}
	*/
	private:
	Geometry m_g;
	feltor::Parameters m_p;
	dg::geo::TokamakMagneticField m_mag;
    dg::SparseTensor<Container > m_metric, m_hh;
    bool m_reversed_field;
	Matrix m_dR;
	Matrix m_dZ;
	Matrix m_dtor;
	Container m_vol;
	Container m_weights;
    Container m_tmp, m_tmp2, m_tmp3, m_tmp4, m_tmp5, m_tmp6; 
};
};//namespace geo
}//namespace dg


