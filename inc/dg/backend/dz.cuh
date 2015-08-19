#pragma once
#include <cusp/transpose.h>
#include <cusp/print.h>
#include "grid.h"
#include "../blas.h"
#include "interpolation.cuh"
#include "fieldaligned.h"
#include "typedefs.cuh"
#include "functions.h"
#include "derivatives.cuh"
#include "../functors.h"
#include "transpose.h"

namespace dg{

/**
 * @brief Default Limiter means there is a limiter everywhere
 */
struct DefaultLimiter
{

    /**
     * @brief return 1
     *
     * @param x x value
     * @param y y value
     *
     * @return 1
     */
    double operator()(double x, double y)
    {
        return 1;
    }

};
/**
 * @brief No Limiter 
 */
struct NoLimiter
{

    /**
     * @brief return 0
     *
     * @param x x value
     * @param y y value
     *
     * @return 0
     */
    double operator()(double x, double y)
    {
        return 0.;
    }

};

////////////////////////////////////DZCLASS////////////////////////////////////////////
/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
cylindrical coordinates
* @ingroup dz
* @tparam Matrix The matrix class of the interpolation matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class Matrix = dg::DMatrix, class container=thrust::device_vector<double> >
struct DZ
{

    /**
    * @brief Construct from a field and a grid
    *
    * @tparam Field The Fieldlines to be integrated: Has to provide void operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) where the first index is R, the second Z and the last s (the length of the field line)
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
    is a limiter and 0 if there isn't. If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply. 
    * @param field The field to integrate
    * @param grid The grid on which to operate
    * @param deltaPhi Must either equal the hz() value of the grid or a fictive deltaPhi if the grid is 2D and Nz=1
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere, note that if bcz is periodic it doesn't matter if there is a limiter or not)
    * @param globalbcz Choose NEU or DIR. Defines BC in parallel on box
    * @note If there is a limiter, the boundary condition is set by the bcz variable from the grid and can be changed by the set_boundaries function. If there is no limiter the boundary condition is periodic.
    */
    template <class Field, class Limiter>
    DZ(Field field, const dg::Grid3d<double>& grid, double deltaPhi, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR);

    /**
    * @brief Apply the derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void forward( const container& f, container& dzf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void backward( const container& f, container& dzf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void centered( const container& f, container& dzf);

    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void forwardT( const container& f, container& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void backwardT( const container& f, container& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void centeredTD( const container& f, container& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void forwardTD( const container& f, container& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void backwardTD( const container& f, container& dzf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void centeredT( const container& f, container& dzf);
    /**
    * @brief is the negative transposed of centered
    *
    * redirects to centered
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    
    void operator()( const container& f, container& dzf);



    /**
     * @brief Discretizes the parallel Laplacian as a symmetric matrix
     *
     * forward followed by forwardT and adding jump terms
     * @param f The vector to derive
     * @param dzTdzf contains result on output (write only)
     */
    void symv( const container& f, container& dzTdzf);

    /**
    * @brief Set boundary conditions in the limiter region
    *
    * if Dirichlet boundaries are used the left value is the left function
    value, if Neumann boundaries are used the left value is the left derivative value
    * @param bcz boundary condition
    * @param left left boundary value
    * @param right right boundary value
    */
    void set_boundaries( dg::bc bcz, double left, double right)
    {

        bcz_ = bcz;
        const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
        left_ = dg::evaluate( dg::CONSTANT(left), g2d);
        right_ = dg::evaluate( dg::CONSTANT(right),g2d);
    }
    /**
     * @brief Set boundary conditions in the limiter region
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param left left boundary value
     * @param right right boundary value
    */
    void set_boundaries( dg::bc bcz, const container& left, const container& right)
    {
        bcz_ = bcz;
        left_ = left;
        right_ = right;
    }
    /**
     * @brief Set boundary conditions in the limiter region
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param global 3D vector containing boundary values
     * @param scal_left left scaling factor
     * @param scal_right right scaling factor
     */
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right);
    /**
     * @brief Compute the second derivative using finite differences
     *
     * discretizes \f$ \nabla_\parallel\cdot \nabla_\parallel\f$
     * @param f input function
     * @param dzzf output (write-only)
     */
    void dzz( const container& f, container& dzzf);



    /**
     * @brief Returns the weights used to make the matrix symmetric 
     *
     * needed by invert class
     * @return weights
     */
    const container& weights()const {return w3d;}
    /**
     * @brief Returns the preconditioner to use in conjugate gradient
     *
     * needed by invert class
     * In this case inverse weights are the best choice
     * @return inverse weights
     */
    const container& precond()const {return v3d;}
    private:
    void eins(const Matrix& interp, const container& n, container& npe);
    void einsPlus( const container& n, container& npe);
    void einsMinus( const container& n, container& nme);
    void einsPlusT( const container& n, container& npe);
    void einsMinusT( const container& n, container& nme);
    typedef cusp::array1d_view< typename container::iterator> View;
    typedef cusp::array1d_view< typename container::const_iterator> cView;
    Matrix plus, minus, plusT, minusT; //interpolation matrices
    Matrix plus_f, minus_f, plusT_f, minusT_f; 
    Matrix jumpX, jumpY;

    container hz, hp,hm, tempP, temp0, tempM, ghostM, ghostP;
    dg::Grid3d<double> g_;
    dg::bc bcz_;
    container left_, right_;
    container limiter;
    container w3d, v3d;
    container invB;
};

////////////////////////////////////DEFINITIONS////////////////////////////////////////
///@cond
template<class M, class container>
template <class Field, class Limiter>
DZ<M,container>::DZ(Field field, const dg::Grid3d<double>& grid, double deltaPhi, double eps, Limiter limit, dg::bc globalbcz):
        jumpX( dg::create::jumpX( grid, grid.bcx())),
        jumpY( dg::create::jumpY( grid, grid.bcy())),
        hz( dg::evaluate( dg::zero, grid)), hp( hz), hm( hz), tempP( hz), temp0( hz), tempM( hz), 
        g_(grid), bcz_(grid.bcz()), w3d( dg::create::weights( grid)), v3d( dg::create::inv_weights( grid)),
        invB(dg::evaluate(field,grid))
{

    dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());  
    unsigned size = g2d.size();
    limiter = dg::evaluate( limit, g2d);
    right_ = left_ = dg::evaluate( zero, g2d);
    ghostM.resize( size); ghostP.resize( size);
    //Set starting points
    std::vector<dg::HVec> y( 3, dg::evaluate( dg::coo1, g2d)), yp(y), ym(y);
    y[1] = dg::evaluate( dg::coo2, g2d);
    y[2] = dg::evaluate( dg::zero, g2d);
    thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
  
//     integrate field lines for all points
    for( unsigned i=0; i<size; i++)
    {
        coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i];

        double phi1 = deltaPhi;
        boxintegrator( field, g2d, coords, coordsP, phi1, eps, globalbcz);
        phi1 =  - deltaPhi;
        boxintegrator( field, g2d, coords, coordsM, phi1, eps, globalbcz);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];

    }
    plus  = dg::create::interpolation( yp[0], yp[1], g2d, globalbcz);
    minus = dg::create::interpolation( ym[0], ym[1], g2d, globalbcz);
// //     Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix and MPI_Matrix lacks of transpose function!!!
#ifndef MPI_VERSION
    cusp::transpose( plus, plusT);
    cusp::transpose( minus, minusT);     
#endif
//     copy into h vectors
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( yp[2].begin(), yp[2].end(), hp.begin() + i*g2d.size());
        thrust::copy( ym[2].begin(), ym[2].end(), hm.begin() + i*g2d.size());        
    }
    dg::blas1::scal( hm, -1.);
    dg::blas1::axpby(  1., hp, +1., hm, hz);    //
 
}
template<class M, class container>
void DZ<M,container>::set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    cView left( global.cbegin(), global.cbegin() + size);
    cView right( global.cbegin()+(g_.Nz()-1)*size, global.cbegin() + g_.Nz()*size);
    View leftView( left_.begin(), left_.end());
    View rightView( right_.begin(), right_.end());
    cusp::copy( left, leftView);
    cusp::copy( right, rightView);
    dg::blas1::scal( left_, scal_left);
    dg::blas1::scal( right_, scal_right);
    bcz_ = bcz;
}

template<class M, class container>
inline void DZ<M,container>::operator()( const container& f, container& dzf) { return centered(f, dzf);}

template<class M, class container>
void DZ<M,container>::centered( const container& f, container& dzf)
{
    //direct discretisation
    assert( &f != &dzf);
    einsPlus( f, tempP);
    einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, hz, dzf);
    
    ////adjoint discretisation
/*    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( w3d, f, dzf);
    dg::blas1::pointwiseDivide( dzf, hz, dzf);
    dg::blas1::pointwiseDivide( dzf, invB, dzf);

    einsPlusT( dzf, tempP);
    einsMinusT( dzf, tempM);
    dg::blas1::axpby( 1., tempM, -1., tempP);
    dg::blas1::pointwiseDot( v3d, tempP, dzf);
    dg::blas1::pointwiseDot( dzf, invB, dzf);  */  

}

template<class M, class container>
void DZ<M,container>::centeredT( const container& f, container& dzf)
{               
//     //adjoint discretisation
        assert( &f != &dzf);    
        dg::blas1::pointwiseDot( w3d, f, dzf);

        dg::blas1::pointwiseDivide( dzf, hz, dzf);
        einsPlusT( dzf, tempP);
        einsMinusT( dzf, tempM);
        dg::blas1::axpby( 1., tempM, -1., tempP);        
        dg::blas1::pointwiseDot( v3d, tempP, dzf); 

//       dg::blas1::pointwiseDot( v3d, tempP,tempP); //make it symmetric
        //stegmeir weights
//         dg::blas1::pointwiseDot( hzh, f, dzf);
//         dg::blas1::pointwiseDot( invB, dzf, dzf);
//         dg::blas1::pointwiseDot( w2d, dzf, dzf);
//         dg::blas1::pointwiseDivide( dzf, hz, dzf);
//         einsPlusT( dzf, tempP);
//         einsMinusT( dzf, tempM);
//         dg::blas1::axpby( 1., tempM, -1., tempP);        
//         dg::blas1::pointwiseDot( v3d, tempP, dzf);
//         dg::blas1::scal(dzf,0.5);
//         dg::blas1::pointwiseDivide( tempP,hzh,  dzf);
//         dg::blas1::pointwiseDivide(  dzf,invB, dzf);
//         dg::blas1::pointwiseDivide( dzf,w2d,  dzf);  
}

template<class M, class container>
void DZ<M,container>::centeredTD( const container& f, container& dzf)
{       
//     Direct discretisation
       assert( &f != &dzf);    
        dg::blas1::pointwiseDot( f, invB, dzf);
        einsPlus( dzf, tempP);
        einsMinus( dzf, tempM);
        dg::blas1::axpby( 1., tempP, -1., tempM);
        dg::blas1::pointwiseDivide( tempM, hz, dzf);        
        dg::blas1::pointwiseDivide( dzf, invB, dzf);

}
template<class M, class container>
void DZ<M,container>::forward( const container& f, container& dzf)
{
    //direct
    assert( &f != &dzf);
    einsPlus( f, tempP);
    dg::blas1::axpby( 1., tempP, -1., f, tempP);
    dg::blas1::pointwiseDivide( tempP, hp, dzf);
    //adjoint discretisation
//     assert( &f != &dzf);    
//     dg::blas1::pointwiseDot( w3d, f, dzf);
//     dg::blas1::pointwiseDivide( dzf, hm, dzf);
//     dg::blas1::pointwiseDivide( dzf, invB, dzf);
//     einsMinusT( dzf, tempP);
//     dg::blas1::axpby( 1., tempP,-1.,dzf,dzf);
//     dg::blas1::pointwiseDot( v3d, dzf, dzf);
//     dg::blas1::pointwiseDot( dzf, invB, dzf);
}
template<class M, class container>
void DZ<M,container>::forwardT( const container& f, container& dzf)
{    
    //adjoint discretisation
    assert( &f != &dzf);
    dg::blas1::pointwiseDot( w3d, f, dzf);   
    dg::blas1::pointwiseDivide( dzf, hp, dzf);
    einsPlusT( dzf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dzf, dzf);
    dg::blas1::pointwiseDot( v3d, dzf, dzf);
    
}
template<class M, class container>
void DZ<M,container>::forwardTD( const container& f, container& dzf)
{
    //direct discretisation
    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( f, invB, dzf);
    einsMinus( dzf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dzf, dzf);
    dg::blas1::pointwiseDivide( dzf, hm, dzf);        
    dg::blas1::pointwiseDivide( dzf, invB, dzf);


}
template<class M, class container>
void DZ<M,container>::backward( const container& f, container& dzf)
{
    //direct
    assert( &f != &dzf);
    einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempM, -1., f, tempM);
    dg::blas1::pointwiseDivide( tempM, hm, dzf);
    
    //adjoint discretisation
//     assert( &f != &dzf);    
//     dg::blas1::pointwiseDot( w3d, f, dzf);
//     dg::blas1::pointwiseDivide( dzf, hp, dzf);
//     dg::blas1::pointwiseDivide( dzf, invB, dzf);
//     einsPlusT( dzf, tempM);
//     dg::blas1::axpby( 1., tempM, -1.,dzf,dzf);
//     dg::blas1::pointwiseDot( v3d,dzf, dzf);
//     dg::blas1::pointwiseDot( dzf, invB, dzf);
}
template<class M, class container>
void DZ<M,container>::backwardT( const container& f, container& dzf)
{    
    //adjoint discretisation
    assert( &f != &dzf);
    dg::blas1::pointwiseDot( w3d, f, dzf);
    dg::blas1::pointwiseDivide( dzf, hm, dzf);
    einsMinusT( dzf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dzf, dzf);
    dg::blas1::pointwiseDot( v3d, dzf, dzf);   
}

template<class M, class container>
void DZ<M,container>::backwardTD( const container& f, container& dzf)
{
    //direct
    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( f, invB, dzf);
    einsPlus( dzf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dzf, dzf);
    dg::blas1::pointwiseDivide( dzf, hp, dzf);        
    dg::blas1::pointwiseDivide( dzf, invB, dzf);
}

template< class M, class container >
void DZ<M,container>::symv( const container& f, container& dzTdzf)
{
// normed
//     centered( f, tempP);
//     centeredT( tempP, dzTdzf);
    forward( f, tempP);
    forwardT( tempP, dzTdzf);
    backward( f, tempM);
    backwardT( tempM, temp0);
    dg::blas1::axpby(0.5,temp0,0.5,dzTdzf,dzTdzf);
//     add jump term 
    #ifndef MPI_VERSION

    dg::blas2::symv( jump, f, temp0);
    dg::blas1::pointwiseDot( v3d, temp0,temp0); //make it symmetric
    dg::blas1::axpby(-1., temp0, 1., dzTdzf);
    #endif

//     //not normed
//     centered( f, tempP);
//     centeredT( tempP, dzTdzf);
// //     forward( f, tempP);
// //     forwardT( tempP, dzTdzf);
// //     backward( f, tempM);
// //     backwardT( tempM, temp0);
// //     dg::blas1::axpby(0.5,temp0,0.5,dzTdzf,dzTdzf);
//     dg::blas1::pointwiseDot( w3d, dzTdzf, dzTdzf); //make it symmetric
//     
//     #ifndef MPI_VERSION
//         
//      dg::blas2::symv( jump, f, temp0);
//      dg::blas1::axpby(-1., temp0, 1., dzTdzf,dzTdzf);
//     #endif

}
template< class M, class container >
void DZ<M,container>::dzz( const container& f, container& dzzf)
{
    assert( &f != &dzzf);
    einsPlus( f, tempP);
    einsMinus( f, tempM);
    dg::blas1::pointwiseDivide( tempP, hp, tempP);
    dg::blas1::pointwiseDivide( tempP, hz, tempP);
    dg::blas1::pointwiseDivide( f, hp, temp0);
    dg::blas1::pointwiseDivide( temp0, hm, temp0);
    dg::blas1::pointwiseDivide( tempM, hm, tempM);
    dg::blas1::pointwiseDivide( tempM, hz, tempM);
    dg::blas1::axpby(  2., tempP, +2., tempM); //fp+fm
    dg::blas1::axpby( -2., temp0, +1., tempM, dzzf); 
}



//enables the use of the dg::blas2::symv function 
template< class M, class V>
struct MatrixTraits< DZ<M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond


}//namespace dg

