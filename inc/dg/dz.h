#pragma once

#include "blas.h"
#include "backend/derivatives.h"
#include "backend/fieldaligned.h"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "backend/mpi_fieldaligned.h"
#endif //MPI_VERSION

/*!@file 
 *
 * This file includes the appropriate headers for parallel derivatives
 */

namespace dg{

/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
cylindrical coordinates
* @ingroup dz
* @tparam FieldAligned Engine class for interpolation
* @tparam Matrix The matrix class of the jump matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class FieldAligned, class Matrix, class container=thrust::device_vector<double> >
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
    * @param deltaPhi Must either equal the f_.hz()() value of the grid or a fictive deltaPhi if the grid is 2D and Nz=1
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere, note that if bcz is periodic it doesn't matter if there is a limiter or not)
    * @param globalbcz Choose NEU or DIR. Defines BC in parallel on box
    * @note If there is a limiter, the boundary condition is set by the bcz variable from the grid and can be changed by the set_boundaries function. If there is no limiter the boundary condition is periodic.
    */
    template<class InvB, class Grid>
    DZ(const FieldAligned& field, InvB invB, const Grid& grid, dg::norm no=dg::normed, dg::direction dir = dg::centered);

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
     * @brief Compute the second derivative using finite differences
     *
     * discretizes \f$ \nabla_\parallel\cdot \nabla_\parallel\f$
     * @param f input function
     * @param dzzf output (write-only)
     */
    void dzz( const container& f, container& dzzf);

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
        f_.set_boundaries( bcz, left, right);
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
        f_.set_boundaries( bcz, left, right);
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
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
    {
        f_.set_boundaries( bcz, global, scal_left, scal_right);
    }

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
    FieldAligned f_;
    Matrix jumpX, jumpY;
    container tempP, temp0, tempM;
    container w3d, v3d;
    container invB;
    container R_;
    dg::norm no_;
    dg::direction dir_;
};

////////////////////////////////////DEFINITIONS////////////////////////////////////////
///@cond
template<class FA, class M, class container>
template <class Field, class Grid>
DZ<FA, M,container>::DZ(const FA& field, Field inverseB, const Grid& grid, dg::norm no, dg::direction dir):
        f_(field),
        jumpX( dg::create::jumpX( grid)),
        jumpY( dg::create::jumpY( grid)),
        tempP( dg::evaluate( dg::zero, grid)), temp0( tempP), tempM( tempP), 
        w3d( dg::create::weights( grid)), v3d( dg::create::inv_weights( grid)),
        invB(dg::evaluate(inverseB,grid)), R_(dg::evaluate(dg::coo1,grid)), 
        no_(no), dir_(dir)
{
    assert( grid.system() == dg::cylindrical);
}

template<class F, class M, class container>
inline void DZ<F,M,container>::operator()( const container& f, container& dzf) { 
    if( dir_ == dg::centered)
        return centered( f, dzf);
    else if( dir_ == dg::forward)
        return forward( f, dzf);
    else
        return backward( f, dzf);
}


template<class F, class M, class container>
void DZ<F,M,container>::centered( const container& f, container& dzf)
{
    //direct discretisation
    assert( &f != &dzf);
    f_.einsPlus( f, tempP);
    f_.einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), dzf);
    
    ////adjoint discretisation
/*    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( w3d, f, dzf);
    dg::blas1::pointwiseDivide( dzf, f_.hz(), dzf);
    dg::blas1::pointwiseDivide( dzf, invB, dzf);

    einsPlusT( dzf, tempP);
    einsMinusT( dzf, tempM);
    dg::blas1::axpby( 1., tempM, -1., tempP);
    dg::blas1::pointwiseDot( v3d, tempP, dzf);
    dg::blas1::pointwiseDot( dzf, invB, dzf);  */  

}

template<class F, class M, class container>
void DZ<F,M,container>::centeredT( const container& f, container& dzf)
{               
//     //adjoint discretisation
        assert( &f != &dzf);    
        dg::blas1::pointwiseDot( w3d, f, dzf);

        dg::blas1::pointwiseDivide( dzf, f_.hz(), dzf);
        f_.einsPlusT( dzf, tempP);
        f_.einsMinusT( dzf, tempM);
        dg::blas1::axpby( 1., tempM, -1., tempP);        
        dg::blas1::pointwiseDot( v3d, tempP, dzf); 

//       dg::blas1::pointwiseDot( v3d, tempP,tempP); //make it symmetric
        //stegmeir weights
//         dg::blas1::pointwiseDot( f_.hz()h, f, dzf);
//         dg::blas1::pointwiseDot( invB, dzf, dzf);
//         dg::blas1::pointwiseDot( w2d, dzf, dzf);
//         dg::blas1::pointwiseDivide( dzf, f_.hz(), dzf);
//         einsPlusT( dzf, tempP);
//         einsMinusT( dzf, tempM);
//         dg::blas1::axpby( 1., tempM, -1., tempP);        
//         dg::blas1::pointwiseDot( v3d, tempP, dzf);
//         dg::blas1::scal(dzf,0.5);
//         dg::blas1::pointwiseDivide( tempP,f_.hz()h,  dzf);
//         dg::blas1::pointwiseDivide(  dzf,invB, dzf);
//         dg::blas1::pointwiseDivide( dzf,w2d,  dzf);  
}

template<class F, class M, class container>
void DZ<F,M,container>::centeredTD( const container& f, container& dzf)
{       
//     Direct discretisation
       assert( &f != &dzf);    
        dg::blas1::pointwiseDot( f, invB, dzf);
        f_.einsPlus( dzf, tempP);
        f_.einsMinus( dzf, tempM);
        dg::blas1::axpby( 1., tempP, -1., tempM);
        dg::blas1::pointwiseDivide( tempM, f_.hz(), dzf);        
        dg::blas1::pointwiseDivide( dzf, invB, dzf);

}
template<class F, class M, class container>
void DZ<F,M,container>::forward( const container& f, container& dzf)
{
    //direct
    assert( &f != &dzf);
    f_.einsPlus( f, tempP);
    dg::blas1::axpby( 1., tempP, -1., f, tempP);
    dg::blas1::pointwiseDivide( tempP, f_.hp(), dzf);
    //adjoint discretisation
//     assert( &f != &dzf);    
//     dg::blas1::pointwiseDot( w3d, f, dzf);
//     dg::blas1::pointwiseDivide( dzf, f_.hm(), dzf);
//     dg::blas1::pointwiseDivide( dzf, invB, dzf);
//     einsMinusT( dzf, tempP);
//     dg::blas1::axpby( 1., tempP,-1.,dzf,dzf);
//     dg::blas1::pointwiseDot( v3d, dzf, dzf);
//     dg::blas1::pointwiseDot( dzf, invB, dzf);
}
template<class F, class M, class container>
void DZ<F,M,container>::forwardT( const container& f, container& dzf)
{    
    //adjoint discretisation
    assert( &f != &dzf);
    dg::blas1::pointwiseDot( w3d, f, dzf);   
    dg::blas1::pointwiseDivide( dzf, f_.hp(), dzf);
    f_.einsPlusT( dzf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dzf, dzf);
    dg::blas1::pointwiseDot( v3d, dzf, dzf);
    
}
template<class F, class M, class container>
void DZ<F,M,container>::forwardTD( const container& f, container& dzf)
{
    //direct discretisation
    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( f, invB, dzf);
    f_.einsMinus( dzf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dzf, dzf);
    dg::blas1::pointwiseDivide( dzf, f_.hm(), dzf);        
    dg::blas1::pointwiseDivide( dzf, invB, dzf);


}
template<class F, class M, class container>
void DZ<F,M,container>::backward( const container& f, container& dzf)
{
    //direct
    assert( &f != &dzf);
    f_.einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempM, -1., f, tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hm(), dzf);
    
    //adjoint discretisation
//     assert( &f != &dzf);    
//     dg::blas1::pointwiseDot( w3d, f, dzf);
//     dg::blas1::pointwiseDivide( dzf, f_.hp(), dzf);
//     dg::blas1::pointwiseDivide( dzf, invB, dzf);
//     einsPlusT( dzf, tempM);
//     dg::blas1::axpby( 1., tempM, -1.,dzf,dzf);
//     dg::blas1::pointwiseDot( v3d,dzf, dzf);
//     dg::blas1::pointwiseDot( dzf, invB, dzf);
}
template<class F, class M, class container>
void DZ<F,M,container>::backwardT( const container& f, container& dzf)
{    
    //adjoint discretisation
    assert( &f != &dzf);
    dg::blas1::pointwiseDot( w3d, f, dzf);
    dg::blas1::pointwiseDivide( dzf, f_.hm(), dzf);
    f_.einsMinusT( dzf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dzf, dzf);
    dg::blas1::pointwiseDot( v3d, dzf, dzf);   
}

template<class F, class M, class container>
void DZ<F,M,container>::backwardTD( const container& f, container& dzf)
{
    //direct
    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( f, invB, dzf);
    f_.einsPlus( dzf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dzf, dzf);
    dg::blas1::pointwiseDivide( dzf, f_.hp(), dzf);        
    dg::blas1::pointwiseDivide( dzf, invB, dzf);
}

template< class F, class M, class container >
void DZ<F,M,container>::symv( const container& f, container& dzTdzf)
{
    if(dir_ == dg::centered)
    {
        centered( f, tempP);
        centeredT( tempP, dzTdzf);
    }
    else 
    {
        forward( f, tempP);
        forwardT( tempP, dzTdzf);
        backward( f, tempM);
        backwardT( tempM, temp0);
        dg::blas1::axpby(0.5,temp0,0.5,dzTdzf,dzTdzf);
    }
//     add jump term 

    dg::blas2::symv( jumpX, f, temp0);
    dg::blas1::pointwiseDivide( temp0, R_, temp0); //there is an R in the weights
    dg::blas1::axpby(-1., temp0, 1., dzTdzf, dzTdzf);
    dg::blas2::symv( jumpY, f, temp0);
    dg::blas1::pointwiseDivide( temp0, R_, temp0);
    dg::blas1::axpby(-1., temp0, 1., dzTdzf, dzTdzf);
    if( no_ == not_normed)
    {
        dg::blas1::pointwiseDot( w3d, dzTdzf, dzTdzf); //make it symmetric
    }

}

template< class F, class M, class container >
void DZ<F,M,container>::dzz( const container& f, container& dzzf)
{
    assert( &f != &dzzf);
    f_.einsPlus( f, tempP);
    f_.einsMinus( f, tempM);
    dg::blas1::pointwiseDivide( tempP, f_.hp(), tempP);
    dg::blas1::pointwiseDivide( tempP, f_.hz(), tempP);
    dg::blas1::pointwiseDivide( f, f_.hp(), temp0);
    dg::blas1::pointwiseDivide( temp0, f_.hm(), temp0);
    dg::blas1::pointwiseDivide( tempM, f_.hm(), tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), tempM);
    dg::blas1::axpby(  2., tempP, +2., tempM); //fp+fm
    dg::blas1::axpby( -2., temp0, +1., tempM, dzzf); 
}


//enables the use of the dg::blas2::symv function 
template< class F, class M, class V>
struct MatrixTraits< DZ<F,M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond


}//namespace dg

