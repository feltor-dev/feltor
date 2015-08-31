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
* @ingroup ds
* @tparam FieldAligned Engine class for interpolation
* @tparam Matrix The matrix class of the jump matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class FA, class Matrix, class container=thrust::device_vector<double> >
struct DS
{
    typedef FA FieldAligned;

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
    DS(const FA& field, InvB invB, const Grid& grid, dg::norm no=dg::normed, dg::direction dir = dg::centered);

    /**
    * @brief Apply the derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forward( const container& f, container& dsf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backward( const container& f, container& dsf);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centered( const container& f, container& dsf);

    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardT( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardT( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredTD( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * forward derivative \f$ \frac{1}{h_z^+}(f_{i+1} - f_{i})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void forwardTD( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * backward derivative \f$ \frac{1}{2h_z^-}(f_{i} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void backwardTD( const container& f, container& dsf);
    /**
    * @brief Apply the negative adjoint derivative on a 3d vector with the direct method
    *
    * centered derivative \f$ \frac{1}{2h_z}(f_{i+1} - f_{i-1})\f$
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    void centeredT( const container& f, container& dsf);
    /**
    * @brief is the negative transposed of centered
    *
    * redirects to centered
    * @param f The vector to derive
    * @param dsf contains result on output (write only)
    */
    
    void operator()( const container& f, container& dsf);


    /**
     * @brief Compute the second derivative using finite differences
     *
     * discretizes \f$ \nabla_\parallel\cdot \nabla_\parallel\f$
     * @param f input function
     * @param dssf output (write-only)
     */
    void dss( const container& f, container& dssf);

    /**
     * @brief Discretizes the parallel Laplacian as a symmetric matrix
     *
     * forward followed by forwardT and adding jump terms
     * @param f The vector to derive
     * @param dsTdsf contains result on output (write only)
     */
    void symv( const container& f, container& dsTdsf);

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

    const FA& fieldaligned() const{return f_;}
    private:
    FA f_;
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
DS<FA, M,container>::DS(const FA& field, Field inverseB, const Grid& grid, dg::norm no, dg::direction dir):
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
inline void DS<F,M,container>::operator()( const container& f, container& dsf) { 
    if( dir_ == dg::centered)
        return centered( f, dsf);
    else if( dir_ == dg::forward)
        return forward( f, dsf);
    else
        return backward( f, dsf);
}


template<class F, class M, class container>
void DS<F,M,container>::centered( const container& f, container& dsf)
{
    //direct discretisation
    assert( &f != &dsf);
    f_.einsPlus( f, tempP);
    f_.einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), dsf);
    
    ////adjoint discretisation
/*    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( w3d, f, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hz(), dsf);
    dg::blas1::pointwiseDivide( dsf, invB, dsf);

    einsPlusT( dsf, tempP);
    einsMinusT( dsf, tempM);
    dg::blas1::axpby( 1., tempM, -1., tempP);
    dg::blas1::pointwiseDot( v3d, tempP, dsf);
    dg::blas1::pointwiseDot( dsf, invB, dsf);  */  

}

template<class F, class M, class container>
void DS<F,M,container>::centeredT( const container& f, container& dsf)
{               
//     //adjoint discretisation
        assert( &f != &dsf);    
        dg::blas1::pointwiseDot( w3d, f, dsf);

        dg::blas1::pointwiseDivide( dsf, f_.hz(), dsf);
        f_.einsPlusT( dsf, tempP);
        f_.einsMinusT( dsf, tempM);
        dg::blas1::axpby( 1., tempM, -1., tempP);        
        dg::blas1::pointwiseDot( v3d, tempP, dsf); 

//       dg::blas1::pointwiseDot( v3d, tempP,tempP); //make it symmetric
        //stegmeir weights
//         dg::blas1::pointwiseDot( f_.hz()h, f, dsf);
//         dg::blas1::pointwiseDot( invB, dsf, dsf);
//         dg::blas1::pointwiseDot( w2d, dsf, dsf);
//         dg::blas1::pointwiseDivide( dsf, f_.hz(), dsf);
//         einsPlusT( dsf, tempP);
//         einsMinusT( dsf, tempM);
//         dg::blas1::axpby( 1., tempM, -1., tempP);        
//         dg::blas1::pointwiseDot( v3d, tempP, dsf);
//         dg::blas1::scal(dsf,0.5);
//         dg::blas1::pointwiseDivide( tempP,f_.hz()h,  dsf);
//         dg::blas1::pointwiseDivide(  dsf,invB, dsf);
//         dg::blas1::pointwiseDivide( dsf,w2d,  dsf);  
}

template<class F, class M, class container>
void DS<F,M,container>::centeredTD( const container& f, container& dsf)
{       
//     Direct discretisation
       assert( &f != &dsf);    
        dg::blas1::pointwiseDot( f, invB, dsf);
        f_.einsPlus( dsf, tempP);
        f_.einsMinus( dsf, tempM);
        dg::blas1::axpby( 1., tempP, -1., tempM);
        dg::blas1::pointwiseDivide( tempM, f_.hz(), dsf);        
        dg::blas1::pointwiseDivide( dsf, invB, dsf);

}
template<class F, class M, class container>
void DS<F,M,container>::forward( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);
    f_.einsPlus( f, tempP);
    dg::blas1::axpby( 1., tempP, -1., f, tempP);
    dg::blas1::pointwiseDivide( tempP, f_.hp(), dsf);
    //adjoint discretisation
//     assert( &f != &dsf);    
//     dg::blas1::pointwiseDot( w3d, f, dsf);
//     dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);
//     dg::blas1::pointwiseDivide( dsf, invB, dsf);
//     einsMinusT( dsf, tempP);
//     dg::blas1::axpby( 1., tempP,-1.,dsf,dsf);
//     dg::blas1::pointwiseDot( v3d, dsf, dsf);
//     dg::blas1::pointwiseDot( dsf, invB, dsf);
}
template<class F, class M, class container>
void DS<F,M,container>::forwardT( const container& f, container& dsf)
{    
    //adjoint discretisation
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( w3d, f, dsf);   
    dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);
    f_.einsPlusT( dsf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dsf, dsf);
    dg::blas1::pointwiseDot( v3d, dsf, dsf);
    
}
template<class F, class M, class container>
void DS<F,M,container>::forwardTD( const container& f, container& dsf)
{
    //direct discretisation
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_.einsMinus( dsf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dsf, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);


}
template<class F, class M, class container>
void DS<F,M,container>::backward( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);
    f_.einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempM, -1., f, tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hm(), dsf);
    
    //adjoint discretisation
//     assert( &f != &dsf);    
//     dg::blas1::pointwiseDot( w3d, f, dsf);
//     dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);
//     dg::blas1::pointwiseDivide( dsf, invB, dsf);
//     einsPlusT( dsf, tempM);
//     dg::blas1::axpby( 1., tempM, -1.,dsf,dsf);
//     dg::blas1::pointwiseDot( v3d,dsf, dsf);
//     dg::blas1::pointwiseDot( dsf, invB, dsf);
}
template<class F, class M, class container>
void DS<F,M,container>::backwardT( const container& f, container& dsf)
{    
    //adjoint discretisation
    assert( &f != &dsf);
    dg::blas1::pointwiseDot( w3d, f, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hm(), dsf);
    f_.einsMinusT( dsf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dsf, dsf);
    dg::blas1::pointwiseDot( v3d, dsf, dsf);   
}

template<class F, class M, class container>
void DS<F,M,container>::backwardTD( const container& f, container& dsf)
{
    //direct
    assert( &f != &dsf);    
    dg::blas1::pointwiseDot( f, invB, dsf);
    f_.einsPlus( dsf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dsf, dsf);
    dg::blas1::pointwiseDivide( dsf, f_.hp(), dsf);        
    dg::blas1::pointwiseDivide( dsf, invB, dsf);
}

template< class F, class M, class container >
void DS<F,M,container>::symv( const container& f, container& dsTdsf)
{
    if(dir_ == dg::centered)
    {
        centered( f, tempP);
        centeredT( tempP, dsTdsf);
    }
    else 
    {
        forward( f, tempP);
        forwardT( tempP, dsTdsf);
        backward( f, tempM);
        backwardT( tempM, temp0);
        dg::blas1::axpby(0.5,temp0,0.5,dsTdsf,dsTdsf);
    }
//     add jump term 

    dg::blas2::symv( jumpX, f, temp0);
    dg::blas1::pointwiseDivide( temp0, R_, temp0); //there is an R in the weights
    dg::blas1::axpby(-1., temp0, 1., dsTdsf, dsTdsf);
    dg::blas2::symv( jumpY, f, temp0);
    dg::blas1::pointwiseDivide( temp0, R_, temp0);
    dg::blas1::axpby(-1., temp0, 1., dsTdsf, dsTdsf);
    if( no_ == not_normed)
    {
        dg::blas1::pointwiseDot( w3d, dsTdsf, dsTdsf); //make it symmetric
    }

}

template< class F, class M, class container >
void DS<F,M,container>::dss( const container& f, container& dssf)
{
    assert( &f != &dssf);
    f_.einsPlus( f, tempP);
    f_.einsMinus( f, tempM);
    dg::blas1::pointwiseDivide( tempP, f_.hp(), tempP);
    dg::blas1::pointwiseDivide( tempP, f_.hz(), tempP);
    dg::blas1::pointwiseDivide( f, f_.hp(), temp0);
    dg::blas1::pointwiseDivide( temp0, f_.hm(), temp0);
    dg::blas1::pointwiseDivide( tempM, f_.hm(), tempM);
    dg::blas1::pointwiseDivide( tempM, f_.hz(), tempM);
    dg::blas1::axpby(  2., tempP, +2., tempM); //fp+fm
    dg::blas1::axpby( -2., temp0, +1., tempM, dssf); 
}


//enables the use of the dg::blas2::symv function 
template< class F, class M, class V>
struct MatrixTraits< DS<F,M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
///@endcond

typedef dg::DS<dg::FieldAligned<dg::IDMatrix, dg::DVec>, dg::DMatrix, dg::DVec> DDS;
typedef dg::DS<dg::FieldAligned<dg::IHMatrix, dg::HVec>, dg::HMatrix, dg::HVec> HDS;
#ifdef MPI_VERSION
typedef dg::DS< dg::MPI_FieldAligned<dg::IDMatrix, dg::BijectiveComm< dg::IDVec, dg::DVec >,  dg::DVec>, dg::MDMatrix, dg::MDVec > MDDS;
typedef dg::DS< dg::MPI_FieldAligned<dg::IHMatrix, dg::BijectiveComm< dg::IHVec, dg::HVec >,  dg::HVec>, dg::MHMatrix, dg::MHVec > MHDS;
#endif //MPI_VERSION


}//namespace dg

