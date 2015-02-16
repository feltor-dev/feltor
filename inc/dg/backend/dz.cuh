

#pragma once
#include <cusp/transpose.h>
#include "grid.h"
#include "../blas.h"
#include "interpolation.cuh"
#include "typedefs.cuh"
#include "functions.h"
#include "derivatives.cuh"
#include "../functors.h"
#include "../nullstelle.h"
#include "../runge_kutta.h"
namespace dg{
struct DefaultLimiter
{
    double operator()(double x, double y)
    {
        return 1;
    }
};
struct NoLimiter
{
    double operator()(double x, double y)
    {
        return 0.;
    }
};

/**
 * @brief Integrate a field line 
 *
 * @tparam Field Must be usable in the integrateRK4 function
 * @tparam Grid must provide 2d boundaries x0(), x1(), y0(), and y1()
 */
template < class Field, class Grid>
struct BoxIntegrator
{
    BoxIntegrator( Field field, const Grid& g, double eps): field_(field), g_(g), coords_(3), coordsp_(3), eps_(eps) {}
    void set_coords( const thrust::host_vector<double>& coords){ coords_ = coords;}
    double operator()( double deltaPhi)
    {
        try{
            dg::integrateRK4( field_, coords_, coordsp_, deltaPhi, eps_);
        }
        catch( dg::NotANumber& exception) { return -1;}
        if (!(coordsp_[0] >= g_.x0() && coordsp_[0] <= g_.x1())) {
            return -1;
        }
        if (!(coordsp_[1] >= g_.y0() && coordsp_[1] <= g_.y1())) {
            return -1;
        }
        return +1;
    }
    private:
    Field field_;
    Grid g_;
    thrust::host_vector<double> coords_, coordsp_;
    double eps_;
};

/**
 * @brief Integrate one field line in a given box, Result is guaranteed to lie inside the box
 *
 * @tparam Field Must be usable in the integrateRK4 function
 * @tparam Grid must provide 2d boundaries x0(), x1(), y0(), and y1()
 * @param field The field to use
 * @param grid instance of the Grid class 
 * @param coords0 The initial condition
 * @param coords1 The resulting points (write only) guaranteed to lie inside the grid
 * @param phi1 The angle (read/write) contains maximum phi on input and resulting phi on output
 * @param eps error
 * @param globalbcz boundary condition  (DIR or NEU)
 */
template< class Field, class Grid>
void boxintegrator( Field& field, const Grid& grid, const thrust::host_vector<double>& coords0, thrust::host_vector<double>& coords1, double& phi1, double eps, dg::bc globalbcz)
{
    dg::integrateRK4( field, coords0, coords1, phi1, eps); //+ integration
    if (    !(coords1[0] >= grid.x0() && coords1[0] <= grid.x1())
         || !(coords1[1] >= grid.y0() && coords1[1] <= grid.y1()))
    {
        if( globalbcz == dg::DIR)
        {
            BoxIntegrator<Field, Grid> boxy( field, grid, eps);
            boxy.set_coords( coords0); //nimm alte koordinaten
            if( phi1 > 0)
            {
                double dPhiMin = 0, dPhiMax = phi1;
                dg::bisection1d( boxy, dPhiMin, dPhiMax,eps); //suche 0 stelle 
                phi1 = (dPhiMin+dPhiMax)/2.;
                dg::integrateRK4( field, coords0, coords1, dPhiMax, eps); //integriere bis über 0 stelle raus damit unten Wert neu gesetzt wird
            }
            else
            {
                double dPhiMin = phi1, dPhiMax = 0;
                dg::bisection1d( boxy, dPhiMin, dPhiMax,eps);
                phi1 = (dPhiMin+dPhiMax)/2.;
                dg::integrateRK4( field, coords0, coords1, dPhiMin, eps);
            }
            if (coords1[0] <= grid.x0()) { coords1[0]=grid.x0();}
            if (coords1[0] >= grid.x1()) { coords1[0]=grid.x1();}
            if (coords1[1] <= grid.y0()) { coords1[1]=grid.y0();}
            if (coords1[1] >= grid.y1()) { coords1[1]=grid.y1();}
        }
        else if (globalbcz == dg::NEU )
        {
             coords1[0] = coords0[0]; coords1[1] = coords0[1];  
        }
        else if (globalbcz == DIR_NEU )std::cerr << "DIR_NEU NOT IMPLEMENTED "<<std::endl;
        else if (globalbcz == NEU_DIR )std::cerr << "NEU_DIR NOT IMPLEMENTED "<<std::endl;
        else if (globalbcz == dg::PER )std::cerr << "PER NOT IMPLEMENTED "<<std::endl;
    }
}
////////////////////////////////////DZCLASS////////////////////////////////////////////
/**
* @brief Class for the evaluation of a parallel derivative
*
* @ingroup dz
* @tparam Matrix The matrix class of the interpolation matrix
* @tparam container The container-class to on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class Matrix = dg::DMatrix, class container=thrust::device_vector<double> >
struct DZ
{
    /**
    * @brief Construct from a field and a grid
    *
    * @tparam Field The Fieldlines to be integrated: Has to provide void operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) where the first index is R, the second Z and the last s (the length of the field line)
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
    is a limiter and 0 if there isn't
    * @param field The field to integrate
    * @param grid The grid on which to operate
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere)
    * @param globalbcz Choose NEU or DIR. Defines BC in parallel on box
    * @note If there is a limiter, the boundary condition is set by the bcz variable from the grid and can be changed by the set_boundaries function. If there is no limiter the boundary condition is periodic.
    */
    template <class Field, class Limiter>
    DZ(Field field, const dg::Grid3d<double>& grid, double hz, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void operator()( const container& f, container& dzf);
    void forward( const container& f, container& dzf);
    void backward( const container& f, container& dzf);

    //void dz2d( const container& f, container& dzf);
    //void dzz2d( const container& f, container& dzzf);
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
     * @param f input function
     * @param dzzf output (write-only)
     */
    void dzz( const container& f, container& dzzf);
    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * @tparam BinaryOp Binary Functor
     * @param f Functor to evaluate
     * @param plane The number of the plane to start
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp>
    container evaluate( BinaryOp f, unsigned plane=0);
    /**
     * @brief Evaluate a 2d functor and transform to all planes along the fieldlines
     *
     * Evaluates the given functor on a 2d plane and then follows fieldlines to
     * get the values in the 3rd dimension. Uses the grid given in the constructor.
     * The second functor is used to scale the values along the fieldlines.
     * The fieldlines are assumed to be periodic.
     * @tparam BinaryOp Binary Functor
     * @tparam UnaryOp Unary Functor
     * @param f Functor to evaluate in x-y
     * @param g Functor to evaluate in z
     * @param plane The number of the plane to start
     * @param rounds The number of rounds to follow a fieldline
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp, class UnaryOp>
    container evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds);
    template< class BinaryOp, class UnaryOp>
    container evaluateAvg( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds);
    void eins(const Matrix& interp, const container& n, container& npe);
    void einsPlus( const container& n, container& npe);
    void einsMinus( const container& n, container& nme);
    void einsPlusT( const container& n, container& npe);
    void einsMinusT( const container& n, container& nme);
    void centeredT( const container& f, container& dzf);
    void forwardT( const container& f, container& dzf);
    void backwardT( const container& f, container& dzf);

    void symv( const container& f, container& dzTdzf);
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
    typedef cusp::array1d_view< typename container::iterator> View;
    typedef cusp::array1d_view< typename container::const_iterator> cView;
    Matrix plus, minus, plusT, minusT; //interpolation matrices
//     Matrix jump;
    container hz, hp,hm, tempP, temp0, tempM, ghostM, ghostP;
    container hz_plane, hp_plane, hm_plane;
    dg::Grid3d<double> g_;
    dg::bc bcz_;
    container left_, right_;
    container limiter;
    container w3d, v3d;
    //container invB;
};

////////////////////////////////////DEFINITIONS////////////////////////////////////////
template<class M, class container>
template <class Field, class Limiter>
DZ<M,container>::DZ(Field field, const dg::Grid3d<double>& grid, double deltaPhi, double eps, Limiter limit, dg::bc globalbcz):
//         jump( dg::create::jump2d( grid, grid.bcx(), grid.bcy(), not_normed)),
        hz( dg::evaluate( dg::zero, grid)), hp( hz), hm( hz), tempP( hz), temp0( hz), tempM( hz), 
        g_(grid), bcz_(grid.bcz()), w3d( dg::create::weights( grid)), v3d( dg::create::inv_weights( grid))//, invB(dg::evaluate(field,grid))
{

    assert( deltaPhi == grid.hz() || grid.Nz() == 1);
    if( deltaPhi != grid.hz())
        std::cout << "Computing in 2D mode!\n";
    //Resize vectors to 2D grid size
    dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    unsigned size = g2d.size();
    limiter = dg::evaluate( limit, g2d);
    right_ = left_ = dg::evaluate( zero, g2d);
    hz_plane.resize( size); hp_plane.resize( size); hm_plane.resize( size);
    ghostM.resize( size); ghostP.resize( size);
    //Set starting points
    std::vector<dg::HVec> y( 3, dg::evaluate( dg::coo1, g2d)), yp(y), ym(y);
    y[1] = dg::evaluate( dg::coo2, g2d);
    y[2] = dg::evaluate( dg::zero, g2d);
    thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
    //integrate field lines for all points
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
    cusp::coo_matrix<int, double, cusp::host_memory> plusH, minusH, plusHT, minusHT;
    plusH  = dg::create::interpolation( yp[0], yp[1], g2d, globalbcz);
    minusH = dg::create::interpolation( ym[0], ym[1], g2d, globalbcz);
    cusp::transpose( plusH, plusHT);
    cusp::transpose( minusH, minusHT);
    plus = plusH, minus = minusH, plusT = plusHT, minusT = minusHT; 
    //copy into h vectors
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( yp[2].begin(), yp[2].end(), hp.begin() + i*g2d.size());
        thrust::copy( ym[2].begin(), ym[2].end(), hm.begin() + i*g2d.size());
    }
    dg::blas1::scal( hm, -1.);
    dg::blas1::axpby(  1., hp, +1., hm, hz);
    //
    dg::blas1::axpby(  1., (container)yp[2], 0, hp_plane);
    dg::blas1::axpby( -1., (container)ym[2], 0, hm_plane);
    dg::blas1::axpby(  1., hp_plane, +1., hm_plane, hz_plane);
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
void DZ<M,container>::operator()( const container& f, container& dzf)
{
    //direct discretisation
    assert( &f != &dzf);
    einsPlus( f, tempP);
    einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM);
    dg::blas1::pointwiseDivide( tempM, hz, dzf);
    ////adjoint discretisation
//     assert( &f != &dzf);    
//     dg::blas1::pointwiseDot( w3d, f, dzf);
//     dg::blas1::pointwiseDivide( dzf, hz, dzf);
//     dg::blas1::pointwiseDivide( dzf, invB, dzf);
// 
//     einsPlusT( dzf, tempP);
//     einsMinusT( dzf, tempM);
//     dg::blas1::axpby( 1., tempM, -1., tempP);
//     dg::blas1::pointwiseDot( v3d, tempP, dzf);
//     dg::blas1::pointwiseDot( dzf, invB, dzf);




}

template<class M, class container>
void DZ<M,container>::centeredT( const container& f, container& dzf)
{       
    //Direct discretisation
    //    assert( &f != &dzf);    
    //     dg::blas1::pointwiseDot( f, invB, dzf);
    //     einsPlus( dzf, tempP);
    //     einsMinus( dzf, tempM);
    //     dg::blas1::axpby( 1., tempP, -1., tempM);
    //     dg::blas1::pointwiseDivide( tempM, hz, dzf);        
    //     dg::blas1::pointwiseDivide( dzf, invB, dzf);
        
    //adjoint discretisation
        assert( &f != &dzf);    
        dg::blas1::pointwiseDot( w3d, f, dzf);
        dg::blas1::pointwiseDivide( dzf, hz, dzf);
        einsPlusT( dzf, tempP);
        einsMinusT( dzf, tempM);
        dg::blas1::axpby( 1., tempM, -1., tempP);
        dg::blas1::pointwiseDot( v3d, tempP, dzf);
}

template<class M, class container>
void DZ<M,container>::forward( const container& f, container& dzf)
{
    assert( &f != &dzf);
    einsPlus( f, tempP);
    dg::blas1::axpby( 1., tempP, -1., f, tempP);
    dg::blas1::pointwiseDivide( tempP, hp, dzf);
}
template<class M, class container>
void DZ<M,container>::forwardT( const container& f, container& dzf)
{
        //direct discretisation
//        assert( &f != &dzf);    
//     dg::blas1::pointwiseDot( f, invB, dzf);
//     einsMinus( dzf, tempM);
//     dg::blas1::axpby( -1., tempM, 1., dzf, dzf);
//     dg::blas1::pointwiseDivide( dzf, hm, dzf);        
//     dg::blas1::pointwiseDivide( dzf, invB, dzf);
    
    //adjoint discretisation
    assert( &f != &dzf);
    dg::blas1::pointwiseDot( w3d, f, dzf);
    dg::blas1::pointwiseDivide( dzf, hp, dzf);
    einsPlusT( dzf, tempP);
    dg::blas1::axpby( -1., tempP, 1., dzf, dzf);
    dg::blas1::pointwiseDot( v3d, dzf, dzf);

}
template<class M, class container>
void DZ<M,container>::backward( const container& f, container& dzf)
{
    assert( &f != &dzf);
    einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempM, -1., f, tempM);
    dg::blas1::pointwiseDivide( tempM, hm, dzf);
}
template<class M, class container>
void DZ<M,container>::backwardT( const container& f, container& dzf)
{
        //direct
//     assert( &f != &dzf);    
//     dg::blas1::pointwiseDot( f, invB, dzf);
//     einsPlus( dzf, tempP);
//     dg::blas1::axpby( -1., tempP, 1., dzf, dzf);
//     dg::blas1::pointwiseDivide( dzf, hp, dzf);        
//     dg::blas1::pointwiseDivide( dzf, invB, dzf);
    
    //adjoint discretisation
    assert( &f != &dzf);
    dg::blas1::pointwiseDot( w3d, f, dzf);
    dg::blas1::pointwiseDivide( dzf, hm, dzf);
    einsMinusT( dzf, tempM);
    dg::blas1::axpby( -1., tempM, 1., dzf, dzf);
    dg::blas1::pointwiseDot( v3d, dzf, dzf);

}
template< class M, class container >
void DZ<M,container>::symv( const container& f, container& dzTdzf)
{
    //this->operator()( f, tempP);
    //centeredT( tempP, dzTdzf);
    forward( f, tempP);
    forwardT( tempP, dzTdzf);
    dg::blas1::pointwiseDot( w3d, dzTdzf, dzTdzf); //make it symmetric
    //dg::blas2::symv( jump, f, tempP);
    //dg::blas1::axpby( 1., tempP, 1., dzTdzf);
    //add jump term (unstable without it)
    einsPlus( f, tempP); 
    dg::blas1::axpby( -1., tempP, 2., f, tempP);
    einsPlusT( f, tempM); 
    dg::blas1::axpby( -1., tempM, 1., tempP);
    dg::blas1::axpby( 0.5, tempP, 1., dzTdzf);
    einsMinusT( f, tempP); 
    dg::blas1::axpby( -1., tempP, 2., f, tempP);
    einsMinus( f, tempM); 
    dg::blas1::axpby( -1., tempM, 1., tempP);
    dg::blas1::axpby( 0.5, tempP, 1., dzTdzf);
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

/*
template<class M, class container>
void DZ<M,container>::dz2d( const container& f, container& dzf)
{
    assert( &f != &dzf);
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView fp( f.cbegin(), f.cend());
    cView fm( f.cbegin(), f.cend());

    cusp::multiply( plus, fp, ghostPV);
    cusp::multiply( minus, fm, ghostMV );
    dg::blas1::axpby( 1., ghostP, -1., ghostM);
    dg::blas1::pointwiseDivide( ghostM, hz_plane, dzf);
}
template< class M, class container >
void DZ<M,container>::dzz2d( const container& f, container& dzzf)
{
    assert( &f != &dzzf);

    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView fp( f.cbegin() , f.cend());
    cView fm( f.cbegin() , f.cend());

    cusp::multiply( plus, fp, ghostPV);
    cusp::multiply( minus, fm, ghostMV );
    dg::blas1::pointwiseDivide( ghostP, hp_plane, ghostP);
    dg::blas1::pointwiseDivide( ghostP, hz_plane, ghostP);
    dg::blas1::pointwiseDivide( f,      hp_plane, dzzf);
    dg::blas1::pointwiseDivide( dzzf,   hm_plane, dzzf);
    dg::blas1::pointwiseDivide( ghostM, hm_plane, ghostM);
    dg::blas1::pointwiseDivide( ghostM, hz_plane, ghostM);
    dg::blas1::axpby( 2., ghostP, +2., ghostM); //fp+fm
    dg::blas1::axpby( -2., dzzf, +1., ghostM, dzzf);
}
*/

template< class M, class container>
template< class BinaryOp>
container DZ<M,container>::evaluate( BinaryOp binary, unsigned p0)
{
    assert( p0 < g_.Nz() && g_.Nz() > 1);
    const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    container vec2d = dg::evaluate( binary, g2d);
    View g0( vec2d.begin(), vec2d.end());
    container vec3d( g_.size());
    View f0( vec3d.begin() + p0*g2d.size(), vec3d.begin() + (p0+1)*g2d.size());
    //copy 2d function into given plane and then follow fieldline in both directions
    cusp::copy( g0, f0);
    for( unsigned i0=p0+1; i0<g_.Nz(); i0++)
    {
        unsigned im = i0-1;
        View fm( vec3d.begin() + im*g2d.size(), vec3d.begin() + (im+1)*g2d.size());
        View f0( vec3d.begin() + i0*g2d.size(), vec3d.begin() + (i0+1)*g2d.size());
        cusp::multiply( minus, fm, f0 );
    }
    for( int i0=p0-1; i0>=0; i0--)
    {
        unsigned ip = i0+1;
        View fp( vec3d.begin() + ip*g2d.size(), vec3d.begin() + (ip+1)*g2d.size());
        View f0( vec3d.begin() + i0*g2d.size(), vec3d.begin() + (i0+1)*g2d.size());
        cusp::multiply( plus, fp, f0 );
    }
    return vec3d;
}

template< class M, class container>
template< class BinaryOp, class UnaryOp>
container DZ<M,container>::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds)
{
    assert( g_.Nz() > 1);
    container vec3d = evaluate( binary, p0);
    const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    //scal
    for( unsigned i=0; i<g_.Nz(); i++)
    {
        View f0( vec3d.begin() + i*g2d.size(), vec3d.begin() + (i+1)*g2d.size());
        cusp::blas::scal(f0, unary( g_.z0() + (double)(i+0.5)*g_.hz() ));
    }
    //make room for plus and minus continuation
    std::vector<container > vec4dP( rounds, vec3d);
    std::vector<container > vec4dM( rounds, vec3d);
    //now follow field lines back and forth
    for( unsigned k=1; k<rounds; k++)
    {
        for( unsigned i0=0; i0<g_.Nz(); i0++)
        {
        int im = i0==0?g_.Nz()-1:i0-1;
        int k0 = k;
        int km = i0==0?k-1:k;
        View fm( vec4dP[km].begin() + im*g2d.size(), vec4dP[km].begin() + (im+1)*g2d.size());
        View f0( vec4dP[k0].begin() + i0*g2d.size(), vec4dP[k0].begin() + (i0+1)*g2d.size());
        cusp::multiply( minus, fm, f0 );
        cusp::blas::scal( f0, unary( g_.z0() + (double)(k*g_.Nz()+i0+0.5)*g_.hz() ) );
        }
        for( int i0=g_.Nz()-1; i0>=0; i0--)
        {
        int ip = i0==g_.Nz()-1?0:i0+1;
        int k0 = k;
        int km = i0==g_.Nz()-1?k-1:k;
        View fp( vec4dM[km].begin() + ip*g2d.size(), vec4dM[km].begin() + (ip+1)*g2d.size());
        View f0( vec4dM[k0].begin() + i0*g2d.size(), vec4dM[k0].begin() + (i0+1)*g2d.size());
        cusp::multiply( plus, fp, f0 );
        cusp::blas::scal( f0, unary( g_.z0() - (double)(k*g_.Nz()-0.5-i0)*g_.hz() ) );
        }
    }
    //sum up results
    for( unsigned i=1; i<rounds; i++)
    {
        dg::blas1::axpby( 1., vec4dP[i], 1., vec3d);
        dg::blas1::axpby( 1., vec4dM[i], 1., vec3d);
    }
    return vec3d;
}

template< class M, class container>
template< class BinaryOp, class UnaryOp>
container DZ<M,container>::evaluateAvg( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds)
{
    assert( g_.Nz() > 1);
    container vec3d = evaluate( f, g, p0, rounds);
    container vec2d(g_.size()/g_.Nz());

    for (unsigned i = 0; i<g_.Nz(); i++)
    {
        container part( vec3d.begin() + i* (g_.size()/g_.Nz()), vec3d.begin()+(i+1)*(g_.size()/g_.Nz()));
        dg::blas1::axpby(1.0,part,1.0,vec2d);
    }
    dg::blas1::scal(vec2d,1./g_.Nz());
    return vec2d;
}

template< class M, class container>
void DZ<M, container>::eins(const M& m, const container& f, container& fpe)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();

    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fpe0( fpe.begin() + i0*size, fpe.begin() + (i0+1)*size);
        cusp::multiply( m, f0, fpe0);       
    }
}
template< class M, class container>
void DZ<M, container>::einsPlus( const container& f, container& fpe)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fP( fpe.begin() + i0*size, fpe.begin() + (i0+1)*size);
        cusp::multiply( plus, fp, fP);
        //make ghostcells i.e. modify fpe in the limiter region
        if( i0==g_.Nz()-1 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            {
                cusp::blas::axpby( rightV, f0, ghostPV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
            {
                thrust::transform( right_.begin(), right_.end(),  hp.begin(), ghostM.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostMV, f0, ghostPV, 1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
            cusp::blas::axpby( ghostPV, fP, ghostPV, 1., -1.);
            dg::blas1::pointwiseDot( limiter, ghostP, ghostP);
            cusp::blas::axpby(  ghostPV, fP, fP, 1.,1.);
        }
    }
}

template< class M, class container>
void DZ<M, container>::einsMinus( const container& f, container& fme)
{
    //note that thrust functions don't work on views
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fM( fme.begin() + i0*size, fme.begin() + (i0+1)*size);
        cusp::multiply( minus, fm, fM );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
            {
                cusp::blas::axpby( leftV,  f0, ghostMV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
            {
                thrust::transform( left_.begin(), left_.end(),  hm.begin(), ghostP.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostPV, f0, ghostMV, -1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fme
            cusp::blas::axpby( ghostMV, fM, ghostMV, 1., -1.);
            dg::blas1::pointwiseDot( limiter, ghostM, ghostM);
            cusp::blas::axpby( ghostMV, fM, fM, 1., 1.);

        }
    }
}
template< class M, class container>
void DZ<M, container>::einsMinusT( const container& f, container& fpe)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fP( fpe.begin() + i0*size, fpe.begin() + (i0+1)*size);
        cusp::multiply( minusT, fp, fP );
        //make ghostcells i.e. modify fpe in the limiter region
        if( i0==g_.Nz()-1 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            {
                cusp::blas::axpby( rightV, f0, ghostPV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
            {
                thrust::transform( right_.begin(), right_.end(),  hp.begin(), ghostM.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostMV, f0, ghostPV, 1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
            cusp::blas::axpby( ghostPV, fP, ghostPV, 1., -1.);
            dg::blas1::pointwiseDot( limiter, ghostP, ghostP);
            cusp::blas::axpby(  ghostPV, fP, fP, 1.,1.);
        }

    }
}
template< class M, class container>
void DZ<M, container>::einsPlusT( const container& f, container& fme)
{
    //note that thrust functions don't work on views
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fM( fme.begin() + i0*size, fme.begin() + (i0+1)*size);
        cusp::multiply( plusT, fm, fM );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
            {
                cusp::blas::axpby( leftV,  f0, ghostMV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
            {
                thrust::transform( left_.begin(), left_.end(),  hm.begin(), ghostP.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostPV, f0, ghostMV, -1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fme
            cusp::blas::axpby( ghostMV, fM, ghostMV, 1., -1.);
            dg::blas1::pointwiseDot( limiter, ghostM, ghostM);
            cusp::blas::axpby( ghostMV, fM, fM, 1., 1.);

        }
    }
}

//enables the use of the dg::blas2::symv function 
template< class M, class V>
struct MatrixTraits< DZ<M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};


}//namespace dg



