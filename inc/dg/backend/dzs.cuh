

#pragma once
#include <cusp/transpose.h>
#include "grid.h"
#include "../blas.h"
#include "ell_interpolation.cuh"
// #include "interpolation.cuh"
#include "typedefs.cuh"
#include "functions.h"
#include "derivatives.cuh"
#include "../functors.h"
#include "../nullstelle.h"
#include "../runge_kutta.h"
///@cond
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
    double operator()( double deltas)
    {
        try{
            dg::integrateRK4( field_, coords_, coordsp_, deltas, eps_);
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
void boxintegrator( Field& field, const Grid& grid, const thrust::host_vector<double>& coords0, thrust::host_vector<double>& coords1, double& s1, double eps, dg::bc globalbcz)
{
    dg::integrateRK4( field, coords0, coords1, s1, eps); //+ integration
    if (    !(coords1[0] >= grid.x0() && coords1[0] <= grid.x1())
         || !(coords1[1] >= grid.y0() && coords1[1] <= grid.y1()))
    {
        if( globalbcz == dg::DIR)
        {
            BoxIntegrator<Field, Grid> boxy( field, grid, eps);
            boxy.set_coords( coords0); //nimm alte koordinaten
            if( s1 > 0)
            {
                double dsMin = 0, dsMax =s1;
                dg::bisection1d( boxy, dsMin, dsMax,eps); //suche 0 stelle 
                s1 = (dsMin+dsMax)/2.;
                dg::integrateRK4( field, coords0, coords1, dsMax, eps); //integriere bis über 0 stelle raus damit unten Wert neu gesetzt wird
            }
            else
            {
                double dsMin = s1, dsMax = 0;
                dg::bisection1d( boxy, dsMin, dsMax,eps);
                s1 = (dsMin+dsMax)/2.;
                dg::integrateRK4( field, coords0, coords1, dsMin, eps);
            }
            if (coords1[0] <= grid.x0()) { coords1[0]=grid.x0();}
            if (coords1[0] >= grid.x1()) { coords1[0]=grid.x1();}
            if (coords1[1] <= grid.y0()) { coords1[1]=grid.y0();}
            if (coords1[1] >= grid.y1()) { coords1[1]=grid.y1();}
        }
        else if (globalbcz == dg::NEU )
        {
             coords1[0] = coords0[0]; 
             coords1[1] = coords0[1];  
             coords1[2] = coords0[2]; //added
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
    DZ(Field field, const dg::Grid3d<double>& grid, double hs, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR);
    /**
    * @brief Apply the derivative on a 3d vector
    *
    * @param f The vector to derive
    * @param dzf contains result on output (write only)
    */
    void operator()( const container& f, container& dzf);

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
    void einsPlus( const container& n, container& npe);
    void einsMinus( const container& n, container& nme);
    void einsPlusT( const container& n, container& npe);
    void einsMinusT( const container& n, container& nme);
    void centeredT( const container& f, container& dzf);
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
    Matrix jump;
    container rp,zp,phip,rm,zm,phim;
    container hz, hp,hm;
    container tempP, temp0, tempM, ghostM, ghostP,dzfp,dzfm;
    container hz_plane, hp_plane, hm_plane;
    dg::Grid3d<double> g_;
    dg::bc bcz_;
    container left_, right_;
    container limiter;
    container w3d, v3d;
    double dsc,dsp,dsm;
};

////////////////////////////////////DEFINITIONS////////////////////////////////////////
template<class M, class container>
template <class Field, class Limiter>
DZ<M,container>::DZ(Field field, const dg::Grid3d<double>& grid, double deltas, double eps, Limiter limit, dg::bc globalbcz):
        jump( dg::create::jump2d( grid, grid.bcx(), grid.bcy(), not_normed)),
        rp( dg::evaluate( dg::zero, grid)), zp( rp), phip( rp), rm(rp),zm(rp),phim(phip),
        hz(rp), hp( rp), hm( rp), tempP( rp), temp0( rp), tempM( rp), dzfp( rp),dzfm( rp),
        g_(grid), bcz_(grid.bcz()), w3d( dg::create::weights( grid)), v3d( dg::create::inv_weights( grid)),dsc(deltas),dsp(deltas*0.5),dsm(deltas*0.5)
{

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

        double s1 = deltas;
        boxintegrator( field, g2d, coords, coordsP,  s1, eps, globalbcz);
        dsp = s1;
        s1 = -deltas;
        boxintegrator( field, g2d, coords, coordsM,  s1, eps, globalbcz);
        dsm = s1;
        dsc = dsp-dsm;
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( yp[0].begin(), yp[0].end(), rp.begin() + i*g2d.size()); //rp
        thrust::copy( ym[0].begin(), ym[0].end(), rm.begin() + i*g2d.size()); //rm
        thrust::copy( yp[1].begin(), yp[1].end(), zp.begin() + i*g2d.size()); //zp
        thrust::copy( ym[1].begin(), ym[1].end(), zm.begin() + i*g2d.size()); //zm
        thrust::copy( yp[2].begin(), yp[2].end(), phip.begin() + i*g2d.size()); //phip
        thrust::copy( ym[2].begin(), ym[2].end(), phim.begin() + i*g2d.size()); //phim
    }
    container phibias( dg::evaluate(dg::coo3,grid));
    dg::blas1::axpby(1.0,phibias,1.0,phip);
    dg::blas1::transform(phip, phip, dg::MOD<>(2.*M_PI)); 

    dg::blas1::axpby(1.0,phibias,1.0,phim); 
    dg::blas1::transform(phim, phim, dg::MOD<>(2.*M_PI));
    //3D interpolation of in + and -
//     cusp::coo_matrix<int, double, cusp::host_memory> plusH, minusH, plusHT, minusHT;
//     plusH  = dg::create::interpolation( rp, zp, phip, grid, globalbcz); 
//     minusH = dg::create::interpolation( rm, zm, phim, grid, globalbcz); 
    cusp::ell_matrix<int, double, cusp::host_memory> plusH, minusH, plusHT, minusHT;
    plusH   = dg::create::ell_interpolation( rp, zp, phip, grid);
    minusH  = dg::create::ell_interpolation( rm, zm, phim, grid);


    //Transpose matrices for adjoint operator
    cusp::transpose( plusH, plusHT);
    cusp::transpose( minusH, minusHT);    
    plus = plusH, minus = minusH, plusT = plusHT, minusT = minusHT; 
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
    assert( &f != &dzf);
    einsPlus( f, tempP);
    einsMinus( f, tempM);
    dg::blas1::axpby( 1., tempP, -1., tempM,dzf);
    dg::blas1::scal(dzf,1./dsc);
    
}

template<class M, class container>
void DZ<M,container>::centeredT( const container& f, container& dzf)
{
    assert( &f != &dzf);    
    dg::blas1::pointwiseDot( w3d, f, dzf);
//     dg::blas1::pointwiseDivide( dzf, hz, dzf);
    einsPlusT( dzf, tempP);
    einsMinusT( dzf, tempM);
    dg::blas1::axpby( 1., tempM, -1., tempP);
    dg::blas1::pointwiseDot( v3d, tempP, dzf);
    dg::blas1::scal(dzf,1./dsc);
}
template< class M, class container >
void DZ<M,container>::symv( const container& f, container& dzTdzf)
{
    this->operator()( f, tempP);
    centeredT( tempP, dzTdzf);
//     forward( f, tempP);
//     forwardT( tempP, dzTdzf);
//     dg::blas1::pointwiseDot( w3d, dzTdzf, dzTdzf); //make it symmetric
//     dg::blas2::symv( jump, f, tempP);
//     dg::blas1::axpby( 1., tempP, 1., dzTdzf);
    //add jump term (unstable without it)
//     einsPlus( f, tempP); 
//     dg::blas1::axpby( -1., tempP, 2., f, tempP);
//     einsPlusT( f, tempM); 
//     dg::blas1::axpby( -1., tempM, 1., tempP);
//     dg::blas1::axpby( 0.5, tempP, 1., dzTdzf);
//     einsMinusT( f, tempP); 
//     dg::blas1::axpby( -1., tempP, 2., f, tempP);
//     einsMinus( f, tempM); 
//     dg::blas1::axpby( -1., tempM, 1., tempP);
//     dg::blas1::axpby( 0.5, tempP, 1., dzTdzf);
}

template< class M, class container>
void DZ<M, container>::einsPlus( const container& f, container& fpe)
{

    dg::blas2::symv( plus, f, fpe);

}
template< class M, class container>
void DZ<M, container>::einsMinus( const container& f, container& fme)
{
    //note that thrust functions don't work on views
   dg::blas2::symv( minus, f, fme );

}
template< class M, class container>
void DZ<M, container>::einsMinusT( const container& f, container& fpe)
{
    dg::blas2::symv( minusT, f, fpe );
}
template< class M, class container>
void DZ<M, container>::einsPlusT( const container& f, container& fme)
{
    dg::blas2::symv( plusT, f, fme );
}

//enables the use of the dg::blas2::symv function 
template< class M, class V>
struct MatrixTraits< DZ<M, V> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};


}//namespace dg
///@endcond



