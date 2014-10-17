#pragma once

#include "grid.h"
#include "interpolation.cuh"
#include "typedefs.cuh"
#include "functions.h"
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

template < class Field>
struct BoxIntegrator
{
    BoxIntegrator( Field field, const Grid2d<double>& g, double eps): field_(field), g_(g), coords_(3), coordsp_(3), eps_(eps) {}
    void set_coords( const thrust::host_vector<double>& coords){ coords_ = coords;}
    double operator()( double deltaPhi)
    {
        try{
            dg::integrateRK4( field_, coords_, coordsp_,  deltaPhi, eps_);
        }catch( dg::NotANumber& exception) { return -1;}
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
    Grid2d<double> g_;
    thrust::host_vector<double> coords_, coordsp_;
    double eps_;

};

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
     * @tparam Field The Fieldlines to be integrated: Has to provide void  operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) where the first index is R, the second Z and the last s (the length of the field line)
     * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there 
     is a limiter and 0 if there isn't
     * @param field The field to integrate
     * @param grid The grid on which to operate
     * @param eps Desired accuracy of runge kutta
     * @param limit Instance of the limiter class (Default is a limiter everywhere)
     * @note If there is a limiter, the boundary condition is set by the bcz variable from the grid and can be changed by the set_boundaries function. If there is no limiter the boundary condition is periodic.
     */
    template <class Field, class Limiter>
    DZ(Field field, const dg::Grid3d<double>& grid, double eps = 1e-4, Limiter limit = DefaultLimiter()): 
        g_(grid), bcz_(grid.bcz())
    {
        dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
        unsigned size = g2d.size();
        limiter = dg::evaluate( limit, g2d);
        left_ = dg::evaluate( zero, g2d);
        right_ = left_;
        //Resize vectors to 2D grid size
        hz.resize( size);     hp.resize( size);     hm.resize( size);
        ghostM.resize( size); ghostP.resize( size);
        tempM.resize( size);  temp0.resize( size);  tempP.resize( size);
        std::vector<dg::HVec> y( 3, dg::evaluate( dg::coo1, g2d)), yp(y), ym(y); 
        //Set starting points
        y[1] = dg::evaluate( dg::coo2, g2d);
        y[2] = dg::evaluate( dg::zero, g2d);
        thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
        for( unsigned i=0; i<size; i++)
        {
            //assumes that perp boundary condition is constant on the entire 3d box surface
            coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i];
            dg::integrateRK4( field, coords, coordsP,  g_.hz(), eps);
            dg::integrateRK4( field, coords, coordsM,  -g_.hz(), eps);

            //if (    !(coordsP[0] >= g_.x0() && coordsP[0] <= g_.x1())
            //     || !(coordsP[1] >= g_.y0() && coordsP[1] <= g_.y1())) 
            //{
            //    //std::cout << "P: "<<coords[0] << " "<<coords[1]<<"\n";
            //    //std::cout << "P: "<<coordsP[0] << " "<<coordsP[1]<<"\n";
            //    BoxIntegrator<Field> boxy( field, g2d, eps);
            //    boxy.set_coords( coords);
            //    double dPhiMin = 0, dPhiMax = g_.hz();
            //    //std::cout << "B: "<<g_.x0() << " "<<g_.x1()<< " "<<g_.y0()<< " "<<g_.y1()<<std::endl;
            //    //std::cout << "B: "<<boxy( 0) << " "<<boxy(dPhiMax)<<std::endl;
            //    dg::bisection1d( boxy, dPhiMin, dPhiMax, eps);
            //    //std::cout << dPhiMin << std::endl;
            //    dg::integrateRK4( field, coords, coordsP, dPhiMin, eps);
            //    //std::cout << "P: "<<coordsP[0] << " "<<coordsP[1]<<"\n";
            //    //std::cout << std::endl;
            //}
            //if (    !(coordsM[0] >= g_.x0() && coordsM[0] <= g_.x1())
            //     || !(coordsM[1] >= g_.y0() && coordsM[1] <= g_.y1())) 
            //{
            //    //std::cout << "M: "<<coords[0] << " "<<coords[1]<<"\n";
            //    //std::cout << "M: "<<coordsM[0] << " "<<coordsM[1]<<"\n";
            //    BoxIntegrator<Field> boxy( field, g2d, eps);
            //    boxy.set_coords( coords);
            //    double dPhiMin = -g_.hz(), dPhiMax = 0;
            //    dg::bisection1d( boxy, dPhiMin, dPhiMax, eps);
            //    dg::integrateRK4( field, coords, coordsM, dPhiMax, eps);
            //    //std::cout << "M: "<<coordsM[0] << " "<<coordsM[1]<<"\n";
            //    //std::cout << std::endl;
            //}
            yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
            ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
        }
        //dg::integrateRK4( field, y, ym, -g_.hz(), eps);
        cut( y, yp, g2d);
        cut( y, ym, g2d);
        plus  = dg::create::interpolation( yp[0], yp[1], g2d);
        minus = dg::create::interpolation( ym[0], ym[1], g2d);
        dg::blas1::axpby(  1., (container)yp[2], 0, hp);
        dg::blas1::axpby( -1., (container)ym[2], 0, hm);
        dg::blas1::axpby(  1., hp, +1., hm, hz);
    }

    /**
     * @brief Apply the derivative on a 3d vector
     *
     * @param f The vector to derive
     * @param dzf contains result on output (write only)
     */
    void operator()( const container& f, container& dzf);

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
        left_  = dg::evaluate( dg::CONSTANT(left), g2d);
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
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
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
  private:
    typedef cusp::array1d_view< typename container::iterator> View;
    typedef cusp::array1d_view< typename container::const_iterator> cView;
    Matrix plus, minus; //interpolation matrices
    container hz, hp,hm, tempP, temp0, tempM, ghostM, ghostP;
    dg::Grid3d<double> g_;
    dg::bc bcz_;
    container left_, right_;
    container limiter;
    void cut( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp, dg::Grid2d<double>& g);
};

template<class M, class container>
void DZ<M,container>::operator()( const container& f, container& dzf)
{
    assert( &f != &dzf);
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View tempPV( tempP.begin(), tempP.end());
    View tempMV( tempM.begin(), tempM.end());

    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;
        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cusp::multiply( plus, fp, tempPV);
        cusp::multiply( minus, fm, tempMV );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
            //overwrite tempM
            cusp::copy( f0, ghostMV);
            if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
            {
                //dg::blas1::scal( ghostM, -1.);
                dg::blas1::axpby( 2., left_, -1, ghostM);
                //dg::blas1::transform( ghostM, ghostM, dg::PLUS<double>( 2.*left_));
            }
            if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
            {
                dg::blas1::pointwiseDot( left_, hm, ghostP);
                dg::blas1::axpby( -1, ghostP, 1., ghostM);
            }
            dg::blas1::axpby( 1., ghostM, -1., tempM, ghostM);
            dg::blas1::pointwiseDot( limiter, ghostM, ghostM);
            dg::blas1::axpby( 1., ghostM, 1., tempM);

        }
        else if( i0==g_.Nz()-1 && bcz_ != dg::PER)
        {
            //overwrite tempP
            cusp::copy( f0, ghostPV);
            if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            {
                //dg::blas1::scal( ghostP, -1.);
                dg::blas1::axpby( 2., right_, -1, ghostP);
                //dg::blas1::transform( ghostP, ghostP, dg::PLUS<double>( 2.*right_));
            }
            if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
            {
                dg::blas1::pointwiseDot( right_, hp, ghostM);
                dg::blas1::axpby( 1., ghostM, 1., ghostP);
            }
            dg::blas1::axpby( 1., ghostP, -1., tempP, ghostP);
            dg::blas1::pointwiseDot( limiter, ghostP, ghostP);
            dg::blas1::axpby( 1., ghostP, 1., tempP);
        }
        dg::blas1::axpby( 1., tempP, -1., tempM);
        thrust::transform( tempM.begin(), tempM.end(), hz.begin(), dzf.begin()+i0*size, thrust::divides<double>());
    }
}

template< class M, class container >
void DZ<M,container>::dzz( const container& f, container& dzzf)
{
    assert( &f != &dzzf);
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View tempPV( tempP.begin(), tempP.end());
    View temp0V( temp0.begin(), temp0.end());
    View tempMV( tempM.begin(), tempM.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;

        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cusp::copy( f0, temp0V);
        cusp::multiply( plus, fp, tempPV);
        cusp::multiply( minus, fm, tempMV );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
            {
                //dg::blas1::axpby( -1., temp0, 0., ghostM);
                //dg::blas1::transform( ghostM, ghostM, dg::PLUS<double>( 2.*left_));
                dg::blas1::axpby( 2., left_, -1, temp0, ghostM);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
            {
                dg::blas1::pointwiseDot( left_, hm, ghostP);
                dg::blas1::axpby( -1, ghostP, 1., temp0, ghostM);
                //dg::blas1::axpby( -left_, hm, 1., temp0, ghostM);
            }
            dg::blas1::axpby( 1., ghostM, -1., tempM, ghostM);
            dg::blas1::pointwiseDot( limiter, ghostM, ghostM);
            dg::blas1::axpby( 1., ghostM, 1., tempM);
        }
        else if( i0==g_.Nz()-1 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            {
                //dg::blas1::axpby( -1., temp0, 0., ghostP);
                //dg::blas1::transform( ghostP, ghostP, dg::PLUS<double>( 2.*right_));
                dg::blas1::axpby( 2., right_, -1, temp0, ghostP);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
            {
                dg::blas1::pointwiseDot( right_, hp, ghostM);
                dg::blas1::axpby( -1, ghostM, 1., temp0, ghostP);
                //dg::blas1::axpby( right_, hp, 1., temp0, ghostP);
            }
            dg::blas1::axpby( 1., ghostP, -1., tempP, ghostP);
            dg::blas1::pointwiseDot( limiter, ghostP, ghostP);
            dg::blas1::axpby( 1., ghostP, 1., tempP);
        }

        {
            dg::blas1::pointwiseDivide( tempP, hp, tempP);
            dg::blas1::pointwiseDivide( tempP, hz, tempP);
            dg::blas1::pointwiseDivide( temp0, hp, temp0);
            dg::blas1::pointwiseDivide( temp0, hm, temp0);
            dg::blas1::pointwiseDivide( tempM, hm, tempM);
            dg::blas1::pointwiseDivide( tempM, hz, tempM);
        }

        dg::blas1::axpby(  2., tempP, +2., tempM); //fp+fm
        dg::blas1::axpby( -2., temp0, +1., tempM); 
        View dzzf0( dzzf.begin() + i0*size, dzzf.begin() + (i0+1)*size);
        cusp::copy( tempMV, dzzf0);
    }
}

template< class M, class V >
void DZ<M,V>::cut( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp, dg::Grid2d<double>& g)
{
    //implements "Neumann" boundaries for lines that cross the wall
    for( unsigned i=0; i<g.size(); i++)
    {            
        if      (yp[0][i] < g.x0())  { yp[0][i] = y[0][i]; yp[1][i] = y[1][i]; }
        else if (yp[0][i] > g.x1())  { yp[0][i] = y[0][i]; yp[1][i] = y[1][i]; }
        else if (yp[1][i] < g.y0())  { yp[0][i] = y[0][i]; yp[1][i] = y[1][i]; }
        else if (yp[1][i] > g.y1())  { yp[0][i] = y[0][i]; yp[1][i] = y[1][i]; }
        else                         { }
            
    }

}

template< class M, class container>
template< class BinaryOp>
container DZ<M,container>::evaluate( BinaryOp f, unsigned p0)
{
    assert( p0 < g_.Nz());
    const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    container vec2d = dg::evaluate( f, g2d);
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
container DZ<M,container>::evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds)
{
    container vec3d = evaluate( f, p0);
    const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    //scal 
    for( unsigned i=0; i<g_.Nz(); i++)
    {
        View f0( vec3d.begin() + i*g2d.size(), vec3d.begin() + (i+1)*g2d.size());
        cusp::blas::scal(f0, g( g_.z0() + (double)(i+0.5)*g_.hz() ));
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
            cusp::blas::scal( f0, g( g_.z0() + (double)(k*g_.Nz()+i0+0.5)*g_.hz() ) );
        }
        for( int i0=g_.Nz()-1; i0>=0; i0--)
        {
            int ip = i0==g_.Nz()-1?0:i0+1;
            int k0 = k;
            int km = i0==g_.Nz()-1?k-1:k;
            View fp( vec4dM[km].begin() + ip*g2d.size(), vec4dM[km].begin() + (ip+1)*g2d.size());
            View f0( vec4dM[k0].begin() + i0*g2d.size(), vec4dM[k0].begin() + (i0+1)*g2d.size());
            cusp::multiply( plus, fp, f0 );
            cusp::blas::scal( f0, g( g_.z0() - (double)(k*g_.Nz()-0.5-i0)*g_.hz() ) );
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

}//namespace dg

