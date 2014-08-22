#pragma once

#include "grid.h"
#include "interpolation.cuh"
#include "typedefs.cuh"
#include "functions.h"
#include "../runge_kutta.h"

namespace dg{

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
     * @param field The field to integrate
     * @param grid The grid on which to operate
     * @param eps Desired accuracy of runge kutta
     */
    template <class Field>
    DZ(Field field, const dg::Grid3d<double>& grid, double eps = 1e-4): g_(grid) 
    {
        std::cout<<"Constructing the parallel derivative" << "\n";
        dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
        hz.resize( g2d.size());
        hp.resize( g2d.size());
        hm.resize( g2d.size());
        tempM.resize( g2d.size());
        temp0.resize( g2d.size());
        tempP.resize( g2d.size());
        std::vector<dg::HVec> y( 3, dg::evaluate( dg::coo1, g2d)), yp(y), ym(y); 
        y[1] = dg::evaluate( dg::coo2, g2d);
        y[2] = dg::evaluate( dg::zero, g2d);
        std::cout<<"Integrate with RK4" << "\n";
        dg::integrateRK4( field, y, yp,  g_.hz(), eps);
        cut( y, yp, g2d);
        dg::integrateRK4( field, y, ym, -g_.hz(), eps);
        cut( y, ym, g2d);
        plus  = dg::create::interpolation( yp[0], yp[1], g2d);
        minus = dg::create::interpolation( ym[0], ym[1], g2d);
        dg::blas1::axpby(  1., (container)yp[2], 0, hp);
        dg::blas1::axpby( -1., (container)ym[2], 0, hm);
        dg::blas1::axpby(  1., hp, +1., hm, hz);
        std::cout<<"Parallel derivative constructed" << "\n";
    }
    /**
     * @brief Apply the derivative on a 3d vector
     *
     * @param f The vector to derive
     * @param dzf contains result on output (write only)
     */
    void operator()( const container& f, container& dzf)
    {
        typedef cusp::array1d_view< typename container::iterator> View;
        typedef cusp::array1d_view< typename container::const_iterator> cView;
        assert( &f != &dzf);
        unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        for( unsigned i=0; i<g_.Nz(); i++)
        {
            unsigned ip = (i==g_.Nz()-1) ? 0:i+1;
            unsigned im = (i==0) ? g_.Nz()-1:i-1;

            cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
            cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);

            {
                View temp( tempP.begin(), tempP.end());
                cusp::multiply( plus, fp, temp);
            }
            {
                View temp( tempM.begin(), tempM.end());
                cusp::multiply( minus, fm, temp );
            }

            dg::blas1::axpby( 1., tempP, -1., tempM);
            thrust::transform( tempM.begin(), tempM.end(), hz.begin(), dzf.begin()+i*size, thrust::divides<double>());
        }
    }
    void dzz( const container& f, container& dzzf)
    {
        typedef cusp::array1d_view< typename container::iterator> View;
        typedef cusp::array1d_view< typename container::const_iterator> cView;
        assert( &f != &dzzf);
        unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        for( unsigned i0=0; i0<g_.Nz(); i0++)
        {
            unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;
            unsigned im = (i0==0) ? g_.Nz()-1:i0-1;

            cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
            cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
            cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);

            {
                View temp( tempP.begin(), tempP.end());
                cusp::multiply( plus, fp, temp);
                dg::blas1::pointwiseDivide( tempP, hp, tempP);
                dg::blas1::pointwiseDivide( tempP, hz, tempP);
            }
            {
                View temp( temp0.begin(), temp0.end());
                cusp::copy( f0, temp);
                dg::blas1::pointwiseDivide( temp0, hp, temp0);
                dg::blas1::pointwiseDivide( temp0, hm, temp0);
            }
            View temp( tempM.begin(), tempM.end());
            cusp::multiply( minus, fm, temp ); 
            dg::blas1::pointwiseDivide( tempM, hm, tempM);
            dg::blas1::pointwiseDivide( tempM, hz, tempM);

            dg::blas1::axpby( 2., tempP, +2., tempM); //fp+fm
            dg::blas1::axpby( -2., temp0, +1., tempM); 
            View dzzf0( dzzf.begin() + i0*size, dzzf.begin() + (i0+1)*size);
            cusp::copy( temp, dzzf0);
        }
    }
  private:
    void cut( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp, dg::Grid2d<double>& g)
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
    Matrix plus, minus; //interpolation matrices
    container hz, hp,hm, tempP, temp0, tempM;
    dg::Grid3d<double> g_;
};

}//namespace dg

