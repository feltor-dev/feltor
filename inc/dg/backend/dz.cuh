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
    DZ(Field field, const dg::Grid3d<double>& grid, double eps = 1e-3): g_(grid) 
    {
        std::cout<<"Constructing the parallel derivative" << "\n";
        dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
        hz.resize( g2d.size());
        tempM.resize( g2d.size());
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
        dg::blas1::axpby( 1., (container)yp[2], -1., (container)ym[2], hz);
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
        assert( &f != &dzf);
        unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        for( unsigned i=0; i<g_.Nz(); i++)
        {
            unsigned ip = (i==g_.Nz()-1) ? 0:i+1;
            unsigned im = (i==0) ? g_.Nz()-1:i-1;

            cusp::array1d_view< typename container::const_iterator> fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
            cusp::array1d_view< typename container::const_iterator> fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);

            {
                cusp::array1d_view< typename container::iterator> temp( tempP.begin(), tempP.end());
                cusp::multiply( plus, fp, temp);
            }
            {
                cusp::array1d_view< typename container::iterator> temp( tempM.begin(), tempM.end());
                cusp::multiply( minus, fm, temp );
            }

            dg::blas1::axpby( 1., tempP, -1., tempM);
            thrust::transform( tempM.begin(), tempM.end(), hz.begin(), dzf.begin()+i*size, thrust::divides<double>());
        }
    }
  private:
    void cut( const std::vector<dg::HVec>& y, std::vector<dg::HVec>& yp, dg::Grid2d<double>& g)
    {
        for( unsigned i=0; i<g.size(); i++)
        {            
            if      (yp[0][i] < g.x0())  { yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else if (yp[0][i] > g.x1())  {  yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else if (yp[1][i] < g.y0())  {  yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else if (yp[1][i] > g.y1())  {  yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else                         { }
                
        }

    }
    Matrix plus, minus; //interpolation matrices
    container hz, tempP, tempM;
    dg::Grid3d<double> g_;
};

}//namespace dg

