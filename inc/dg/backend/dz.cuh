#pragma once

#include "grid.h"
#include "interpolation.cuh"
#include "typedefs.cuh"
#include "../runge_kutta.h"

namespace dg{

    ///@cond
namespace detail{
double oneR( double R, double Z){return R;}
double oneZ( double R, double Z){return Z;}
double zero( double R, double Z){return 0;}


} //namespace detail
///@endcond

/**
 * @brief Class for the evaluation of a parallel derivative
 *
 * @ingroup dz
 * @tparam Field The Fieldlines to be integrated: Has to provide void  operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) where the first index is R, the second Z and the last s (the length of the field line)
 dg::HVec has to be used because of the cutting routine
 * @tparam container The container-class to operate on (does not need to be dg::HVec)
 */
template< class container=thrust::device_vector<double> >
struct DZ 
{
    typedef typename container::value_type value_type;
    typedef typename thrust::iterator_system<typename container::iterator>::type MemorySpace;
    //typedef cusp::ell_matrix<int, value_type, MemorySpace> Matrix;
    typedef dg::DMatrix Matrix;
    /**
     * @brief Construct from a field and a grid
     *
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
        tempM.resize( g2d.size());
        tempP.resize( g2d.size());
        std::vector<dg::HVec> y( 3, dg::evaluate( detail::oneR, g2d)), yp(y), ym(y); 
        y[1] = dg::evaluate( detail::oneZ, g2d);
        y[2] = dg::evaluate( detail::zero, g2d);
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
//         int c1=0,c2=0,c3=0,c4=0,c5=0;
        for( unsigned i=0; i<g.size(); i++)
        {            
            if      (yp[0][i] < g.x0())  { yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else if (yp[0][i] > g.x1())  {  yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else if (yp[1][i] < g.y0())  {  yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else if (yp[1][i] > g.y1())  {  yp[0][i] = y[0][i]; yp[1][i] = y[1][i];  }
            else                         { }
                
//             else {
//                 yp[0][i] = y[0][i];
//                 yp[1][i] = y[1][i];
//             }
            //if( func(y[0][i], y[1][i], M_PI/2.) - func(yp[0][i], yp[1][i], M_PI/2.) > 1e-10 )
            //{
            //    std::cerr << "Not on same radius\n";
            //    std::cerr << func(y[0][i], y[1][i], M_PI/2.) - func(yp[0][i], yp[1][i], M_PI/2.)<<"\n";
            //}
        }
//         std::cout << "c1 = " <<c1 <<  "\n";
//         std::cout << "c2 = " <<c2 <<  "\n";
//         std::cout << "c3 = " <<c3 <<  "\n";
//         std::cout << "c4 = " <<c4 <<  "\n";
//         std::cout << "c5 = " <<c5 <<  "\n";
//         std::cout << "sum= " <<c1+c2+c3+c4+c5 <<  "\n";

    }
    Matrix plus, minus; //interpolation matrices
    container hz, tempP, tempM;
    dg::Grid3d<double> g_;

};
}//namespace dg

