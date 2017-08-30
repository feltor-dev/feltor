#ifndef _DG_ARAKAWA_CUH
#define _DG_ARAKAWA_CUH

#include "blas.h"
#include "geometry.h"
#include "enums.h"
#include "backend/evaluation.cuh"
#include "backend/derivatives.h"
#ifdef MPI_VERSION
#include "backend/mpi_derivatives.h"
#include "backend/mpi_evaluation.h"
#endif

template<class container>
void pointwiseDot( double alpha, const std::vector<const container* >& x1, const std::vector<const container* >& y1, 
                   double beta,  const std::vector<const container* >& x2, const std::vector<const container* >& y2, 
                   double gamma, std::vector<container* > & z)
{
    unsigned K=x1.size();
    const double * RESTRICT x1_ptr[K]; 
    const double * RESTRICT y1_ptr[K]; 
    const double * RESTRICT x2_ptr[K]; 
    const double * RESTRICT y2_ptr[K]; 
    double * RESTRICT z_ptr[K];
    //const double *  x1_ptr[K]; 
    //const double *  y1_ptr[K]; 
    //const double *  x2_ptr[K]; 
    //const double *  y2_ptr[K]; 
    //double *  z_ptr[K];
    for(unsigned i=0; i<K; i++)
    {
        x1_ptr[i] = thrust::raw_pointer_cast( &(x1[i]->data()[0]));
        x2_ptr[i] = thrust::raw_pointer_cast( &(x2[i]->data()[0]));
        y1_ptr[i] = thrust::raw_pointer_cast( &(y1[i]->data()[0]));
        y2_ptr[i] = thrust::raw_pointer_cast( &(y2[i]->data()[0]));
         z_ptr[i] = thrust::raw_pointer_cast( &(z[i]->data()[0]));
    }
    unsigned size = x1[0]->size();
#pragma omp parallel
{
    double temp[K];
#pragma omp for simd
    for( unsigned i=0; i<size; i++)
    {
        for( unsigned l=0; l<K; l++)
        {
            temp[l] = alpha*x1_ptr[l][i]*y1_ptr[l][i] 
                      +beta*x2_ptr[l][i]*y2_ptr[l][i]
                      +gamma*z_ptr[l][i];
        }
        for( unsigned l=0; l<K; l++)
            z_ptr[l][i] = temp[l];
    }
}
}
template<class container>
void pointwiseDot( double alpha, const container& x1, const container& y1, 
                   double beta,  const container& x2, const container& y2, 
                   double gamma, container & z)
{
    const double * RESTRICT x1_ptr; 
    const double * RESTRICT y1_ptr; 
    const double * RESTRICT x2_ptr; 
    const double * RESTRICT y2_ptr; 
    double * RESTRICT z_ptr;
    //const double *  x1_ptr; 
    //const double *  y1_ptr; 
    //const double *  x2_ptr; 
    //const double *  y2_ptr; 
    //double *  z_ptr;
    {
        x1_ptr = thrust::raw_pointer_cast( &(x1.data()[0]));
        x2_ptr = thrust::raw_pointer_cast( &(x2.data()[0]));
        y1_ptr = thrust::raw_pointer_cast( &(y1.data()[0]));
        y2_ptr = thrust::raw_pointer_cast( &(y2.data()[0]));
         z_ptr = thrust::raw_pointer_cast( &(z.data()[0]));
    }
    unsigned size = x1.size();
if(gamma!=0)
{
#pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
    {
        z_ptr[i] = alpha*x1_ptr[i]*y1_ptr[i] 
                  +beta*x2_ptr[i]*y2_ptr[i]
                  +gamma*z_ptr[i];
    }
}
else
{
#pragma omp parallel for simd
    for( unsigned i=0; i<size; i++)
    {
        z_ptr[i] = alpha*x1_ptr[i]*y1_ptr[i] 
                  +beta*x2_ptr[i]*y2_ptr[i];
    }
}
}

/*! @file 
  
  object for computation of Poisson bracket
  */
namespace dg
{

/**
 * @brief X-space generalized version of Arakawa's scheme
 *
 * @ingroup arakawa
 * @tparam Matrix The Matrix class to use
 * @tparam container The vector class on which to operate on. The blas2 function symv( m, x, y) must be callable and may not change x. 
 */
template< class Geometry, class Matrix, class container >
struct ArakawaX
{
    /**
     * @brief Create Arakawa on a grid
     *
     * @tparam Grid The Grid class. The functions dg::create::dx( g, bcx) and
     * dg::create::dy( g, bcy) must be callable and return an instance of the Matrix class. Furthermore dg::evaluate( one, g) must return an instance of the container class.
     * @param g The grid
     */
    ArakawaX( Geometry g);
    /**
     * @brief Create Arakawa on a grid using different boundary conditions
     *
     * @tparam Grid The Grid class. The functions dg::create::dx( g, bcx) and
     * dg::create::dy( g, bcy) must be callable and return an instance of the Matrix class. Furthermore dg::evaluate( one, g) must return an instance of the container class.
     * @param g The grid
     * @param bcx The boundary condition in x
     * @param bcy The boundary condition in y
     */
    ArakawaX( Geometry g, bc bcx, bc bcy);

    /**
     * @brief Compute poisson's bracket
     *
     * Computes \f[ [f,g] := 1/\sqrt{g_{2d}}\left(\partial_x f\partial_y g - \partial_y f\partial_x g\right) \f]
     * where \f$ g_{2d} = g/g_{zz}\f$ is the two-dimensional volume element of the plane in 2x1 product space. 
     * @param lhs left hand side in x-space
     * @param rhs rights hand side in x-space
     * @param result Poisson's bracket in x-space
     */
    void operator()( const container& lhs, const container& rhs, container& result);

    /**
     * @brief Return internally used x - derivative 
     *
     * The same as a call to dg::create::dx( g, bcx)
     * @return derivative
     */
    const Matrix& dx() {return bdxf;}
    /**
     * @brief Return internally used y - derivative
     *
     * The same as a call to dg::create::dy( g, bcy)
     * @return derivative
     */
    const Matrix& dy() {return bdyf;}

    /**
     * @brief Compute the total variation integrand 
     *
     * Computes \f[ (\nabla\phi)^2 = \partial_i \phi g^{ij}\partial_j \phi \f]
     * in the plane of a 2x1 product space
     * @param phi function 
     * @param varphi may equal phi, contains result on output
     */
    void variation( const container& phi, container& varphi)
    {
        blas2::symv( bdxf, phi, dxrhs);
        blas2::symv( bdyf, phi, dyrhs);
        blas1::copy( dxrhs, dxlhs);//save results
        blas1::copy( dyrhs, dylhs);
        geo::raisePerpIndex( dxlhs, dylhs, varphi, helper_, grid); //input gets destroyed
        blas1::pointwiseDot( varphi, dxrhs, varphi);
        blas1::pointwiseDot( 1., helper_, dyrhs,1., varphi );
    }

  private:
    container dxlhs, dxrhs, dylhs, dyrhs, helper_;
    Matrix bdxf, bdyf;
    Geometry grid;
};

template<class Geometry, class Matrix, class container>
ArakawaX<Geometry, Matrix, container>::ArakawaX( Geometry g ): 
    dxlhs( dg::evaluate( one, g) ), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper_( dxlhs), 
    bdxf( dg::create::dx( g, g.bcx())),
    bdyf( dg::create::dy( g, g.bcy())), grid( g)
{ }
template<class Geometry, class Matrix, class container>
ArakawaX<Geometry, Matrix, container>::ArakawaX( Geometry g, bc bcx, bc bcy): 
    dxlhs( dg::evaluate( one, g) ), dxrhs(dxlhs), dylhs(dxlhs), dyrhs( dxlhs), helper_( dxlhs),
    bdxf(dg::create::dx( g, bcx)),
    bdyf(dg::create::dy( g, bcy)), grid(g)
{ }

template< class Geometry, class Matrix, class container>
void ArakawaX< Geometry, Matrix, container>::operator()( const container& lhs, const container& rhs, container& result)
{
    //compute derivatives in x-space
    blas2::symv( bdxf, lhs, dxlhs);
    blas2::symv( bdyf, lhs, dylhs);
    blas2::symv( bdxf, rhs, dxrhs);
    blas2::symv( bdyf, rhs, dyrhs);

    //std::vector<const container* > s0(3), t0(3), s1(3), t1(3); 
    //std::vector<container* > s2(3);
    //s0[0] = &dxlhs, t0[0] = &dyrhs, s1[0] = &dylhs, t1[0] = &dxrhs;
    //s0[1] =   &lhs, t0[1] = &dyrhs, s1[1] = &dylhs, t1[1] =   &rhs;
    //s0[2] = &dxlhs, t0[2] =   &rhs, s1[2] =   &lhs, t1[2] = &dxrhs;
    //s2[0] = &result, s2[1] = &dxlhs, s2[2] = &dxrhs;
    //pointwiseDot( 1./3., s0, t0, -1./3., s1, t1, 0., s2);

    pointwiseDot( 1./3., dxlhs, dyrhs, -1./3., dylhs, dxrhs, 0., result);
    pointwiseDot( 1./3.,   lhs, dyrhs, -1./3., dylhs,   rhs, 0., helper_);
    pointwiseDot( 1./3., dxlhs,   rhs, -1./3.,   lhs, dxrhs, 0., dylhs);
    blas2::symv( 1., bdxf, helper_, 1., result);
    blas2::symv( 1., bdyf, dylhs, 1., result);
    
    //// order is important now
    //// +x (1) -> result und (2) -> blhs
    //blas1::pointwiseDot( lhs, dyrhs, result);
    //blas1::pointwiseDot( lhs, dxrhs, helper_);

    //// ++ (1) -> dyrhs and (2) -> dxrhs
    //blas1::pointwiseDot( dxlhs, dyrhs, dyrhs);
    //blas1::pointwiseDot( dylhs, dxrhs, dxrhs);

    //// x+ (1) -> dxlhs and (2) -> dylhs
    //blas1::pointwiseDot( dxlhs, rhs, dxlhs);
    //blas1::pointwiseDot( dylhs, rhs, dylhs);

    //blas1::axpby( 1./3., dyrhs, -1./3., dxrhs);  //dxl*dyr - dyl*dxr -> dxrhs
    ////everything which needs a dx 
    //blas1::axpby( 1./3., dxlhs, -1./3., helper_);   //dxl*r - l*dxr     -> helper 
    ////everything which needs a dy
    //blas1::axpby( 1./3., result, -1./3., dylhs); //l*dyr - dyl*r     -> dylhs

    ////blas1::axpby( 0., dyrhs,  -0., dxrhs); //++
    //////for testing purposes (note that you need to set criss-cross)
    ////blas1::axpby( 1., dxlhs,  -0., helper); //x+ - +x
    ////blas1::axpby( 0., result, -1., dylhs);  //+x - x+

    //blas2::symv( 1., bdxf, dxlhs, 1., result);
    //blas2::symv( 1., bdyf, dxrhs, 1., result);
    geo::dividePerpVolume( result, grid);
}

}//namespace dg

#endif //_DG_ARAKAWA_CUH
