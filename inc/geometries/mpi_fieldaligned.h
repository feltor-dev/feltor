#pragma once

#include "fieldaligned.h"
#include "dg/backend/grid.h"
#include "dg/backend/mpi_evaluation.h"
#include "dg/backend/mpi_matrix.h"
#include "dg/backend/mpi_matrix_blas.h"
#include "dg/backend/mpi_collective.h"
#include "dg/backend/mpi_grid.h"
#include "dg/backend/mpi_projection.h"
#include "dg/backend/interpolation.cuh"
#include "dg/backend/functions.h"
#include "dg/runge_kutta.h"

namespace dg{
namespace geo{
 
///@cond
namespace detail{

///basically a copy across processes
template<class thrust_vector>
void sendForward( const thrust_vector& in, thrust_vector& out, MPI_Comm comm) //send to next plane
{
    int source, dest;
    MPI_Status status;
    MPI_Cart_shift( comm, 2, +1, &source, &dest);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize();//wait until device functions are finished before sending data
#endif //THRUST_DEVICE_SYSTEM
    unsigned size = in.size();
    MPI_Sendrecv(   thrust::raw_pointer_cast(in.data()), size, MPI_DOUBLE,  //sender
                    dest, 9,  //destination
                    thrust::raw_pointer_cast(out.data()), size, MPI_DOUBLE, //receiver
                    source, 9, //source
                    comm, &status);
}
///basically a copy across processes
template<class thrust_vector>
void sendBackward( const thrust_vector& in, thrust_vector& out, MPI_Comm comm) //send to next plane
{
    int source, dest;
    MPI_Status status;
    MPI_Cart_shift( comm, 2, -1, &source, &dest);
#if THRUST_DEVICE_SYSTEM==THRUST_DEVICE_SYSTEM_CUDA
    cudaDeviceSynchronize();//wait until device functions are finished before sending data
#endif //THRUST_DEVICE_SYSTEM
    unsigned size = in.size();
    MPI_Sendrecv(   thrust::raw_pointer_cast(in.data()), size, MPI_DOUBLE,  //sender
                    dest, 3,  //destination
                    thrust::raw_pointer_cast(out.data()), size, MPI_DOUBLE, //receiver
                    source, 3, //source
                    comm, &status);
}
}//namespace detail

template <class ProductMPIGeometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
struct Fieldaligned< ProductMPIGeometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> > 
{
    Fieldaligned(){}
    template <class Limiter>
    Fieldaligned(const dg::geo::BinaryVectorLvl0& vec, const ProductMPIGeometry& grid, unsigned multiplyX, unsigned multiplyY, bool dependsOnX, bool dependsOnY, double eps = 1e-5, dg::bc globalbcx = dg::NEU, dg::bc globalbcy = dg::NEU, Limiter limit = FullLimiter(), double deltaPhi = -1)
    {
        construct( vec, grid, multiplyX, multiplyY, dependsOnX, dependsOnY, eps, globalbcx, globalbcy, limit, deltaPhi);
    }
    template <class Limiter>
    void construct(const dg::geo::BinaryVectorLvl0& vec, const ProductMPIGeometry& grid, unsigned multiplyX, unsigned multiplyY, bool dependsOnX, bool dependsOnY, double eps = 1e-5, dg::bc globalbcx = dg::NEU, dg::bc globalbcy = dg::NEU, Limiter limit = FullLimiter(), double deltaPhi = -1);

    bool dependsOnX()const{return m_dependsOnX;}
    bool dependsOnY()const{return m_dependsOnY;}

    void set_boundaries( dg::bc bcz, double left, double right)
    {
        m_bcz = bcz; 
        const dg::MPIGrid2d g2d( 0., 1., 0., 1., m_g.get().global().n(), m_g.get().global().Nx(), m_g.get().global().Ny(), m_g.get().perp_communicator() );
        m_left  = dg::evaluate( dg::CONSTANT(left), g2d);
        m_right = dg::evaluate( dg::CONSTANT(right),g2d);
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& left, const MPI_Vector<LocalContainer>& right)
    {
        m_bcz = bcz; 
        m_left = left;
        m_right = right;
    }

    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& global, double scal_left, double scal_right)
    {
        dg::split( global, m_temp);
        dg::blas1::axpby( scal_left, m_temp[0],               0., m_left);
        dg::blas1::axpby( scal_right, m_temp[m_g.get().Nz()], 0., m_left);
        m_bcz = bcz;
    }

    template< class BinaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp binary, unsigned p0=0) const
    {
        return evaluate( binary, dg::CONSTANT(1), p0, 0);
    }

    template< class BinaryOp, class UnaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const;

    void operator()(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);

    const MPI_Vector<LocalContainer>& hz_inv()const {return m_hz_inv;}
    const MPI_Vector<LocalContainer>& hp_inv()const {return m_hp_inv;}
    const MPI_Vector<LocalContainer>& hm_inv()const {return m_hm_inv;}
    const ProductMPIGeometry& grid() const{return m_g.get();}
  private:
    void ePlus( enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    void eMinus(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    MPIDistMat<LocalIMatrix, CommunicatorXY> m_plus, m_minus, m_plusT, m_minusT; //2d interpolation matrices
    MPI_Vector<LocalContainer> m_hz_inv, m_hp_inv, m_hm_inv; //3d size
    MPI_Vector<LocalContainer> m_hp, m_hm;      //2d size
    MPI_Vector<LocalContainer> m_left, m_right; //2d size
    MPI_Vector<LocalContainer> m_limiter; //2d size
    MPI_Vector<LocalContainer> m_ghostM, m_ghostP; //2d size
    unsigned m_Nz, m_perp_size; 
    dg::bc m_bcz;
    std::vector<MPI_Vector<LocalContainer> > m_f, m_temp;  //split 3d vectors
    dg::Handle<ProductMPIGeometry> m_g;
    bool m_dependsOnX, m_dependsOnY;
    unsigned m_coords2, m_sizeZ; //number of processes in z
};
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class MPIGeometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
template <class Limiter>
void Fieldaligned<MPIGeometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> >::construct(
    const dg::geo::BinaryVectorLvl0& vec, const MPIGeometry& grid, unsigned mx, unsigned my, bool bx, bool by, double eps, dg::bc globalbcx, dg::bc globalbcy, Limiter limit, double deltaPhi)
{
    m_dependsOnX=bx, m_dependsOnY=by;
    m_Nz=grid.Nz(), m_bcz=grid.bcz(); 
    m_g.reset(grid);
    dg::blas1::transfer( dg::evaluate( dg::zero, grid), m_hz_inv), m_hp_inv= m_hz_inv, m_hm_inv= m_hz_inv;
    dg::split( m_hz_inv, m_temp, grid);
    dg::split( m_hz_inv, m_f, grid);
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( m_g.get().communicator(), 3, dims, periods, coords);
    m_coords2 = coords[2], m_sizeZ = dims[2];
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    const aMPIGeometry2d* grid2d_ptr = grid.perp_grid();
    //%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    m_perp_size = grid2d_ptr->size();
    dg::blas1::transfer( dg::pullback(limit, *grid2d_ptr), m_limiter);
    m_right = m_left = dg::evaluate( zero, *grid2d_ptr);
    dg::blas1::transfer( dg::evaluate(zero, *grid2d_ptr), m_left);
    m_ghostM = m_ghostP = m_right = m_left;
    //%%%%%%%%%%%%%%%%%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%%%%
    std::vector<thrust::host_vector<double> > yp_coarse( 3), ym_coarse(yp_coarse), yp, ym; 
    
    dg::aMPIGeometry2d* g2dField_ptr = grid2d_ptr->clone();//INTEGRATE HIGH ORDER GRID
    //g2dField_ptr->set( 7, g2dField_ptr->global().Nx(), g2dField_ptr->global().Ny());
    dg::aGeometry2d* global_g2dField_ptr = g2dField_ptr->global_geometry();
    dg::Grid2d local_g2dField = g2dField_ptr->local();
    detail::integrate_all_fieldlines2d( vec, global_g2dField_ptr, &local_g2dField, yp_coarse, ym_coarse, deltaPhi, eps);

    dg::MPIGrid2d g2dFine((dg::MPIGrid2d(*grid2d_ptr)));//FINE GRID
    g2dFine.multiplyCellNumbers((double)mx, (double)my);
    dg::IHMatrix interpolate = dg::create::interpolation( g2dFine.local(), local_g2dField);  //INTERPOLATE TO FINE GRID
    dg::geo::detail::interpolate_and_clip( interpolate, g2dFine.local(), *global_g2dField_ptr, yp_coarse, ym_coarse, yp, ym);
    delete g2dField_ptr;
    delete global_g2dField_ptr;
    //%%%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dg::IHMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], grid2d_ptr->global(), globalbcx, globalbcy), plus, plusT;
    dg::IHMatrix minusFine = dg::create::interpolation( ym[0], ym[1], grid2d_ptr->global(), globalbcx, globalbcy), minus, minusT;
    dg::IHMatrix projection = dg::create::projection( grid2d_ptr->local(), g2dFine.local());
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, minusFine, minus);
    //%Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix!!!
    dg::MIHMatrix temp = dg::convert( plus, *grid2d_ptr);
    dg::blas2::transfer( temp, m_plus);
    temp = dg::convert( minus, *grid2d_ptr);
    dg::blas2::transfer( temp, m_minus);
    m_plusT = dg::transpose( m_plus);
    m_minusT = dg::transpose( m_minus);
    //%%%%%%%%%%%%%%%%%%%%%%%project h and copy into h vectors%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    dg::MHVec hp( dg::evaluate( dg::zero, *grid2d_ptr)), hm(hp), hz(hp);
    dg::blas2::symv( projection, yp[2], hp.data());
    dg::blas2::symv( projection, ym[2], hm.data());
    dg::blas1::scal( hm, -1.);
    dg::blas1::axpby(  1., hp, +1., hm, hz);
    dg::blas1::transfer( hp, m_hp);
    dg::blas1::transfer( hm, m_hm);
    dg::blas1::transform( hp, hp, dg::INVERT<double>());
    dg::blas1::transform( hm, hm, dg::INVERT<double>());
    dg::blas1::transform( hz, hz, dg::INVERT<double>());
    dg::join( std::vector<dg::MHVec >( m_Nz, hp), m_hp_inv, grid);
    dg::join( std::vector<dg::MHVec >( m_Nz, hm), m_hm_inv, grid);
    dg::join( std::vector<dg::MHVec >( m_Nz, hz), m_hz_inv, grid);
    delete grid2d_ptr;
}

template<class G, class M, class C, class container>
template< class BinaryOp, class UnaryOp>
MPI_Vector<container> Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( p0 < m_g.get().global().Nz());
    const aMPIGeometry2d* g2d_ptr = m_g.get().perp_grid();
    MPI_Vector<container> init2d = dg::pullback( binary, *g2d_ptr); 
    delete g2d_ptr;
    unsigned Nz = m_g.get().global().Nz();

    MPI_Vector<container> temp(init2d), tempP(init2d), tempM(init2d);
    MPI_Vector<container> vec3d = dg::evaluate( dg::zero, m_g.get());
    std::vector<MPI_Vector<container> >  plus2d, minus2d, result;
    dg::split( vec3d, plus2d, m_g.get());
    dg::split( vec3d, minus2d, m_g.get());
    dg::split( vec3d, result, m_g.get());
    unsigned turns = rounds; 
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<Nz; i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = i0 + r*Nz; 
            for(unsigned k=0; k<rep; k++)
            {
                dg::blas2::symv( m_plus, tempP, temp);
                temp.swap( tempP);
                dg::blas2::symv( m_minus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*m_g.get().hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*m_g.get().hz() ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<Nz; i0++)
        {
            int idx = (int)(i0+m_coords2*m_g.get().Nz())  - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].data().begin(), result[i0].data().end(), vec3d.data().begin() + i0*m_perp_size);
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<Nz; i0++)
        {
            unsigned revi0 = (Nz - i0)%Nz; //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<Nz; i0++)
        {
            int idx = ((int)i0 + m_coords2*m_g.get().Nz() -(int)p0 + Nz)%Nz; //shift index
            thrust::copy( result[idx].data().begin(), result[idx].data().end(), vec3d.data().begin() + i0*m_perp_size);
        }
    }
    return vec3d;
}

template<class G, class M, class C, class container>
void Fieldaligned<G, MPIDistMat<M,C>, MPI_Vector<container> >::operator()(enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fe)
{
    if(which == einsPlus || which == einsMinusT) ePlus( which, f, fe);
    if(which == einsMinus || which == einsPlusT) eMinus( which, f, fe);
}

template<class G, class M, class C, class container>
void Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::ePlus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fpe ) 
{
    dg::split( f, m_f, m_g.get());
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned ip = (i0==m_Nz-1) ? 0:i0+1;
        if(which == einsPlus)           dg::blas2::symv( m_plus,   m_f[ip], m_temp[i0]);
        else if(which == einsMinusT)    dg::blas2::symv( m_minusT, m_f[ip], m_temp[i0]);
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = m_Nz-1;
        detail::sendBackward( m_temp[i0].data(), m_ghostM.data(), m_g.get().communicator());
        m_temp[i0].swap( m_ghostM);
    }

    //3. apply right boundary conditions in last plane
    unsigned i0=m_Nz-1;
    if( m_bcz != dg::PER && m_g.get().z1() == m_g.get().global().z1())
    {
        if( m_bcz == dg::DIR || m_bcz == dg::NEU_DIR)
            dg::blas1::axpby( 2, m_right, -1., m_f[i0], m_ghostP);
        if( m_bcz == dg::NEU || m_bcz == dg::DIR_NEU)
        {
            dg::blas1::pointwiseDot( m_right, m_hp, m_ghostP);
            dg::blas1::axpby( 1., m_ghostP, 1., m_f[i0], m_ghostP);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
        dg::blas1::axpby( 1., m_ghostP, -1., m_temp[i0], m_ghostP);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostP, 1., m_temp[i0]);
    }
    dg::join( m_temp, fpe, m_g.get());
}

template<class G, class M, class C, class container>
void Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::eMinus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fme ) 
{
    dg::split( f, m_f, m_g.get());
    //1. compute 2d interpolation in every plane and store in m_temp
    for( unsigned i0=0; i0<m_Nz; i0++)
    {
        unsigned im = (i0==0) ? m_Nz-1:i0-1;
        if(which == einsPlusT)         dg::blas2::symv( m_plusT, m_f[im], m_temp[i0]);
        else if(which == einsMinus)    dg::blas2::symv( m_minus, m_f[im], m_temp[i0]);
    }

    //2. communicate halo in z
    if( m_sizeZ != 1)
    {
        unsigned i0 = 0;
        detail::sendForward( m_temp[i0].data(), m_ghostP.data(), m_g.get().communicator());
        m_temp[i0].swap( m_ghostP);
    }

    //3. apply left boundary conditions in first plane
    unsigned i0=0;
    if( m_bcz != dg::PER && m_g.get().z0() == m_g.get().global().z0())
    {
        if( m_bcz == dg::DIR || m_bcz == dg::DIR_NEU)
            dg::blas1::axpby( 2., m_left,  -1., m_f[i0], m_ghostM);
        if( m_bcz == dg::NEU || m_bcz == dg::NEU_DIR)
        {
            dg::blas1::pointwiseDot( m_left, m_hm, m_ghostM);
            dg::blas1::axpby( -1., m_ghostM, 1., m_f[i0], m_ghostM);
        }
        //interlay ghostcells with periodic cells: L*g + (1-L)*fme
        dg::blas1::axpby( 1., m_ghostM, -1., m_temp[i0], m_ghostM);
        dg::blas1::pointwiseDot( 1., m_limiter, m_ghostM, 1., m_temp[i0]);
    }
    dg::join( m_temp, fme, m_g.get());
}


///@endcond

}//namespace geo
}//namespace dg
