#pragma once

#include "dg/backend/mpi_matrix.h"
#include "dg/backend/blas2_dispatch_mpi.h"
#include "dg/backend/mpi_collective.h"
#include "dg/topology/grid.h"
#include "dg/topology/mpi_evaluation.h"
#include "dg/topology/mpi_grid.h"
#include "dg/topology/mpi_projection.h"
#include "dg/topology/interpolation.h"
#include "dg/topology/functions.h"
#include "dg/runge_kutta.h"
#include "fieldaligned.h"
#ifdef DG_BENCHMARK
#include "dg/backend/timer.h"
#endif

namespace dg{
namespace geo{

///@cond
namespace detail{

///basically a copy across processes
template<class thrust_vector0, class thrust_vector1>
void sendForward( const thrust_vector0& in, thrust_vector1& out, MPI_Comm comm) //send to next plane
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
template<class thrust_vector0, class thrust_vector1>
void sendBackward( const thrust_vector0& in, thrust_vector1& out, MPI_Comm comm) //send to next plane
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
    Fieldaligned(const dg::geo::TokamakMagneticField& vec,
        const ProductMPIGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi = -1):
            Fieldaligned( dg::geo::createBHat(vec),
                grid, bcx, bcy, limit, eps, mx, my, deltaPhi)
    {
    }
    template <class Limiter>
    Fieldaligned(const dg::geo::BinaryVectorLvl0& vec,
        const ProductMPIGeometry& grid,
        dg::bc bcx = dg::NEU,
        dg::bc bcy = dg::NEU,
        Limiter limit = FullLimiter(),
        double eps = 1e-5,
        unsigned mx=10, unsigned my=10,
        double deltaPhi = -1);
    template<class ...Params>
    void construct( Params&& ...ps)
    {
        //construct and swap
        *this = Fieldaligned( std::forward<Params>( ps)...);
    }

    dg::bc bcx()const{
        return m_bcx;
    }
    dg::bc bcy()const{
        return m_bcy;
    }

    void set_boundaries( dg::bc bcz, double left, double right)
    {
        m_bcz = bcz;
        const dg::MPIGrid2d g2d( 0., 1., 0., 1., m_g->global().n(), m_g->global().Nx(), m_g->global().Ny(), m_g->get_perp_comm() );
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
        dg::split( global, m_f, *m_g);
        dg::blas1::axpby( scal_left, m_f[0],               0., m_left);
        dg::blas1::axpby( scal_right, m_f[m_g->local().Nz()], 0., m_left);
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

    const MPI_Vector<LocalContainer>& hm_inv()const {
        return m_hm_inv;
    }
    const MPI_Vector<LocalContainer>& hp_inv()const {
        return m_hp_inv;
    }
    const MPI_Vector<LocalContainer>& h0_inv()const {
        return m_h0_inv;
    }
    const ProductMPIGeometry& grid() const{return *m_g;}
  private:
    void ePlus( enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    void eMinus(enum whichMatrix which, const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    MPIDistMat<LocalIMatrix, CommunicatorXY> m_plus, m_minus, m_plusT, m_minusT; //2d interpolation matrices
    MPI_Vector<LocalContainer> m_h0_inv, m_hm_inv, m_hp_inv; //3d size
    MPI_Vector<LocalContainer> m_hm, m_hp; //2d size
    MPI_Vector<LocalContainer> m_left, m_right; //2d size
    MPI_Vector<LocalContainer> m_limiter; //2d size
    MPI_Vector<LocalContainer> m_ghostM, m_ghostP; //2d size
    unsigned m_Nz, m_perp_size;
    dg::bc m_bcx, m_bcy, m_bcz;
    std::vector<MPI_Vector<dg::View<const LocalContainer>> > m_f;
    std::vector<MPI_Vector<dg::View<LocalContainer>> > m_temp;
    dg::ClonePtr<ProductMPIGeometry> m_g;
    unsigned m_coords2, m_sizeZ; //number of processes in z
};
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class MPIGeometry, class LocalIMatrix, class CommunicatorXY, class LocalContainer>
template <class Limiter>
Fieldaligned<MPIGeometry, MPIDistMat<LocalIMatrix, CommunicatorXY>, MPI_Vector<LocalContainer> >::Fieldaligned(
    const dg::geo::BinaryVectorLvl0& vec, const MPIGeometry& grid,
    dg::bc bcx, dg::bc bcy, Limiter limit, double eps,
    unsigned mx, unsigned my, double deltaPhi)
{
    ///Let us check boundary conditions:
    if( (grid.bcx() == PER && bcx != PER) || (grid.bcx() != PER && bcx == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting periodicity in x. The grid says "<<bc2str(grid.bcx())<<" while the parameter says "<<bc2str(bcx)));
    if( (grid.bcy() == PER && bcy != PER) || (grid.bcy() != PER && bcy == PER) )
        throw( dg::Error(dg::Message(_ping_)<<"Fieldaligned: Got conflicting boundary conditions in y. The grid says "<<bc2str(grid.bcy())<<" while the parameter says "<<bc2str(bcy)));
    m_Nz=grid.local().Nz(), m_bcz=grid.bcz(), m_bcx = bcx, m_bcy = bcy;
    m_g.reset(grid);
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( m_g->communicator(), 3, dims, periods, coords);
    m_coords2 = coords[2], m_sizeZ = dims[2];
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    dg::ClonePtr<aMPIGeometry2d> grid_coarse( grid.perp_grid());
    ///%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%//
    m_perp_size = grid_coarse->local().size();
    dg::assign( dg::pullback(limit, *grid_coarse), m_limiter);
    dg::assign( dg::evaluate(zero, *grid_coarse), m_left);
    m_ghostM = m_ghostP = m_right = m_left;
    ///%%%%%%%%%%Set starting points and integrate field lines%%%%%%%%%%%//
#ifdef DG_BENCHMARK
    dg::Timer t;
    int rank;
    MPI_Comm_rank( grid.communicator(), &rank);
    t.tic();
#endif
    std::array<thrust::host_vector<double>,3> yp_coarse, ym_coarse, yp, ym;
    dg::ClonePtr<dg::aMPIGeometry2d> grid_magnetic = grid_coarse;//INTEGRATE HIGH ORDER GRID
    grid_magnetic->set( 7, grid_magnetic->Nx(), grid_magnetic->Ny());
    dg::ClonePtr<dg::aGeometry2d> global_grid_magnetic = grid_magnetic->global_geometry();
    dg::MPIGrid2d grid_fine( *grid_coarse);//FINE GRID
    grid_fine.multiplyCellNumbers((double)mx, (double)my);
#ifdef DG_BENCHMARK
    t.toc();
    if(rank==0) std::cout << "# DS: High order grid gen   took: "<<t.diff()<<"\n";
    t.tic();
#endif
    detail::integrate_all_fieldlines2d( vec, *global_grid_magnetic, grid_coarse->local(), yp_coarse, ym_coarse, deltaPhi, eps);
    dg::IHMatrix interpolate = dg::create::interpolation( grid_fine.local(), grid_coarse->local());  //INTERPOLATE TO FINE GRID
    yp.fill(dg::evaluate( dg::zero, grid_fine.local())); ym = yp;
    for( int i=0; i<2; i++)
    {
        dg::blas2::symv( interpolate, yp_coarse[i], yp[i]);
        dg::blas2::symv( interpolate, ym_coarse[i], ym[i]);
    }
#ifdef DG_BENCHMARK
    t.toc();
    if(rank==0) std::cout << "# DS: Fieldline integration took: "<<t.diff()<<"\n";
    t.tic();
#endif
    ///%%%%%%%%%%%%%%%%Create interpolation and projection%%%%%%%%%%%%%%//
    dg::IHMatrix plusFine  = dg::create::interpolation( yp[0], yp[1], grid_coarse->global(), bcx, bcy), plus;
    dg::IHMatrix minusFine = dg::create::interpolation( ym[0], ym[1], grid_coarse->global(), bcx, bcy), minus;
    dg::IHMatrix projection = dg::create::projection( grid_coarse->local(), grid_fine.local());
    cusp::multiply( projection, plusFine, plus);
    cusp::multiply( projection, minusFine, minus);
#ifdef DG_BENCHMARK
    t.toc();
    if(rank==0) std::cout << "# DS: Multiplication PI     took: "<<t.diff()<<"\n";
    t.tic();
#endif
    dg::MIHMatrix temp = dg::convert( plus, *grid_coarse), tempT;
    tempT  = dg::transpose( temp);
    dg::blas2::transfer( temp, m_plus);
    dg::blas2::transfer( tempT, m_plusT);
    temp = dg::convert( minus, *grid_coarse);
    tempT  = dg::transpose( temp);
    dg::blas2::transfer( temp, m_minus);
    dg::blas2::transfer( tempT, m_minusT);

#ifdef DG_BENCHMARK
    t.toc();
    if(rank==0) std::cout << "# DS: Conversion            took: "<<t.diff()<<"\n";
#endif
    ///%%%%%%%%%%%%%%%%%%%%copy into h vectors %%%%%%%%%%%%%%%%%%%//
    dg::assign( dg::evaluate( dg::zero, grid), m_h0_inv);
    m_hp_inv = m_hm_inv = m_h0_inv;
    dg::assign( dg::evaluate( dg::zero, *grid_coarse), m_hp);
    m_hm = m_hp;
    dg::assign( yp_coarse[2], m_hp.data()); //2d vector
    dg::split( m_hp_inv, m_temp, grid); //3d vector
    for( unsigned i=0; i<m_Nz; i++)
        dg::blas1::copy( m_hp, m_temp[i]);
    dg::assign( ym_coarse[2], m_hm.data()); //2d vector
    dg::split( m_hm_inv, m_temp, grid); //3d vector
    for( unsigned i=0; i<m_Nz; i++)
        dg::blas1::copy( m_hm, m_temp[i]);
    dg::blas1::axpby( 1., m_hp_inv, -1., m_hm_inv, m_h0_inv);//hm is negative
    dg::blas1::pointwiseDivide( -1., m_hm_inv, m_hm_inv);
    dg::blas1::pointwiseDivide(  1., m_hp_inv, m_hp_inv);
    dg::blas1::pointwiseDivide(  1., m_h0_inv, m_h0_inv);
}

template<class G, class M, class C, class container>
template< class BinaryOp, class UnaryOp>
MPI_Vector<container> Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( p0 < m_g->global().Nz());
    const dg::ClonePtr<aMPIGeometry2d> g2d = m_g->perp_grid();
    MPI_Vector<container> init2d = dg::pullback( binary, *g2d);
    MPI_Vector<container> zero2d = dg::evaluate( dg::zero, *g2d);
    unsigned globalNz = m_g->global().Nz();

    MPI_Vector<container> temp(init2d), tempP(init2d), tempM(init2d);
    MPI_Vector<container> vec3d = dg::evaluate( dg::zero, *m_g);
    std::vector<MPI_Vector<container> >  plus2d(globalNz, zero2d), minus2d(plus2d), result(plus2d);
    unsigned turns = rounds;
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<globalNz; i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = r*globalNz + i0;
            for(unsigned k=0; k<rep; k++)
            {
                dg::blas2::symv( m_plus, tempP, temp);
                temp.swap( tempP);
                dg::blas2::symv( m_minus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*m_g->hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*m_g->hz() ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<m_Nz; i0++)
        {
            int idx = (int)(i0+m_coords2*m_Nz)  - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].data().begin(), result[i0].data().end(), vec3d.data().begin() + i0*m_perp_size);
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<globalNz; i0++)
        {
            unsigned revi0 = (globalNz - i0)%globalNz; //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<m_Nz; i0++)
        {
            int idx = ((int)i0 + m_coords2*m_Nz -(int)p0 + globalNz)%globalNz; //shift index
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
    dg::split( f, m_f, *m_g);
    dg::split( fpe, m_temp, *m_g);
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
        detail::sendBackward( m_temp[i0].data(), m_ghostM.data(), m_g->communicator());
        dg::blas1::copy( m_ghostM, m_temp[i0]);
    }

    //3. apply right boundary conditions in last plane
    unsigned i0=m_Nz-1;
    if( m_bcz != dg::PER && m_g->z1() == m_g->global().z1())
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
}

template<class G, class M, class C, class container>
void Fieldaligned<G,MPIDistMat<M,C>, MPI_Vector<container> >::eMinus( enum whichMatrix which, const MPI_Vector<container>& f, MPI_Vector<container>& fme )
{
    int rank;
    MPI_Comm_rank(m_g->communicator(), &rank);
    dg::split( f, m_f, *m_g);
    dg::split( fme, m_temp, *m_g);
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
        detail::sendForward( m_temp[i0].data(), m_ghostP.data(), m_g->communicator());
        dg::blas1::copy( m_ghostP, m_temp[i0]);
    }

    //3. apply left boundary conditions in first plane
    unsigned i0=0;
    if( m_bcz != dg::PER && m_g->z0() == m_g->global().z0())
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
}


///@endcond

}//namespace geo
}//namespace dg
