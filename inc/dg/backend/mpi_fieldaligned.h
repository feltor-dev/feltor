#pragma once

#include "fieldaligned.h"
#include "grid.h"
#include "mpi_evaluation.h"
#include "mpi_matrix.h"
#include "mpi_matrix_blas.h"
#include "mpi_collective.h"
#include "mpi_grid.h"
#include "interpolation.cuh"
#include "functions.h"
#include "../runge_kutta.h"

namespace dg{


/**
 * @brief Class for the evaluation of a parallel derivative (MPI Version)
 *
 * @ingroup dz
 * @tparam Matrix The matrix class of the interpolation matrix
 * @tparam container The container-class to on which the interpolation matrix operates on (does not need to be dg::HVec)
 */
template <class LocalMatrix, class Communicator, class LocalContainer>
struct MPI_FieldAligned
{
    /**
    * @brief Construct from a field and a grid
    *
    * @tparam Field The Fieldlines to be integrated: Has to provide void operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) where the first index is R, the second Z and the last s (the length of the field line)
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
    is a limiter and 0 if there isn't. If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply. 
    * @param field The field to integrate
    * @param grid The grid on which to operate
    * @param deltaPhi Must either equal the hz_() value of the grid or a fictive deltaPhi if the grid is 2D and Nz=1
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere, note that if bcz is periodic it doesn't matter if there is a limiter or not)
    * @param globalbcz Choose NEU or DIR. Defines BC in parallel on box
    * @note If there is a limiter, the boundary condition is set by the bcz variable from the grid and can be changed by the set_boundaries function. If there is no limiter the boundary condition is periodic.
    */
    template <class Field, class Limiter>
    MPI_FieldAligned(Field field, const dg::MPI_Grid3d& grid, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR, double deltaPhi = -1 );

    /**
     * @brief Set boundary conditions
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
     * @brief Set boundary conditions
     *
     * if Dirichlet boundaries are used the left value is the left function
     value, if Neumann boundaries are used the left value is the left derivative value
     * @param bcz boundary condition
     * @param left left boundary value 
     * @param right right boundary value
     */
    void set_boundaries( dg::bc bcz, const LocalContainer& left, const LocalContainer& right)
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
    void set_boundaries( dg::bc bcz, const MPI_Vector<LocalContainer>& global, double scal_left, double scal_right);

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
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, unsigned plane=0) const;

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
     * @param p0 The number of the plane to start
     * @param rounds The number of rounds to follow a fieldline
     *
     * @return Returns an instance of container
     */
    template< class BinaryOp, class UnaryOp>
    MPI_Vector<LocalContainer> evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const;

    /**
    * @brief Applies the interpolation to the next planes 
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsPlus( const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    /**
    * @brief Applies the interpolation to the previous planes
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsMinus( const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    /**
    * @brief Applies the transposed interpolation to the previous plane 
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsPlusT( const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    /**
    * @brief Applies the transposed interpolation to the next plane 
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsMinusT( const MPI_Vector<LocalContainer>& in, MPI_Vector<LocalContainer>& out);
    /**
    * @brief hz is the distance between the plus and minus planes
    *
    * @return three-dimensional vector
    */
    const MPI_Vector<LocalContainer>& hz()const {return hz_;}
    /**
    * @brief hp is the distance between the plus and current planes
    *
    * @return three-dimensional vector
    */
    const MPI_Vector<LocalContainer>& hp()const {return hp_;}
    /**
    * @brief hm is the distance between the current and minus planes
    *
    * @return three-dimensional vector
    */
    const MPI_Vector<LocalContainer>& hm()const {return hm_;}
    /**
    * @brief Access the underlying grid
    *
    * @return the grid
    */
    const MPI_Grid3d& grid() const{return g_;}
  private:
    typedef cusp::array1d_view< typename LocalContainer::iterator> View;
    typedef cusp::array1d_view< typename LocalContainer::const_iterator> cView;
    MPI_Vector<LocalContainer> hz_, hp_, hm_; 
    LocalContainer ghostM, ghostP;
    MPI_Grid3d g_;
    dg::bc bcz_;
    LocalContainer left_, right_;
    LocalContainer limiter_;
    ColDistMat<LocalMatrix, Communicator> plus, minus; //interpolation matrices
    RowDistMat<LocalMatrix, Communicator> plusT, minusT; //interpolation matrices
    //Communicator collM_, collP_;

    dg::FieldAligned<LocalMatrix, LocalContainer > dz_;
};
///@cond
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class LocalMatrix, class Communicator, class LocalContainer>
template <class Field, class Limiter>
MPI_FieldAligned<LocalMatrix, Communicator, LocalContainer>::MPI_FieldAligned(Field field, const dg::MPI_Grid3d& grid, double eps, Limiter limit, dg::bc globalbcz, double deltaPhi ): 
    hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
    g_(grid), bcz_(grid.bcz()),  
    dz_(field, grid.global(), eps, limit, globalbcz)
{
    //Resize vector to local 2D grid size
    dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());  
    unsigned size = g2d.size();
    limiter_ = dg::evaluate( limit, g2d);
    right_ = left_ = dg::evaluate( zero, g2d);
    ghostM.resize( size); ghostP.resize( size);
    //set up grid points as start for fieldline integrations 
    std::vector<thrust::host_vector<double> > y( 3);
    y[0] = dg::evaluate( dg::coo1, grid.local());
    y[1] = dg::evaluate( dg::coo2, grid.local());
    y[2] = dg::evaluate( dg::zero, grid.local());//distance (not angle)
    //integrate to next z-planes
    std::vector<thrust::host_vector<double> > yp(y), ym(y); 
    if(deltaPhi<=0) deltaPhi = g_.hz();
    else assert( g_.Nz() == 1 || grid.hz()==deltaPhi);
    unsigned localsize = grid.local().size();
#ifdef _OPENMP
#pragma omp parallel for firstprivate(field)
#endif //_OPENMP
    for( unsigned i=0; i<localsize; i++)
    {
        thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
        coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i];
        double phi1 = deltaPhi;
        boxintegrator( field, g_.global(), coords, coordsP, phi1, eps, globalbcz);
        phi1 = -deltaPhi;
        boxintegrator( field, g_.global(), coords, coordsM, phi1, eps, globalbcz);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }


    //determine pid of result 
    thrust::host_vector<int> pids( localsize);
    thrust::host_vector<double> angle = dg::evaluate( dg::coo3, grid.local());
    for( unsigned i=0; i<pids.size(); i++)
    {
        angle[i] += deltaPhi;
        if( angle[i] >= grid.global().z1()) angle[i] -= grid.global().lz();
        pids[i]  = grid.pidOf( yp[0][i], yp[1][i], angle[i]);
        if( pids[i]  == -1)
        {
            std::cerr << "ERROR: PID NOT FOUND!\n";
            return;
        }
    }
    //construct scatter operation from pids
    Communicator cp( pids, grid.communicator());
    thrust::host_vector<double> pX = cp.collect( yp[0]),
                                pY = cp.collect( yp[1]),
                                pZ = cp.collect( angle);
    //construt interpolation matrix
    LocalMatrix inter = dg::create::interpolation( pX, pY, pZ, grid.local());
    plus = ColDistMat<LocalMatrix, Communicator>(inter, cp);
    LocalMatrix interT;
    cusp::transpose( inter, interT);
    plusT = RowDistMat<LocalMatrix, Communicator>( interT, cp);

    

    //do the same for the minus z-plane
    for( unsigned i=0; i<pids.size(); i++)
    {
        angle[i] -= 2.*deltaPhi;
        if( angle[i] <= grid.global().z0()) angle[i] += grid.global().lz();
        pids[i]  = grid.pidOf( ym[0][i], ym[1][i], angle[i]);
        if( pids[i] == -1)
        {
            std::cerr << "ERROR: PID NOT FOUND!\n";
            return;
        }
    }
    Communicator cm( pids, grid.communicator());
    pX = cm.collect( ym[0]),
    pY = cm.collect( ym[1]),
    pZ = cm.collect( angle);
    inter = dg::create::interpolation( pX, pY, pZ, grid.local());
    minus = ColDistMat<LocalMatrix, Communicator>(inter, cm);
    cusp::transpose( inter, interT);
    minusT = RowDistMat<LocalMatrix, Communicator>( interT, cm);
    //copy to device
    thrust::copy( yp[2].begin(), yp[2].end(), hp_.data().begin());
    thrust::copy( ym[2].begin(), ym[2].end(), hm_.data().begin());
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);
}

template<class M, class C, class container>
template< class BinaryOp>
MPI_Vector<container> MPI_FieldAligned<M,C,container>::evaluate( BinaryOp f, unsigned p0) const
{
    container global_vec = dz_.evaluate( f, p0);
    container vec = dg::evaluate( dg::zero, g_.local());
    //now take the relevant part 
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
    unsigned Nx = g_.Nx()*g_.n(), Ny = g_.Ny()*g_.n(), Nz = g_.Nz();
    for( unsigned s=0; s<Nz; s++)
        for( unsigned i=0; i<Ny; i++)
            for( unsigned j=0; j<Nx; j++)
                vec[ (s*Ny+i)*Nx + j ] 
                    = global_vec[ j + Nx*(coords[0] + dims[0]*( i +Ny*(coords[1] + dims[1]*(s +Nz*coords[2])))) ];
                   
    return MPI_Vector<container>( vec, g_.communicator());
}

template<class M, class C, class container>
template< class BinaryOp, class UnaryOp>
MPI_Vector<container> MPI_FieldAligned<M,C,container>::evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const
{
    //let all processes integrate the fieldlines
    container global_vec = dz_.evaluate( f, g,p0, rounds);
    container vec = dg::evaluate( dg::zero, g_.local());
    //now take the relevant part 
    int dims[3], periods[3], coords[3];
    MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
    unsigned Nx = g_.Nx()*g_.n(), Ny = g_.Ny()*g_.n(), Nz = g_.Nz();
    for( unsigned s=0; s<Nz; s++)
        for( unsigned i=0; i<Ny; i++)
            for( unsigned j=0; j<Nx; j++)
                vec[ (s*Ny+i)*Nx + j ] 
                    = global_vec[ j + Nx*(coords[0] + dims[0]*( i +Ny*(coords[1] + dims[1]*(s +Nz*coords[2])))) ];
    return MPI_Vector<container>( vec, g_.communicator());
}

template<class M, class C, class container>
void MPI_FieldAligned<M,C,container>::einsPlus( const MPI_Vector<container>& f, MPI_Vector<container>& fplus ) 
{
    dg::blas2::detail::doSymv( plus, f, fplus, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    //make ghostcells in last plane
    const container& in = f.data();
    container& out = fplus.data();
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z1() == g_.global().z1())
    {
        unsigned i0 = g_.Nz()-1, im = g_.Nz()-2, ip = 0;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostPV);
        if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::axpby( 2., right_, -1, ghostP);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
        {
            //note that hp_ is 3d and the rest 2d
            thrust::transform( right_.begin(), right_.end(),  hp_.data().begin(), ghostM.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( 1., ghostM, 1., ghostP);
        }
        cusp::blas::axpby(  ghostPV,  outV, ghostPV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
        cusp::blas::axpby(  ghostPV,  outV, outV, 1.,1.);
    }

}

template<class M, class C, class container>
void MPI_FieldAligned<M,C,container>::einsMinus( const MPI_Vector<container>& f, MPI_Vector<container>& fminus ) 
{
    dg::blas2::detail::doSymv( minus, f, fminus, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    //make ghostcells in first plane
    const container& in = f.data();
    container& out = fminus.data();
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z0() == g_.global().z0())
    {
        unsigned i0 = 0, im = g_.Nz()-1, ip = 1;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostMV);
        if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::axpby( 2., left_, -1, ghostM);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
        {
            thrust::transform( left_.begin(), left_.end(), hm_.data().begin(), ghostP.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( -1, ghostP, 1., ghostM);
        }
        cusp::blas::axpby(  ghostMV,  outV, ghostMV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
        cusp::blas::axpby(  ghostMV,  outV, outV, 1.,1.);
    }
}
template< class M, class C, class container>
void MPI_FieldAligned<M,C,container>::einsMinusT( const MPI_Vector<container>& f, MPI_Vector<container>& fpe)
{
    dg::blas2::detail::doSymv( minusT, f, fpe, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    //make ghostcells in last plane
    const container& in = f.data();
    container& out = fpe.data();
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z1() == g_.global().z1())
    {
        unsigned i0 = g_.Nz()-1, im = g_.Nz()-2, ip = 0;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostPV);
        if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
        {
            dg::blas1::axpby( 2., right_, -1, ghostP);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
        {
            //note that hp_ is 3d and the rest 2d
            thrust::transform( right_.begin(), right_.end(),  hp_.data().begin(), ghostM.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( 1., ghostM, 1., ghostP);
        }
        cusp::blas::axpby(  ghostPV,  outV, ghostPV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
        cusp::blas::axpby(  ghostPV,  outV, outV, 1.,1.);
    }
}
template< class M, class C, class container>
void MPI_FieldAligned<M,C,container>::einsPlusT( const MPI_Vector<container>& f, MPI_Vector<container>& fme)
{
    dg::blas2::detail::doSymv( plusT, f, fme, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    //make ghostcells in first plane
    const container& in = f.data();
    container& out = fme.data();
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    if( bcz_ != dg::PER && g_.z0() == g_.global().z0())
    {
        unsigned i0 = 0, im = g_.Nz()-1, ip = 1;
        cView fp( in.cbegin() + ip*size, in.cbegin() + (ip+1)*size);
        cView f0( in.cbegin() + i0*size, in.cbegin() + (i0+1)*size);
        cView fm( in.cbegin() + im*size, in.cbegin() + (im+1)*size);
        View outV( out.begin() + i0*size, out.begin() + (i0+1)*size);
        View ghostPV( ghostP.begin(), ghostP.end());
        View ghostMV( ghostM.begin(), ghostM.end());
        //overwrite out
        cusp::copy( f0, ghostMV);
        if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
        {
            dg::blas1::axpby( 2., left_, -1, ghostM);
        }
        if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
        {
            thrust::transform( left_.begin(), left_.end(), hm_.data().begin(), ghostP.begin(), thrust::multiplies<double>());
            dg::blas1::axpby( -1, ghostP, 1., ghostM);
        }
        cusp::blas::axpby(  ghostMV,  outV, ghostMV, 1.,-1.);
        dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
        cusp::blas::axpby(  ghostMV,  outV, outV, 1.,1.);
    }
}

///@endcond
}//namespace dg

