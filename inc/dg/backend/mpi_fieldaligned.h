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
 
   
struct ZShifter
{
    ZShifter(){}
    ZShifter( int size, MPI_Comm comm) 
    {
        number_ = size;
        comm_ = comm;
        sb_.resize( number_), rb_.resize( number_);
    }
    int number() const {return number_;}
    int size() const {return number_;}
    MPI_Comm communicator() const {return comm_;}
    void sendForward( HVec& sb, HVec& rb)const
    {
        int source, dest;
        MPI_Status status;
        MPI_Cart_shift( comm_, 2, +1, &source, &dest);
        MPI_Sendrecv(   sb.data(), number_, MPI_DOUBLE,  //sender
                        dest, 9,  //destination
                        rb.data(), number_, MPI_DOUBLE, //receiver
                        source, 9, //source
                        comm_, &status);
    }
    void sendBackward( HVec& sb, HVec& rb)const
    {
        int source, dest;
        MPI_Status status;
        MPI_Cart_shift( comm_, 2, -1, &source, &dest);
        MPI_Sendrecv(   sb.data(), number_, MPI_DOUBLE,  //sender
                        dest, 3,  //destination
                        rb.data(), number_, MPI_DOUBLE, //receiver
                        source, 3, //source
                        comm_, &status);
    }
    void sendForward( const thrust::device_vector<double>& sb, thrust::device_vector<double>& rb)
    {
        sb_ = sb; 
        sendForward( sb_, rb_);
        rb = rb_;
    }
    void sendBackward( const thrust::device_vector<double>& sb, thrust::device_vector<double>& rb)
    {
        sb_ = sb; 
        sendBackward( sb_, rb_);
        rb = rb_;
    }
    private:
    typedef thrust::host_vector<double> HVec;
    HVec sb_, rb_;
    int number_; //deepness, dimensions
    MPI_Comm comm_;

};



/**
 * @brief Class for the evaluation of a parallel derivative (MPI Version)
 *
 * @ingroup dz
 * @tparam Matrix The matrix class of the interpolation matrix
 * @tparam container The container-class to on which the interpolation matrix operates on (does not need to be dg::HVec)
 */
template <class LocalMatrix, class CommunicatorXY, class LocalContainer>
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
    std::vector<LocalContainer> tempXYplus_, tempXYminus_, temp_; 
    LocalContainer tempZ_;
    CommunicatorXY commXYplus_, commXYminus_;
    ZShifter  commZ_;
    LocalMatrix plus, minus; //interpolation matrices
    LocalMatrix plusT, minusT; //interpolation matrices
    //Communicator collM_, collP_;

    dg::FieldAligned<LocalMatrix, LocalContainer > dz_; //only stores 2D matrix so no memory pb.
};
///@cond
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class LocalMatrix, class CommunicatorXY, class LocalContainer>
template <class Field, class Limiter>
MPI_FieldAligned<LocalMatrix, CommunicatorXY, LocalContainer>::MPI_FieldAligned(Field field, const dg::MPI_Grid3d& grid, double eps, Limiter limit, dg::bc globalbcz, double deltaPhi ): 
    hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
    g_(grid), bcz_(grid.bcz()), 
    tempXYplus_(g_.Nz()), tempXYminus_(g_.Nz()),
    temp_(g_.Nz()),
    dz_(field, grid.global(), eps, limit, globalbcz, deltaPhi)
{
    //Resize vector to local 2D grid size
    MPI_Comm planeComm;
    int remain_dims[] = {true,true,false}; //true true false
    MPI_Cart_sub( grid.communicator(), remain_dims, &planeComm);
    dg::MPI_Grid2d g2d( g_.global().x0(), g_.global().x1(), g_.global().y0(), g_.global().y1(), g_.global().n(), g_.global().Nx(), g_.global().Ny(), g_.bcx(), g_.bcy(), planeComm);  
    unsigned size = g2d.size();
    limiter_ = dg::evaluate( limit, g2d.local());
    right_ = left_ = dg::evaluate( zero, g2d.local());
    ghostM.resize( size); ghostP.resize( size);
    //set up grid points as start for fieldline integrations 
    std::vector<thrust::host_vector<double> > y( 3);
    y[0] = dg::evaluate( dg::coo1, g2d.local());
    y[1] = dg::evaluate( dg::coo2, g2d.local());
    y[2] = dg::evaluate( dg::zero, g2d.local());//distance (not angle)
    //integrate to next z-planes
    std::vector<thrust::host_vector<double> > yp(y), ym(y); 
    if(deltaPhi<=0) deltaPhi = g_.hz();
    else assert( g_.Nz() == 1 || grid.hz()==deltaPhi);
    unsigned localsize = g2d.size();
#ifdef _OPENMP
#pragma omp parallel for shared(field)
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
    for( unsigned i=0; i<localsize; i++)
    {
        pids[i]  = g2d.pidOf( yp[0][i], yp[1][i]);
        if( pids[i]  == -1)
        {
            std::cerr << "ERROR: PID NOT FOUND!\n";
            return;
        }
    }

    CommunicatorXY cp( pids, g2d.communicator());
    commXYplus_ = cp;
    thrust::host_vector<double> pX = cp.collect( yp[0]),
                                pY = cp.collect( yp[1]);
    //construt interpolation matrix
    plus = dg::create::interpolation( pX, pY, g2d.local(), globalbcz); //inner points hopefully never lie exactly on local boundary
    cusp::transpose( plus, plusT);

    //do the same for the minus z-plane
    for( unsigned i=0; i<pids.size(); i++)
    {
        pids[i]  = g2d.pidOf( ym[0][i], ym[1][i]);
        if( pids[i] == -1)
        {
            std::cerr << "ERROR: PID NOT FOUND!\n";
            return;
        }
    }
    CommunicatorXY cm( pids, g2d.communicator());
    commXYminus_ = cm;
    pX = cm.collect( ym[0]);
    pY = cm.collect( ym[1]);
    minus = dg::create::interpolation( pX, pY, g2d.local(), globalbcz); //inner points hopefully never lie exactly on local boundary
    cusp::transpose( minus, minusT);
    //copy to device
    for( unsigned i=0; i<g_.Nz(); i++)
    {
        thrust::copy( yp[2].begin(), yp[2].end(), hp_.data().begin() + i*localsize);
        thrust::copy( ym[2].begin(), ym[2].end(), hm_.data().begin() + i*localsize);
    }
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);
    for( unsigned i=0; i<g_.Nz(); i++)
    {
        tempXYplus_[i].resize( commXYplus_.size());
        tempXYminus_[i].resize( commXYminus_.size());
        temp_[i].resize( localsize);
    }
    commZ_ = ZShifter( localsize, g_.communicator() );
    tempZ_.resize( commZ_.size());
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
MPI_Vector<container> MPI_FieldAligned<M,C, container>::evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const
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
void MPI_FieldAligned<M,C, container>::einsPlus( const MPI_Vector<container>& f, MPI_Vector<container>& fplus ) 
{
    //dg::blas2::detail::doSymv( plus, f, fplus, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    const container& in = f.data();
    container& out = fplus.data();
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];

    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( tempXYplus_[i0].begin(), tempXYplus_[i0].end() );
            cusp::multiply( plus, inV, tempV);
            //exchange data in XY
            commXYplus_.send_and_reduce( tempXYplus_[i0], temp_[i0]);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( plus, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<g_.Nz(); i0++)
    {
        int ip = i0 + 1;
        if( ip > (int)g_.Nz()-1) ip -= (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].begin() + size2d, out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        commZ_.sendBackward( temp_[0], tempZ_);
        thrust::copy( tempZ_.begin(), tempZ_.end(), out.begin() + (g_.Nz()-1)*size2d);
    }

    //make ghostcells in last plane
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
    const container& in = f.data();
    container& out = fminus.data();
    //dg::blas2::detail::doSymv( minus, f, fminus, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];
    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minus.num_cols, in.cbegin() + (i0+1)*minus.num_cols);
            View tempV( tempXYminus_[i0].begin(), tempXYminus_[i0].end());
            cusp::multiply( minus, inV, tempV);
            //exchange data in XY
            commXYminus_.send_and_reduce( tempXYminus_[i0], temp_[i0]);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minus.num_cols, in.cbegin() + (i0+1)*minus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( minus, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<g_.Nz(); i0++)
    {
        int ip = i0 -1;
        if( ip < 0) ip += (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].end(), out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        commZ_.sendForward( temp_[g_.Nz()-1], tempZ_);
        thrust::copy( tempZ_.begin(), tempZ_.end(), out.begin());
    }
    //make ghostcells in first plane
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
    //dg::blas2::detail::doSymv( minusT, f, fpe, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    const container& in = f.data();
    container& out = fpe.data();
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];

    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        //first exchange data in XY
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            thrust::copy( in.cbegin() + i0*size2d, in.cbegin() + (i0+1)*size2d, temp_[i0].begin());
            tempXYminus_[i0] = commXYminus_.collect( temp_[i0] );
            cView inV( tempXYminus_[i0].cbegin(), tempXYminus_[i0].cend() );
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( minusT, inV, tempV);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minusT.num_cols, in.cbegin() + (i0+1)*minusT.num_cols);
            View tempV( temp_[i0].begin() , temp_[i0].end() );
            cusp::multiply( minusT, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<g_.Nz(); i0++)
    {
        int ip = i0 + 1;
        if( ip > (int)g_.Nz()-1) ip -= (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].end(), out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        commZ_.sendBackward( temp_[0], tempZ_);
        thrust::copy( tempZ_.begin(), tempZ_.end(), out.begin() + (g_.Nz()-1)*size2d);
    }
    //make ghostcells in last plane
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
    //dg::blas2::detail::doSymv( plusT, f, fme, MPIMatrixTag(), MPIVectorTag(), MPIVectorTag());
    const container& in = f.data();
    container& out = fme.data();
    int size2d = g_.n()*g_.n()*g_.Nx()*g_.Ny();
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
        int sizeZ = dims[2];

    //1. compute 2d interpolation in every plane and store in temp_
    if( sizeXY != 1) //communication needed
    {
        //first exchange data in XY
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            thrust::copy( in.cbegin() + i0*size2d, in.cbegin() + (i0+1)*size2d, temp_[i0].begin());
            tempXYplus_[i0] = commXYplus_.collect( temp_[i0]);
            cView inV( tempXYplus_[i0].cbegin(), tempXYplus_[i0].cend() );
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( plusT, inV, tempV);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end());
            cusp::multiply( plusT, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<g_.Nz(); i0++)
    {
        int ip = i0 - 1;
        if( ip < 0 ) ip += (int)g_.Nz();
        thrust::copy( temp_[ip].begin(), temp_[ip].end(), out.begin() + i0*size2d);
    }
    if( sizeZ != 1)
    {
        commZ_.sendForward( temp_[g_.Nz()-1], tempZ_);
        thrust::copy( tempZ_.begin(), tempZ_.end(), out.begin());
    }
    //make ghostcells in first plane
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

