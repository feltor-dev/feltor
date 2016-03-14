#pragma once

#include "fieldaligned.h"
#include "../backend/grid.h"
#include "../backend/mpi_evaluation.h"
#include "../backend/mpi_matrix.h"
#include "../backend/mpi_matrix_blas.h"
#include "../backend/mpi_collective.h"
#include "../backend/mpi_grid.h"
#include "../backend/interpolation.cuh"
#include "../backend/functions.h"
#include "../runge_kutta.h"

namespace dg{
 
///@cond

/**
 * @brief Class to shift values in the z - direction 
 */
struct ZShifter
{
    ZShifter(){}
    /**
     * @brief Constructor
     *
     * @param size number of elements to exchange between processes
     * @param comm the communicator (cartesian)
     */
    ZShifter( int size, MPI_Comm comm) 
    {
        number_ = size;
        comm_ = comm;
        sb_.resize( number_), rb_.resize( number_);
    }
    int number() const {return number_;}
    int size() const {return number_;}
    MPI_Comm communicator() const {return comm_;}
    //host and device versions
    template<class container>
    void sendForward( const container& sb, container& rb)
    {
        dg::blas1::transfer( sb, sb_);
        sendForward_( sb_, rb_);
        dg::blas1::transfer( rb_, rb);
    }
    template<class container>
    void sendBackward( const container& sb, container& rb)
    {
        dg::blas1::transfer( sb, sb_);
        sendBackward_( sb_, rb_);
        dg::blas1::transfer( rb_, rb);
    }
    private:
    void sendForward_( HVec& sb, HVec& rb)const //send to next plane
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
    void sendBackward_( HVec& sb, HVec& rb)const //send to previous plane
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
    typedef thrust::host_vector<double> HVec;
    HVec sb_, rb_;
    int number_; //deepness, dimensions
    MPI_Comm comm_;

};

///@endcond



/**
 * @brief Class for the evaluation of a parallel derivative (MPI Version)
 *
 * @ingroup dz
 * @tparam LocalMatrix The matrix class of the interpolation matrix
 * @tparam Communicator The communicator used to exchange data in the RZ planes
 * @tparam LocalContainer The container-class to on which the interpolation matrix operates on (does not need to be dg::HVec)
 */
template <class Geometry, class LocalMatrix, class Communicator, class LocalContainer>
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
    MPI_FieldAligned(Field field, Geometry grid, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR, double deltaPhi = -1 );

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
    const Geometry& grid() const{return g_;}
  private:
    typedef cusp::array1d_view< typename LocalContainer::iterator> View;
    typedef cusp::array1d_view< typename LocalContainer::const_iterator> cView;
    MPI_Vector<LocalContainer> hz_, hp_, hm_; 
    LocalContainer ghostM, ghostP;
    Geometry g_;
    dg::bc bcz_;
    LocalContainer left_, right_;
    LocalContainer limiter_;
    std::vector<LocalContainer> tempXYplus_, tempXYminus_, temp_; 
    LocalContainer tempZ_;
    Communicator commXYplus_, commXYminus_;
    ZShifter  commZ_;
    LocalMatrix plus, minus; //interpolation matrices
    LocalMatrix plusT, minusT; //interpolation matrices
};
///@cond
//////////////////////////////////////DEFINITIONS/////////////////////////////////////
template<class MPIGeometry, class LocalMatrix, class CommunicatorXY, class LocalContainer>
template <class Field, class Limiter>
MPI_FieldAligned<MPIGeometry, LocalMatrix, CommunicatorXY, LocalContainer>::MPI_FieldAligned(Field field, MPIGeometry grid, double eps, Limiter limit, dg::bc globalbcz, double deltaPhi ): 
    hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
    g_(grid), bcz_(grid.bcz()), 
    tempXYplus_(g_.Nz()), tempXYminus_(g_.Nz()), temp_(g_.Nz())
{
    //create communicator with all processes in plane
    typename MPIGeometry::perpendicular_grid g2d = grid.perp_grid();
    unsigned localsize = g2d.size();
    limiter_ = dg::evaluate( limit, g2d.local());
    right_ = left_ = dg::evaluate( zero, g2d.local());
    ghostM.resize( localsize); ghostP.resize( localsize);
    //set up grid points as start for fieldline integrations 
    std::vector<MPI_Vector<thrust::host_vector<double> > > y( 5, dg::evaluate(dg::zero, g2d));
    y[0] = dg::evaluate( dg::coo1, g2d);
    y[1] = dg::evaluate( dg::coo2, g2d);
    y[2] = dg::evaluate( dg::zero, g2d);//distance (not angle)
    y[3] = dg::pullback( dg::coo1, g2d);
    y[4] = dg::pullback( dg::coo2, g2d);
    //integrate to next z-planes
    std::vector<thrust::host_vector<double> > yp(3, y[0].data()), ym(yp); 
    if(deltaPhi<=0) deltaPhi = grid.hz();
    else assert( g_.Nz() == 1 || grid.hz()==deltaPhi);
#ifdef _OPENMP
#pragma omp parallel for shared(field)
#endif //_OPENMP
    for( unsigned i=0; i<localsize; i++)
    {
        thrust::host_vector<double> coords(5), coordsP(5), coordsM(5);
        coords[0] = y[0].data()[i], coords[1] = y[1].data()[i], coords[2] = y[2].data()[i], coords[3] = y[3].data()[i], coords[4] = y[4].data()[i];
        double phi1 = deltaPhi;
        boxintegrator( field, g2d.global(), coords, coordsP, phi1, eps, globalbcz);
        phi1 = -deltaPhi;
        boxintegrator( field, g2d.global(), coords, coordsM, phi1, eps, globalbcz);
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
    thrust::host_vector<double> pX, pY;
    dg::blas1::transfer( cp.collect( yp[0]), pX);
    dg::blas1::transfer( cp.collect( yp[1]), pY);

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
    dg::blas1::transfer( cm.collect( ym[0]), pX);
    dg::blas1::transfer( cm.collect( ym[1]), pY);
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

template<class G, class M, class C, class container>
template< class BinaryOp>
MPI_Vector<container> MPI_FieldAligned<G,M,C,container>::evaluate( BinaryOp binary, unsigned p0) const
{
    return evaluate( binary, dg::CONSTANT(1), p0, 0);
}

template<class G, class M, class C, class container>
template< class BinaryOp, class UnaryOp>
MPI_Vector<container> MPI_FieldAligned<G,M,C, container>::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( g_.Nz() > 1);
    assert( p0 < g_.global().Nz());
    const typename G::perpendicular_grid g2d = g_.perp_grid();
    MPI_Vector<container> init2d = dg::pullback( binary, g2d); 
    container temp(init2d.data()), tempP(init2d.data()), tempM(init2d.data());
    MPI_Vector<container> vec3d = dg::evaluate( dg::zero, g_);
    std::vector<container>  plus2d( g_.global().Nz(), (container)dg::evaluate(dg::zero, g2d.local()) ), minus2d( plus2d), result( plus2d);
    container tXYplus( tempXYplus_[0]), tXYminus( tempXYminus_[0]);
    unsigned turns = rounds; 
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
        int dims[3], periods[3], coords[3];
        MPI_Cart_get( g_.communicator(), 3, dims, periods, coords);
        int sizeXY = dims[0]*dims[1];
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<g_.global().Nz(); i0++)
        {
            dg::blas1::copy( init2d.data(), tempP);
            dg::blas1::copy( init2d.data(), tempM);
            unsigned rep = i0 + r*g_.global().Nz(); 
            for(unsigned k=0; k<rep; k++)
            {
                if( sizeXY != 1){
                    dg::blas2::symv( plus, tempP, tXYplus);
                    commXYplus_.send_and_reduce( tXYplus, temp);
                }
                else
                    dg::blas2::symv( plus, tempP, temp);
                temp.swap( tempP);
                if( sizeXY != 1){
                    dg::blas2::symv( minus, tempM, tXYminus);
                    commXYminus_.send_and_reduce( tXYminus, temp);
                }
                else
                    dg::blas2::symv( minus, tempM, temp);
                temp.swap( tempM);
            }
            dg::blas1::scal( tempP, unary(  (double)rep*g_.hz() ) );
            dg::blas1::scal( tempM, unary( -(double)rep*g_.hz() ) );
            dg::blas1::axpby( 1., tempP, 1., plus2d[i0]);
            dg::blas1::axpby( 1., tempM, 1., minus2d[i0]);
        }
    //now we have the plus and the minus filaments
    if( rounds == 0) //there is a limiter
    {
        for( unsigned i0=0; i0<g_.Nz(); i0++)
        {
            int idx = (int)(i0+coords[2]*g_.Nz())  - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].begin(), result[i0].end(), vec3d.data().begin() + i0*g2d.size());
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<g_.global().Nz(); i0++)
        {
            //int idx = (int)(i0+coords[2]*g_.Nz());
            unsigned revi0 = (g_.global().Nz() - i0)%g_.global().Nz(); //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d.data(), 1., result[0]);
        for(unsigned i0=0; i0<g_.Nz(); i0++)
        {
            int idx = ((int)i0 + coords[2]*g_.Nz() -(int)p0 + g_.global().Nz())%g_.global().Nz(); //shift index
            thrust::copy( result[idx].begin(), result[idx].end(), vec3d.data().begin() + i0*g2d.size());
        }
    }
    return vec3d;
}

template<class G, class M, class C, class container>
void MPI_FieldAligned<G,M,C, container>::einsPlus( const MPI_Vector<container>& f, MPI_Vector<container>& fplus ) 
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
        for( int i0=0; i0<(int)g_.Nz(); i0++)
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
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( plus, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
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

template<class G,class M, class C, class container>
void MPI_FieldAligned<G,M,C,container>::einsMinus( const MPI_Vector<container>& f, MPI_Vector<container>& fminus ) 
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
        for( int i0=0; i0<(int)g_.Nz(); i0++)
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
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minus.num_cols, in.cbegin() + (i0+1)*minus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( minus, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
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
template< class G, class M, class C, class container>
void MPI_FieldAligned<G,M,C,container>::einsMinusT( const MPI_Vector<container>& f, MPI_Vector<container>& fpe)
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
        for( int i0=0; i0<(int)g_.Nz(); i0++)
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
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*minusT.num_cols, in.cbegin() + (i0+1)*minusT.num_cols);
            View tempV( temp_[i0].begin() , temp_[i0].end() );
            cusp::multiply( minusT, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
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
template< class G,class M, class C, class container>
void MPI_FieldAligned<G,M,C,container>::einsPlusT( const MPI_Vector<container>& f, MPI_Vector<container>& fme)
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
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            //first exchange data in XY
            thrust::copy( in.cbegin() + i0*size2d, in.cbegin() + (i0+1)*size2d, temp_[i0].begin());
            tempXYplus_[i0] = commXYplus_.collect( temp_[i0]);
            cView inV( tempXYplus_[i0].cbegin(), tempXYplus_[i0].cend() );
            View tempV( temp_[i0].begin(), temp_[i0].end() );
            cusp::multiply( plusT, inV, tempV);
        }
    }
    else //directly compute in temp_
    {
        for( int i0=0; i0<(int)g_.Nz(); i0++)
        {
            cView inV( in.cbegin() + i0*plus.num_cols, in.cbegin() + (i0+1)*plus.num_cols);
            View tempV( temp_[i0].begin(), temp_[i0].end());
            cusp::multiply( plusT, inV, tempV);
        }
    }
    //2. reorder results and communicate halo in z
    for( int i0=0; i0<(int)g_.Nz(); i0++)
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

