#pragma once
#include <cusp/transpose.h>
#include <cusp/csr_matrix.h>

#include "grid.h"
#include "../blas.h"
#include "interpolation.cuh"
#include "functions.h"

#include "../functors.h"
#include "../nullstelle.h"
#include "../runge_kutta.h"

namespace dg{

/**
 * @brief Default Limiter means there is a limiter everywhere
 */
struct DefaultLimiter
{
    /**
     * @brief return 1
     *
     * @param x x value
     * @param y y value
     * @return 1
     */
    double operator()(double x, double y) { return 1; }
};

/**
 * @brief No Limiter 
 */
struct NoLimiter
{
    /**
     * @brief return 0
     *
     * @param x x value
     * @param y y value
     * @return 0
     */
    double operator()(double x, double y) { return 0.; }
};

/**
 * @brief Integrate a field line to find whether the result lies inside or outside of the box
 *
 * @tparam Field Must be usable in the integrateRK4 function
 * @tparam Grid must provide 2d boundaries x0(), x1(), y0(), and y1()
 */
template < class Field, class Grid>
struct BoxIntegrator
{
    /**
     * @brief Construct from a given Field and Grid and accuracy
     *
     * @param field field must overload operator() with dg::HVec for three entries
     * @param g The 2d or 3d grid
     * @param eps the accuracy of the runge kutta integrator
     */
    BoxIntegrator( Field field, const Grid& g, double eps): field_(field), g_(g), coords_(3), coordsp_(3), eps_(eps) {}
    /**
     * @brief Set the starting coordinates for next field line integration
     *
     * @param coords the new coords (must have size = 3)
     */
    void set_coords( const thrust::host_vector<double>& coords){ coords_ = coords;}
    /**
     * @brief Integrate from 0 to deltaPhi
     *
     * @param deltaPhi upper integration boundary
     *
     * @return 1 if point is inside the box, -1 else
     */
    double operator()( double deltaPhi)
    {
        try{
            dg::integrateRK4( field_, coords_, coordsp_, deltaPhi, eps_);
        }
        catch( dg::NotANumber& exception) { return -1;}
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
    Grid g_;
    thrust::host_vector<double> coords_, coordsp_;
    double eps_;
};

/**
 * @brief Integrate one field line in a given box, Result is guaranteed to lie inside the box
 *
 * @tparam Field Must be usable in the integrateRK4 function
 * @tparam Grid must provide 2d boundaries x0(), x1(), y0(), and y1()
 * @param field The field to use
 * @param grid instance of the Grid class 
 * @param coords0 The initial condition
 * @param coords1 The resulting points (write only) guaranteed to lie inside the grid
 * @param phi1 The angle (read/write) contains maximum phi on input and resulting phi on output
 * @param eps error
 * @param globalbcz boundary condition  (DIR or NEU)
 */
template< class Field, class Grid>
void boxintegrator( Field& field, const Grid& grid, 
        const thrust::host_vector<double>& coords0, 
        thrust::host_vector<double>& coords1, 
        double& phi1, double eps, dg::bc globalbcz)
{
    dg::integrateRK4( field, coords0, coords1, phi1, eps); //+ integration
    if (    !(coords1[0] >= grid.x0() && coords1[0] <= grid.x1())
         || !(coords1[1] >= grid.y0() && coords1[1] <= grid.y1()))
    {
        if( globalbcz == dg::DIR)
        {
            BoxIntegrator<Field, Grid> boxy( field, grid, eps);
            boxy.set_coords( coords0); //nimm alte koordinaten
            if( phi1 > 0)
            {
                double dPhiMin = 0, dPhiMax = phi1;
                dg::bisection1d( boxy, dPhiMin, dPhiMax,eps); //suche 0 stelle 
                phi1 = (dPhiMin+dPhiMax)/2.;
                dg::integrateRK4( field, coords0, coords1, dPhiMax, eps); //integriere bis über 0 stelle raus damit unten Wert neu gesetzt wird
            }
            else
            {
                double dPhiMin = phi1, dPhiMax = 0;
                dg::bisection1d( boxy, dPhiMin, dPhiMax,eps);
                phi1 = (dPhiMin+dPhiMax)/2.;
                dg::integrateRK4( field, coords0, coords1, dPhiMin, eps);
            }
            if (!(coords1[0] > grid.x0())) { coords1[0]=grid.x0();}
            if (!(coords1[0] < grid.x1())) { coords1[0]=grid.x1();}
            if (!(coords1[1] > grid.y0())) { coords1[1]=grid.y0();}
            if (!(coords1[1] < grid.y1())) { coords1[1]=grid.y1();}
        }
        else if (globalbcz == dg::NEU )
        {
             coords1[0] = coords0[0]; coords1[1] = coords0[1];  
        }
        else if (globalbcz == DIR_NEU )std::cerr << "DIR_NEU NOT IMPLEMENTED "<<std::endl;
        else if (globalbcz == NEU_DIR )std::cerr << "NEU_DIR NOT IMPLEMENTED "<<std::endl;
        else if (globalbcz == dg::PER )std::cerr << "PER NOT IMPLEMENTED "<<std::endl;
    }
}

template<class CuspMatrix>
struct PlaneMultiplier
{
    PlaneMultiplier( const CuspMatrix& m, unsigned Nz, int offset):Nz_(Nz), m_(m), offset_(offset){
    }
    PlaneMultiplier(){}
    template <class Vector>
    void symv( const Vector& in, Vector& out) const
    {
        std::cout << in.size()<< " " << out.size()<< " "<<m_.num_cols<<" "<<m_.num_rows<<" "<<Nz_<<std::endl;
        assert( in.size() == m_.num_cols*Nz_);
        assert( out.size() == m_.num_rows*Nz_);
        typedef cusp::array1d_view< typename Vector::iterator> View;
        typedef cusp::array1d_view< typename Vector::const_iterator> cView;
        for( int i0=0; i0<Nz_; i0++)
        {
            int ip = i0 + offset_;
            if( ip > (int)Nz_-1) ip -= (int)Nz_;
            if( ip < 0) ip += (int)Nz_;

            cView inV( in.cbegin() + ip*m_.num_cols, in.cbegin() + (ip+1)*m_.num_cols);
            View outV( out.begin() + i0*m_.num_rows, out.begin() + (i0+1)*m_.num_rows);
            cusp::multiply( m_, inV, outV);
        }
    }
    const CuspMatrix& matrix() const {return m_;}
    private:
    unsigned Nz_;
    CuspMatrix m_;
    int offset_;
};

template<class M>
struct MatrixTraits<PlaneMultiplier<M> >
{
    typedef double value_type;
    typedef SelfMadeMatrixTag matrix_category;
};
////////////////////////////////////FieldAlignedCLASS////////////////////////////////////////////
/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
cylindrical coordinates
* @ingroup ds
* @tparam Matrix The matrix class of the interpolation matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class Matrix = cusp::csr_matrix<int, double, cusp::device_memory>, class container=thrust::device_vector<double> >
struct FieldAligned
{

    /**
    * @brief Construct from a field and a grid
    *
    * @tparam Field The Fieldlines to be integrated: 
        Has to provide void operator()( const std::vector<dg::HVec>&, std::vector<dg::HVec>&) 
        where the first index is R, the second Z and the last s (the length of the field line)
    * @tparam Limiter Class that can be evaluated on a 2d grid, returns 1 if there
        is a limiter and 0 if there isn't. 
        If a field line crosses the limiter in the plane \f$ \phi=0\f$ then the limiter boundary conditions apply. 
    * @param field The field to integrate
    * @param grid The grid on which to operate
    * @param eps Desired accuracy of runge kutta
    * @param limit Instance of the limiter class (Default is a limiter everywhere, 
        note that if bcz is periodic it doesn't matter if there is a limiter or not)
    * @param globalbcz Choose NEU or DIR. Defines BC in parallel on bounding box
    * @param deltaPhi Is either <0 (then it's ignored), may differ from hz() only if Nz() == 1
    * @note If there is a limiter, the boundary condition on the first/last plane is set 
        by the bcz variable from the grid and can be changed by the set_boundaries function. 
        If there is no limiter the boundary condition is periodic.
    */
    template <class Field, class Limiter>
    FieldAligned(Field field, const dg::Grid3d<double>& grid, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR, double deltaPhi = -1);


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
        left_ = dg::evaluate( dg::CONSTANT(left), g2d);
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
    void set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right);

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
    container evaluate( BinaryOp f, unsigned plane=0) const;
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
    container evaluate( BinaryOp f, UnaryOp g, unsigned p0, unsigned rounds) const;

    /**
    * @brief Applies the interpolation to the next planes 
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsPlus( const container& in, container& out);
    /**
    * @brief Applies the interpolation to the previous planes
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsMinus( const container& in, container& out);
    /**
    * @brief Applies the transposed interpolation to the previous plane 
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsPlusT( const container& in, container& out);
    /**
    * @brief Applies the transposed interpolation to the next plane 
    *
    * @param in input 
    * @param out output may not equal intpu
    */
    void einsMinusT( const container& in, container& out);

    /**
    * @brief hz is the distance between the plus and minus planes
    *
    * @return three-dimensional vector
    */
    const container& hz()const {return hz_;}
    /**
    * @brief hp is the distance between the plus and current planes
    *
    * @return three-dimensional vector
    */
    const container& hp()const {return hp_;}
    /**
    * @brief hm is the distance between the current and minus planes
    *
    * @return three-dimensional vector
    */
    const container& hm()const {return hm_;}
    /**
    * @brief Access the underlying grid
    *
    * @return the grid
    */
    const Grid3d<double>& grid() const{return g_;}
    private:
    typedef cusp::array1d_view< typename container::iterator> View;
    typedef cusp::array1d_view< typename container::const_iterator> cView;
    PlaneMultiplier<Matrix> plus, minus, plusT, minusT; //interpolation matrices
    container hz_, hp_,hm_, ghostM, ghostP;
    dg::Grid3d<double> g_;
    dg::bc bcz_;
    container left_, right_;
    container limiter_;
};

///@cond 
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class M, class container>
template <class Field, class Limiter>
FieldAligned<M,container>::FieldAligned(Field field, const dg::Grid3d<double>& grid, double eps, Limiter limit, dg::bc globalbcz, double deltaPhi):
        hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
        g_(grid), bcz_(grid.bcz())
{
    //Resize vector to 2D grid size
    dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());  
    unsigned size = g2d.size();
    limiter_ = dg::evaluate( limit, g2d);
    right_ = left_ = dg::evaluate( zero, g2d);
    ghostM.resize( size); ghostP.resize( size);
    //Set starting points
    std::vector<thrust::host_vector<double> > y( 3, dg::evaluate( dg::coo1, g2d)), yp(y), ym(y);
    y[1] = dg::evaluate( dg::coo2, g2d);
    y[2] = dg::evaluate( dg::zero, g2d);
  
//     integrate field lines for all points
    
    if( deltaPhi <=0) deltaPhi = g_.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
#ifdef _OPENMP
#pragma omp parallel for shared(field)
#endif //_OPENMP
    for( unsigned i=0; i<size; i++)
    {
        thrust::host_vector<double> coords(3), coordsP(3), coordsM(3);
        coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i];
        double phi1 = deltaPhi;
        boxintegrator( field, g2d, coords, coordsP, phi1, eps, globalbcz);
        phi1 =  - deltaPhi;
        boxintegrator( field, g2d, coords, coordsM, phi1, eps, globalbcz);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }
    M plusPlane  = dg::create::interpolation( yp[0], yp[1], g2d, globalbcz);
    M minusPlane = dg::create::interpolation( ym[0], ym[1], g2d, globalbcz);
// //     Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix and MPI_Matrix lacks of transpose function!!!
    M plusTPlane, minusTPlane;
    cusp::transpose( plusPlane, plusTPlane);
    cusp::transpose( minusPlane, minusTPlane);     
    plus = PlaneMultiplier<M>( plusPlane, grid.Nz(), +1);
    minus = PlaneMultiplier<M>( minusPlane, grid.Nz(), -1);
    plusT = PlaneMultiplier<M>( plusTPlane, grid.Nz(), -1);
    minusT = PlaneMultiplier<M>( minusTPlane, grid.Nz(), +1);
//     copy into h vectors
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( yp[2].begin(), yp[2].end(), hp_.begin() + i*g2d.size());
        thrust::copy( ym[2].begin(), ym[2].end(), hm_.begin() + i*g2d.size());        
    }
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);    //
 
}

template<class M, class container>
void FieldAligned<M,container>::set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
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

template< class M, class container>
template< class BinaryOp>
container FieldAligned<M,container>::evaluate( BinaryOp binary, unsigned p0) const
{

    assert( p0 < g_.Nz() && g_.Nz() > 1);
    const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    container vec2d = dg::evaluate( binary, g2d);
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
        cusp::multiply( minus.matrix(), fm, f0 );
    }
    for( int i0=p0-1; i0>=0; i0--)
    {
        unsigned ip = i0+1;
        View fp( vec3d.begin() + ip*g2d.size(), vec3d.begin() + (ip+1)*g2d.size());
        View f0( vec3d.begin() + i0*g2d.size(), vec3d.begin() + (i0+1)*g2d.size());
        cusp::multiply( plus.matrix(), fp, f0 );
    }
    return vec3d;
}

template< class M, class container>
template< class BinaryOp, class UnaryOp>
container FieldAligned<M,container>::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{

    assert( g_.Nz() > 1);
    container vec3d = evaluate( binary, p0);
    const dg::Grid2d<double> g2d( g_.x0(), g_.x1(), g_.y0(), g_.y1(), g_.n(), g_.Nx(), g_.Ny());
    //scal
    for( unsigned i=0; i<g_.Nz(); i++)
    {
        View f0( vec3d.begin() + i*g2d.size(), vec3d.begin() + (i+1)*g2d.size());
        cusp::blas::scal(f0, unary( g_.z0() + (double)(i+0.5)*g_.hz() ));
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
        cusp::multiply( minus.matrix(), fm, f0 );
        cusp::blas::scal( f0, unary( g_.z0() + (double)(k*g_.Nz()+i0+0.5)*g_.hz() ) );
        }
        for( int i0=g_.Nz()-1; i0>=0; i0--)
        {
        int ip = i0==g_.Nz()-1?0:i0+1;
        int k0 = k;
        int km = i0==g_.Nz()-1?k-1:k;
        View fp( vec4dM[km].begin() + ip*g2d.size(), vec4dM[km].begin() + (ip+1)*g2d.size());
        View f0( vec4dM[k0].begin() + i0*g2d.size(), vec4dM[k0].begin() + (i0+1)*g2d.size());
        cusp::multiply( plus.matrix(), fp, f0 );
        cusp::blas::scal( f0, unary( g_.z0() - (double)(k*g_.Nz()-0.5-i0)*g_.hz() ) );
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


template< class M, class container>
void FieldAligned<M, container>::einsPlus( const container& f, container& fpe)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    dg::blas2::detail::doSymv( plus, f, fpe, SelfMadeMatrixTag(), ThrustVectorTag(), ThrustVectorTag());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fP( fpe.begin() + i0*size, fpe.begin() + (i0+1)*size);
        //cusp::multiply( plus, fp, fP);
        //make ghostcells i.e. modify fpe in the limiter region
        if( i0==g_.Nz()-1 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            {
                cusp::blas::axpby( rightV, f0, ghostPV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
            {
                thrust::transform( right_.begin(), right_.end(),  hp_.begin(), ghostM.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostMV, f0, ghostPV, 1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
            cusp::blas::axpby( ghostPV, fP, ghostPV, 1., -1.);
            dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
            cusp::blas::axpby(  ghostPV, fP, fP, 1.,1.);
        }
    }
}

template< class M, class container>
void FieldAligned<M, container>::einsMinus( const container& f, container& fme)
{
    //note that thrust functions don't work on views
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    dg::blas2::detail::doSymv( minus, f, fme, SelfMadeMatrixTag(), ThrustVectorTag(), ThrustVectorTag());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fM( fme.begin() + i0*size, fme.begin() + (i0+1)*size);
        //cusp::multiply( minus, fm, fM );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
            {
                cusp::blas::axpby( leftV,  f0, ghostMV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
            {
                thrust::transform( left_.begin(), left_.end(),  hm_.begin(), ghostP.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostPV, f0, ghostMV, -1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fme
            cusp::blas::axpby( ghostMV, fM, ghostMV, 1., -1.);
            dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
            cusp::blas::axpby( ghostMV, fM, fM, 1., 1.);

        }
    }
}

template< class M, class container>
void FieldAligned<M, container>::einsMinusT( const container& f, container& fpe)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    dg::blas2::detail::doSymv( minusT, f, fpe, SelfMadeMatrixTag(), ThrustVectorTag(), ThrustVectorTag());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fP( fpe.begin() + i0*size, fpe.begin() + (i0+1)*size);
        //cusp::multiply( minusT, fp, fP );
        //make ghostcells i.e. modify fpe in the limiter region
        if( i0==g_.Nz()-1 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::NEU_DIR)
            {
                cusp::blas::axpby( rightV, f0, ghostPV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::DIR_NEU)
            {
                thrust::transform( right_.begin(), right_.end(),  hp_.begin(), ghostM.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostMV, f0, ghostPV, 1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fpe
            cusp::blas::axpby( ghostPV, fP, ghostPV, 1., -1.);
            dg::blas1::pointwiseDot( limiter_, ghostP, ghostP);
            cusp::blas::axpby(  ghostPV, fP, fP, 1.,1.);
        }

    }
}

template< class M, class container>
void FieldAligned<M, container>::einsPlusT( const container& f, container& fme)
{
    //note that thrust functions don't work on views
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    dg::blas2::detail::doSymv( plusT, f, fme, SelfMadeMatrixTag(), ThrustVectorTag(), ThrustVectorTag());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fM( fme.begin() + i0*size, fme.begin() + (i0+1)*size);
        //cusp::multiply( plusT, fm, fM );
        //make ghostcells
        if( i0==0 && bcz_ != dg::PER)
        {
            if( bcz_ == dg::DIR || bcz_ == dg::DIR_NEU)
            {
                cusp::blas::axpby( leftV,  f0, ghostMV, 2., -1.);
            }
            if( bcz_ == dg::NEU || bcz_ == dg::NEU_DIR)
            {
                thrust::transform( left_.begin(), left_.end(),  hm_.begin(), ghostP.begin(), thrust::multiplies<double>());
                cusp::blas::axpby( ghostPV, f0, ghostMV, -1., 1.);
            }
            //interlay ghostcells with periodic cells: L*g + (1-L)*fme
            cusp::blas::axpby( ghostMV, fM, ghostMV, 1., -1.);
            dg::blas1::pointwiseDot( limiter_, ghostM, ghostM);
            cusp::blas::axpby( ghostMV, fM, fM, 1., 1.);

        }
    }
}

///@endcond 


}//namespace dg

