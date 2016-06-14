#pragma once
#include <cmath>
#include <cusp/transpose.h>
#include <cusp/csr_matrix.h>

#include "../backend/grid.h"
#include "../blas.h"
#include "../backend/interpolation.cuh"
#include "../backend/functions.h"

#include "../functors.h"
#include "../nullstelle.h"
#include "../runge_kutta.h"

namespace dg{

/**
 * @brief With the Default field ds becomes a dz
 */
struct DefaultField
{
    void operator()( const dg::HVec& y, dg::HVec& yp)
    {
        yp[0] = yp[1] = 0;
        yp[2] = 1.;
    }
    double operator()( double x, double y)
    {
        return 1.;
    }
    double operator()( double x, double y, double z)
    {
        return 1.;
    }

};

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
 * @brief Integrates the differential equation using a s stage RK scheme and a rudimentary stepsize-control
 *
 * @ingroup algorithms
 * Doubles the number of timesteps until the desired accuracy is reached
 * Checks for NaN errors on the way and if the fieldline diverges. The error is computed in the first three vector elements (x,y,s)
 * @tparam RHS The right-hand side class
 * @tparam Vector Vector-class (needs to be copyable)
 * @param rhs The right-hand-side
 * @param begin initial condition (size 3)
 * @param end (write-only) contains solution on output
 * @param T_max final time
 * @param eps_abs desired absolute accuracy
 */
template< class RHS, class Vector, unsigned s>
void integrateRK(RHS& rhs, const Vector& begin, Vector& end, double T_max, double eps_abs )
{
    RK_classic<s, Vector > rk( begin); 
    Vector old_end(begin), temp(begin),diffm(begin);
    end = begin;
    if( T_max == 0) return;
    double dt = T_max/10;
    int NT = 10;
    double error = 1e10;
 
    while( error > eps_abs && NT < pow( 2, 18) )
    {
        dt /= 2.;
        NT *= 2;
        end = begin;

        int i=0;
        while (i<NT && NT < pow( 2, 18))
        {
            rk( rhs, end, temp, dt); 
            end.swap( temp); //end is one step further 
            dg::blas1::axpby( 1., end, -1., old_end,diffm); //abs error=oldend = end-oldend
            double temp = diffm[0]*diffm[0]+diffm[1]*diffm[1]+diffm[2]*diffm[2];
            error = sqrt( temp );
            if ( isnan(end[0]) || isnan(end[1]) || isnan(end[2])        ) 
            {
                dt /= 2.;
                NT *= 2;
                i=-1;
                end = begin;
                #ifdef DG_DEBUG
                    std::cout << "---------Got NaN -> choosing smaller step size and redo integration" << " NT "<<NT<<" dt "<<dt<< std::endl;
                #endif
            }
            //if new integrated point outside domain
            //if ((1e-5 > end[0]  ) || (1e10 < end[0])  ||(-1e10  > end[1]  ) || (1e10 < end[1])||(-1e10 > end[2]  ) || (1e10 < end[2])  )
            if( (end[3] < 1e-5) || end[3]*end[3] > 1e10 ||end[1]*end[1] > 1e10 ||end[2]*end[2] > 1e10 ||(end[4]*end[4] > 1e10) )
            {
                error = eps_abs/10;
                #ifdef DG_DEBUG
                std::cerr << "---------Point outside box -> stop integration" << std::endl; 
                #endif
                i=NT;
            }
            i++;
        }  


        old_end = end;
#ifdef DG_DEBUG
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        std::cout << "NT "<<NT<<" dt "<<dt<<" error "<<error<<"\n";
#endif //DG_DEBUG
    }

    if( std::isnan( error) )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is NAN! "<<std::endl;
        throw NotANumber();
    }
    if( error > eps_abs )
    {
        std::cerr << "ATTENTION: Runge Kutta failed to converge. Error is "<<error<<std::endl;
        throw Fail( eps_abs);
    }


}

template< class RHS, class Vector>
void integrateRK4(RHS& rhs, const Vector& begin, Vector& end, double T_max, double eps_abs )
{
    integrateRK<RHS, Vector, 4>( rhs, begin, end, T_max, eps_abs);
}

template< class RHS, class Vector>
void integrateRK6(RHS& rhs, const Vector& begin, Vector& end, double T_max, double eps_abs )
{
    integrateRK<RHS, Vector, 6>( rhs, begin, end, T_max, eps_abs);
}
template< class RHS, class Vector>
void integrateRK17(RHS& rhs, const Vector& begin, Vector& end, double T_max, double eps_abs )
{
    integrateRK<RHS, Vector, 17>( rhs, begin, end, T_max, eps_abs);
}

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
    //First catch periodic domain
    grid.shift_topologic( coords0[0], coords0[1], coords1[0], coords1[1]);
    if ( !grid.contains( coords1[0], coords1[1]))   //Punkt liegt immer noch außerhalb 
    {
#ifdef DG_DEBUG
        std::cerr << "point "<<coords1[0]<<" "<<coords1[1]<<" "<<coords1[3]<<" "<<coords1[4]<<" is somewhere else!\n";
#endif //DG_DEBUG
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
////////////////////////////////////FieldAlignedCLASS////////////////////////////////////////////
/**
* @brief Class for the evaluation of a parallel derivative
*
* This class discretizes the operators \f$ \nabla_\parallel = 
\mathbf{b}\cdot \nabla = b_R\partial_R + b_Z\partial_Z + b_\phi\partial_\phi \f$, \f$\nabla_\parallel^\dagger\f$ and \f$\Delta_\parallel=\nabla_\parallel^\dagger\cdot\nabla_\parallel\f$ in
cylindrical coordinates
* @ingroup algorithms
* @tparam Matrix The matrix class of the interpolation matrix
* @tparam container The container-class on which the interpolation matrix operates on (does not need to be dg::HVec)
*/
template< class Geometry, class Matrix, class container >
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
    FieldAligned(Field field, Geometry grid, double eps = 1e-4, Limiter limit = DefaultLimiter(), dg::bc globalbcz = dg::DIR, double deltaPhi = -1);


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
     * @note g is evaluated such that p0 corresponds to z=0, p0+1 corresponds to z=hz, p0-1 to z=-hz, ...
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
    const Geometry& grid() const{return g_;}
    private:
    typedef cusp::array1d_view< typename container::iterator> View;
    typedef cusp::array1d_view< typename container::const_iterator> cView;
    Matrix plus, minus, plusT, minusT; //interpolation matrices
    container hz_, hp_,hm_, ghostM, ghostP;
    Geometry g_;
    dg::bc bcz_;
    container left_, right_;
    container limiter_;
};

///@cond 
////////////////////////////////////DEFINITIONS////////////////////////////////////////

template<class Geometry, class M, class container>
template <class Field, class Limiter>
FieldAligned<Geometry, M,container>::FieldAligned(Field field, Geometry grid, double eps, Limiter limit, dg::bc globalbcz, double deltaPhi):
        hz_( dg::evaluate( dg::zero, grid)), hp_( hz_), hm_( hz_), 
        g_(grid), bcz_(grid.bcz())
{
    //Resize vector to 2D grid size

    typename Geometry::perpendicular_grid g2d = grid.perp_grid( );
    unsigned size = g2d.size();
    limiter_ = dg::evaluate( limit, g2d);
    right_ = left_ = dg::evaluate( zero, g2d);
    ghostM.resize( size); ghostP.resize( size);
    //Set starting points
    std::vector<thrust::host_vector<double> > y( 5, dg::evaluate( dg::coo1, g2d)); // x
    y[1] = dg::evaluate( dg::coo2, g2d); //y
    y[2] = dg::evaluate( dg::zero, g2d);
    y[3] = dg::pullback( dg::coo1, g2d); //R
    y[4] = dg::pullback( dg::coo2, g2d); //Z
    //integrate field lines for all points
    std::vector<thrust::host_vector<double> > yp( 3, dg::evaluate(dg::zero, g2d)), ym(yp); 
    if( deltaPhi <=0) deltaPhi = grid.hz();
    else assert( grid.Nz() == 1 || grid.hz()==deltaPhi);
#ifdef _OPENMP
#pragma omp parallel for shared(field)
#endif //_OPENMP
    for( unsigned i=0; i<size; i++)
    {
        thrust::host_vector<double> coords(5), coordsP(5), coordsM(5);
        coords[0] = y[0][i], coords[1] = y[1][i], coords[2] = y[2][i], coords[3] = y[3][i], coords[4] = y[4][i]; //x,y,s,R,Z
        double phi1 = deltaPhi;
        boxintegrator( field, g2d, coords, coordsP, phi1, eps, globalbcz);
        phi1 =  - deltaPhi;
        boxintegrator( field, g2d, coords, coordsM, phi1, eps, globalbcz);
        yp[0][i] = coordsP[0], yp[1][i] = coordsP[1], yp[2][i] = coordsP[2];
        ym[0][i] = coordsM[0], ym[1][i] = coordsM[1], ym[2][i] = coordsM[2];
    }
    //fange Periodische RB ab
    plus  = dg::create::interpolation( yp[0], yp[1], g2d, globalbcz);
    minus = dg::create::interpolation( ym[0], ym[1], g2d, globalbcz);
// //     Transposed matrices work only for csr_matrix due to bad matrix form for ell_matrix and MPI_Matrix lacks of transpose function!!!
    cusp::transpose( plus, plusT);
    cusp::transpose( minus, minusT);     
//     copy into h vectors
    for( unsigned i=0; i<grid.Nz(); i++)
    {
        thrust::copy( yp[2].begin(), yp[2].end(), hp_.begin() + i*g2d.size());
        thrust::copy( ym[2].begin(), ym[2].end(), hm_.begin() + i*g2d.size());
    }
    dg::blas1::scal( hm_, -1.);
    dg::blas1::axpby(  1., hp_, +1., hm_, hz_);    //
 
}

template<class G, class M, class container>
void FieldAligned<G,M,container>::set_boundaries( dg::bc bcz, const container& global, double scal_left, double scal_right)
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

template< class G, class M, class container>
template< class BinaryOp>
container FieldAligned<G,M,container>::evaluate( BinaryOp binary, unsigned p0) const
{
    return evaluate( binary, dg::CONSTANT(1), p0, 0);
}

template<class G, class M, class container>
template< class BinaryOp, class UnaryOp>
container FieldAligned<G,M,container>::evaluate( BinaryOp binary, UnaryOp unary, unsigned p0, unsigned rounds) const
{
    //idea: simply apply I+/I- enough times on the init2d vector to get the result in each plane
    //unary function is always such that the p0 plane is at x=0
    assert( g_.Nz() > 1);
    assert( p0 < g_.Nz());
    const typename G::perpendicular_grid g2d = g_.perp_grid();
    container init2d = dg::pullback( binary, g2d), temp(init2d), tempP(init2d), tempM(init2d);
    container vec3d = dg::evaluate( dg::zero, g_);
    std::vector<container>  plus2d( g_.Nz(), (container)dg::evaluate(dg::zero, g2d) ), minus2d( plus2d), result( plus2d);
    unsigned turns = rounds; 
    if( turns ==0) turns++;
    //first apply Interpolation many times, scale and store results
    for( unsigned r=0; r<turns; r++)
        for( unsigned i0=0; i0<g_.Nz(); i0++)
        {
            dg::blas1::copy( init2d, tempP);
            dg::blas1::copy( init2d, tempM);
            unsigned rep = i0 + r*g_.Nz();
            for(unsigned k=0; k<rep; k++)
            {
                dg::blas2::symv( plus, tempP, temp);
                temp.swap( tempP);
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
            int idx = (int)i0 - (int)p0;
            if(idx>=0)
                result[i0] = plus2d[idx];
            else
                result[i0] = minus2d[abs(idx)];
            thrust::copy( result[i0].begin(), result[i0].end(), vec3d.begin() + i0*g2d.size());
        }
    }
    else //sum up plus2d and minus2d
    {
        for( unsigned i0=0; i0<g_.Nz(); i0++)
        {
            unsigned revi0 = (g_.Nz() - i0)%g_.Nz(); //reverted index
            dg::blas1::axpby( 1., plus2d[i0], 0., result[i0]);
            dg::blas1::axpby( 1., minus2d[revi0], 1., result[i0]);
        }
        dg::blas1::axpby( -1., init2d, 1., result[0]);
        for(unsigned i0=0; i0<g_.Nz(); i0++)
        {
            int idx = ((int)i0 -(int)p0 + g_.Nz())%g_.Nz(); //shift index
            thrust::copy( result[idx].begin(), result[idx].end(), vec3d.begin() + i0*g2d.size());
        }
    }
    return vec3d;
}


template< class G, class M, class container>
void FieldAligned<G,M, container>::einsPlus( const container& f, container& fpe)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fP( fpe.begin() + i0*size, fpe.begin() + (i0+1)*size);
        cusp::multiply( plus, fp, fP);
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

template< class G,class M, class container>
void FieldAligned<G,M, container>::einsMinus( const container& f, container& fme)
{
    //note that thrust functions don't work on views
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fM( fme.begin() + i0*size, fme.begin() + (i0+1)*size);
        cusp::multiply( minus, fm, fM );
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

template< class G,class M, class container>
void FieldAligned<G,M, container>::einsMinusT( const container& f, container& fpe)
{
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView rightV( right_.begin(), right_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned ip = (i0==g_.Nz()-1) ? 0:i0+1;

        cView fp( f.cbegin() + ip*size, f.cbegin() + (ip+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fP( fpe.begin() + i0*size, fpe.begin() + (i0+1)*size);
        cusp::multiply( minusT, fp, fP );
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

template< class G,class M, class container>
void FieldAligned<G,M, container>::einsPlusT( const container& f, container& fme)
{
    //note that thrust functions don't work on views
    unsigned size = g_.n()*g_.n()*g_.Nx()*g_.Ny();
    View ghostPV( ghostP.begin(), ghostP.end());
    View ghostMV( ghostM.begin(), ghostM.end());
    cView leftV( left_.begin(), left_.end());
    for( unsigned i0=0; i0<g_.Nz(); i0++)
    {
        unsigned im = (i0==0) ? g_.Nz()-1:i0-1;
        cView fm( f.cbegin() + im*size, f.cbegin() + (im+1)*size);
        cView f0( f.cbegin() + i0*size, f.cbegin() + (i0+1)*size);
        View fM( fme.begin() + i0*size, fme.begin() + (i0+1)*size);
        cusp::multiply( plusT, fm, fM );
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

