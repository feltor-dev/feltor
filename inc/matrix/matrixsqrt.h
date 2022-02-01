#include <cmath>
#include "dg/algorithm.h"

#include "functors.h"
#include "lanczos.h"
#include "sqrt_cauchy.h"
#include "sqrt_ode.h"

#ifdef DG_BENCHMARK
#include "backend/timer.h"
#endif //DG_BENCHMARK
namespace dg
{
/**
 * @brief Shortcut for \f$b \approx \sqrt{A} x  \f$ solve directly via sqrt ODE solve with adaptive ERK class as timestepper
 * @ingroup matrixfunctionapproximation
*/
template< class Geometry, class Matrix, class Container>
struct DirectSqrtODESolve
{
   public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    DirectSqrtODESolve() {}
    /**
     * @brief Construct DirectSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     */
    DirectSqrtODESolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG, value_type epsTimerel, value_type epsTimeabs)
    { 
        Container temp = dg::evaluate(dg::zero,g);
        m_sqrtode.construct(A, temp, epsCG, true, true);
        m_epsTimerel = epsTimerel;
        m_epsTimeabs = epsTimeabs;
        m_sqrtode.set_weights(A.weights());
        m_sqrtode.set_inv_weights(A.inv_weights());
        m_sqrtode.set_precond(A.precond());
    }
    /**
     * @brief Compute \f$b \approx \sqrt{A} x  \f$ via sqrt ODE solve
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx \sqrt{A} x  \f$
     * @return number of timesteps of sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        return dg::integrateERK( "Dormand-Prince-7-4-5", m_sqrtode, 0., x, 1., b, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs);
    }
  private:
    dg::SqrtODE<dg::Helmholtz<Geometry,  Matrix, Container>, Container> m_sqrtode;  
    value_type m_epsTimerel, m_epsTimeabs;
};

/** @brief Shortcut for \f$b \approx \sqrt{A} x  \f$ solve directly via sqrt Cauchy integral solve
 * @ingroup matrixfunctionapproximation
*/
template< class Geometry, class Matrix, class Container>
struct DirectSqrtCauchySolve
{
   public:
    using geometry_type = Geometry;
    using matrix_type = Matrix;
    using container_type = Container;
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    DirectSqrtCauchySolve() {}
    /**
     * @brief Construct DirectSqrtCauchySolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver
     * @param iterCauchy maximum number of Cauchy iterations
     */
    DirectSqrtCauchySolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG, unsigned iterCauchy)
    {
        construct(A, g, epsCG, iterCauchy);
    }
    void construct(const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG, unsigned iterCauchy)
    {
        Container temp = dg::evaluate(dg::zero,g);
        m_iterCauchy = iterCauchy;
        m_cauchysqrtint.construct(A, temp, epsCG, true, true);
        m_cauchysqrtint.set_weights(A.weights());
        m_cauchysqrtint.set_inv_weights(A.inv_weights());
        m_cauchysqrtint.set_precond(A.precond());
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmin = 1.-A.alpha()*hxhy*(1+1);
        m_EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));
    }
    /**
     * @brief Compute \f$b \approx \sqrt{A} x  \f$ via sqrt Cauchy integral solve
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx \sqrt{A} x  \f$
     * @return number of integration steps of sqrt cauchy solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        m_cauchysqrtint(x, b, m_EVmin, m_EVmax, m_iterCauchy);
        return m_iterCauchy;
    }
  private:
    unsigned m_iterCauchy;
    dg::SqrtCauchyInt<dg::Helmholtz<Geometry,  Matrix, Container>, Container> m_cauchysqrtint;
    value_type m_EVmin,m_EVmax;
};

/*! 
 * @brief Shortcut for \f$b \approx \sqrt{A} x  \f$ solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note The approximation relies on Projection \f$b \approx \sqrt{A} x \approx b \approx ||x||_M V \sqrt{T} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the Lanczos solve and \f$e_1\f$ is the normalized unit vector. The vector \f$\sqrt{T} e_1\f$ is computed via the sqrt ODE solve.
 */
template< class Geometry, class Matrix, class Container>
struct KrylovSqrtODESolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief empty object ( no memory allocation)
    KrylovSqrtODESolve() {}
    /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param copyable
     * @param epsCG accuracy of conjugate gradient solver in adaptive ODE solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param max_iterations max number of iterations
     * @param eps accuracy of Lanczos method
     */
    KrylovSqrtODESolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable, value_type epsCG, value_type epsTimerel, value_type epsTimeabs, unsigned max_iterations, value_type eps)  
    { 
        construct(A, g, copyable, epsCG, epsTimerel, epsTimeabs, max_iterations, eps);
    }
    void construct( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, value_type epsTimerel, value_type epsTimeabs, unsigned max_iterations, value_type eps)
    {      
        m_A = A;
        m_epsCG = epsCG;
        m_epsTimerel = epsTimerel;
        m_epsTimeabs = epsTimeabs;
        m_max_iter = max_iterations;
        m_eps = eps;
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_TH.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_sqrtodeH.construct(m_TH, m_e1H, epsCG, true, false);
        m_lanczos.construct(copyable, max_iterations);
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0); //EVs of Helmholtz
        value_type max_weights =   dg::blas1::reduce(m_A.weights(), 0., dg::AbsMax<double>() );
        value_type min_weights =  -dg::blas1::reduce(m_A.weights(), max_weights, dg::AbsMin<double>() );
        m_kappa = sqrt(max_weights/min_weights); //condition number 
#ifdef DG_DEBUG
        std::cout << "min(EV) = "<<m_EVmin <<"\n";
        std::cout << "min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
        std::cout << "kappa   = "<<m_kappa <<"\n";
        std::cout << "res_fac = "<<m_kappa*sqrt(m_EVmin)<< "\n";
#endif //DG_DEBUG
    }
    /**
     * @brief Compute \f$b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f$ via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f$
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    std::array<unsigned,2> operator()(const Container& x, Container& b)
    {
#ifdef DG_BENCHMARK
        Timer t;
        t.tic();
#endif //DG_BENCHMARK
        //Lanczos solve first         
        value_type xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        if( xnorm == 0)
        {
            dg::blas1::copy( x,b);
            return {0,0} ;
        }
        m_TH  = m_lanczos(m_A, x, b, m_A.inv_weights(), m_A.weights(),  m_eps, false, m_kappa*sqrt(m_EVmin)); 
        unsigned iter = m_lanczos.get_iter();
        m_e1H.resize(iter, 0.);
        m_e1H[0] = 1.;
        m_yH.resize( iter);
        
        m_sqrtodeH.new_size(iter); //resize  vectors in sqrtODE
        m_sqrtodeH.set_A(m_TH); //set T in sqrtODE 

        unsigned counter = dg::integrateERK( "Dormand-Prince-7-4-5", m_sqrtodeH, 0., m_e1H, 1., m_yH, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs); // y = T^(1/2) e_1

        m_lanczos.normMxVy(m_A, m_TH, m_A.inv_weights(), m_A.weights(),  m_yH,  b, x, xnorm, iter);          // b = ||x|| V T^(1/2) e_1  
        //reset max iterations if () operator is called again
        m_lanczos.set_iter(m_max_iter);
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        {
            std::cout << "# SQRT solve with {"<< iter << "," << counter<< "} iterations took "<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        return {iter, counter};
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter;
    value_type m_epsCG, m_epsTimerel, m_epsTimeabs, m_eps, m_EVmin, m_kappa;
    HVec m_e1H, m_yH;
    dg::SqrtODE<HDiaMatrix, HVec> m_sqrtodeH;  
    dg::Lanczos< Container > m_lanczos;
    HDiaMatrix m_TH; 
};

/*! 
 * @brief Shortcut for \f$b \approx \sqrt{A} x  \f$ solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly cauchy solve
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note The approximation relies on Projection \f$b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the Lanczos solve and \f$e_1\f$ is the normalized unit vector. The vector \f$\sqrt{T} e_1\f$ is computed via the sqrt ODE solve.
 */
template< class Geometry, class Matrix, class Container>
struct KrylovSqrtCauchySolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief empty object ( no memory allocation)
    KrylovSqrtCauchySolve() {}
    /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param copyable
     * @param epsCG accuracy of conjugate gradient solver in Cauchy integral solver
     * @param max_iterations Max iterations of Lanczos method (e.g. 500)
     * @param iterCauchy iterations of cauchy integral
     * @param eps accuracy of lanczos method
     */
    KrylovSqrtCauchySolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable, value_type epsCG, unsigned max_iterations, unsigned iterCauchy, value_type eps)
    { 
        construct(A, g, copyable, epsCG, max_iterations, iterCauchy, eps);
    }
    void construct( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, unsigned max_iterations, unsigned iterCauchy, value_type eps)
    {      
        m_A = A;
        m_max_iter = max_iterations;
        m_eps = eps;
        m_iterCauchy = iterCauchy;
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_TH.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_TH.diagonal_offsets[0] = -1;
        m_TH.diagonal_offsets[1] =  0;
        m_TH.diagonal_offsets[2] =  1;
        m_cauchysqrtH.construct(m_TH, m_e1H, epsCG, false, true);
        m_lanczos.construct(copyable, max_iterations);
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0); //EVs of Helmholtz
        m_EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny())); //EVs of Helmholtz
        value_type max_weights =   dg::blas1::reduce(m_A.weights(), 0., dg::AbsMax<double>() );
        value_type min_weights =  -dg::blas1::reduce(m_A.weights(), max_weights, dg::AbsMin<double>() );
        m_kappa = sqrt(max_weights/min_weights); //condition number 
#ifdef DG_DEBUG
        std::cout << "min(EV) = "<<m_EVmin <<"  max(EV) = "<<m_EVmax << "\n";
        std::cout << "min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
        std::cout << "kappa   = "<<m_kappa <<"\n";
        std::cout << "res_fac = "<<m_kappa*sqrt(m_EVmin)<< "\n";
#endif //DG_DEBUG
    }
    /**
     * @brief Compute \f$b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f$ via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f$
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    std::array<unsigned,2> operator()(const Container& x, Container& b)
    {
#ifdef DG_BENCHMARK
        Timer t;
        t.tic();
#endif //DG_BENCHMARK
        value_type xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        if( xnorm == 0)
        {
            dg::blas1::copy( x,b);
            return {0, m_iterCauchy};
        }
        
        m_TH = m_lanczos(m_A, x, b, m_A.inv_weights(), m_A.weights(), m_eps, false, m_kappa*sqrt(m_EVmin)); 
        unsigned iter = m_lanczos.get_iter();

        m_e1H.resize(iter, 0.);
        m_e1H[0] = 1.;
        m_yH.resize(iter);        

        m_cauchysqrtH.new_size(iter); //resize vectors in cauchy
        m_cauchysqrtH.set_A(m_TH);         //set T in cauchy
        m_cauchysqrtH(m_e1H, m_yH, m_EVmin, m_EVmax, m_iterCauchy); //(minEV, maxEV) estimated from Helmholtz operator, which are close to the min and max EVs of T
        m_lanczos.normMxVy(m_A, m_TH, m_A.inv_weights(), m_A.weights(),  m_yH,  b, x, xnorm, iter);

        //reset max iterations if () operator is called again
        m_lanczos.set_iter(m_max_iter);
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        {
            std::cout << "# SQRT solve with {"<< iter << "," << m_iterCauchy<< "} iterations took "<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        return {iter, m_iterCauchy};
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter, m_iterCauchy;
    value_type m_eps, m_EVmin, m_EVmax, m_kappa;
    HVec m_e1H, m_yH;
    HDiaMatrix m_TH; 
    dg::SqrtCauchyInt<HDiaMatrix, HVec> m_cauchysqrtH;  
    dg::Lanczos< Container> m_lanczos;
};

/*! 
 * @brief Shortcut for \f$x \approx \sqrt{A}^{-1} b  \f$ solve via exploiting first a Krylov projection achieved by the PCG method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper.
 *
 *@ingroup matrixfunctionapproximation 
 * 
 * @note The approximation relies on Projection \f$x = \sqrt{A}^{-1} b  \approx  R \sqrt{T^{-1}} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the PCG solve and \f$e_1\f$ is the normalized unit vector. The vector \f$\sqrt{T^{-1}} e_1\f$ is computed via the sqrt ODE solve.
 */
template<class Geometry, class Matrix, class Container>
class KrylovSqrtODEinvert
{
  public:
    using value_type = dg::get_value_type<Container>; //!< value type of the ContainerType class
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    KrylovSqrtODEinvert(){}
    ///@copydoc construct()
 /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param copyable
     * @param epsCG accuracy of conjugate gradient solver in adaptive ODE solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param max_iterations max number of iterations
     * @param eps accuracy of MCG method
     */
    KrylovSqrtODEinvert( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable, value_type epsCG, value_type epsTimerel, value_type epsTimeabs, unsigned max_iterations, value_type eps)  
    { 
        construct(A, g, copyable, epsCG, epsTimerel, epsTimeabs, max_iterations, eps);
    }
    void construct( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, value_type epsTimerel, value_type epsTimeabs, unsigned max_iterations, value_type eps)
    {      
        m_A = A;
        m_epsCG = epsCG;
        m_epsTimerel = epsTimerel;
        m_epsTimeabs = epsTimeabs;
        m_max_iter = max_iterations;
        m_b = copyable;
        m_eps = eps;
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_TinvH.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_sqrtodeH.construct(m_TinvH, m_e1H, epsCG, false, false);
        m_mcg.construct(copyable, max_iterations);
    }
    /**
     * @brief Solve the system \f$\sqrt{A}*x = b \f$ for x using PCG method and sqrt ODE solve
     * 
     * @param x Contains an initial value
     * @param b The right hand side vector. (is multiplied by weights)
     * 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium.
      */
    std::array<unsigned,2> operator()(Container& x, const Container& b)
    {

        if( sqrt(dg::blas2::dot(m_A.weights(), b)) == 0)
        {
            dg::blas1::copy( b, x);
            return {0,0};
        }
        //multiply weights
        dg::blas2::symv(m_A.weights(), b, m_b);
        //Compute x (with initODE with gemres replacing cg invert)
        m_TH = m_mcg(m_A, x, m_b, m_A.inv_weights(), m_A.weights(), m_eps, 1., false); 
        unsigned iter = m_mcg.get_iter();
        m_TridiaginvH.resize(iter);
        m_TinvH = m_TridiaginvH(m_TH);
        
        m_e1H.resize(iter, 0.);
        m_e1H[0] = 1.;
        m_yH.resize( iter, 0.);
        m_sqrtodeH.new_size(iter); //resize  vectors in sqrtODE solver
        m_sqrtodeH.set_A(m_TinvH);

        //could be replaced by Cauchy sqrt solve
        unsigned time_iter = dg::integrateERK( "Dormand-Prince-7-4-5", m_sqrtodeH, 0., m_e1H, 1., m_yH, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs);
        m_mcg.Ry(m_A, m_TH, m_A.inv_weights(), m_A.weights(), m_yH, x, m_b,  iter); // x =  R T^(-1/2) e_1

        m_mcg.set_iter(m_max_iter);
        return {iter,time_iter};
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter;
    value_type m_epsCG, m_epsTimerel, m_epsTimeabs, m_eps;
    Container m_b;
    HVec m_e1H, m_yH;
    dg::SqrtODE<HDiaMatrix, HVec> m_sqrtodeH;  
    dg::TridiagInvDF<HVec, HDiaMatrix, HCooMatrix> m_TridiaginvH;
    dg::MCG< Container> m_mcg;
    HCooMatrix m_TinvH;
    HDiaMatrix m_TH;
};



/*! 
 * @brief Shortcut for \f$x \approx \sqrt{A}^{-1} b  \f$ solve via exploiting first a Krylov projection achieved by the PCG method and and secondly a sqrt cauchy solve
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note The approximation relies on Projection \f$x = \sqrt{A}^{-1} b  \approx  R \sqrt{T^{-1}} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the PCG solve and \f$e_1\f$ is the normalized unit vector. The vector \f$\sqrt{T^{-1}} e_1\f$ is computed via the sqrt ODE solve.
 */
template<class Geometry, class Matrix, class Container>
class KrylovSqrtCauchyinvert
{
  public:
    using value_type = dg::get_value_type<Container>; //!< value type of the ContainerType class
    using HCooMatrix = cusp::coo_matrix<int, value_type, cusp::host_memory>;
    using HDiaMatrix = cusp::dia_matrix<int, value_type, cusp::host_memory>;
    using HVec = dg::HVec;
    ///@brief Allocate nothing, Call \c construct method before usage
    KrylovSqrtCauchyinvert(){}
    ///@copydoc construct()
 /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param copyable
     * @param epsCG accuracy of conjugate gradient solver in Cauchy integral solver
     * @param max_iterations max number of iterations
     * @param iterCauchy iterations of cauchy integral
     * @param eps accuracy of MCG method
     */
    KrylovSqrtCauchyinvert( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable, value_type epsCG, unsigned max_iterations, unsigned iterCauchy, value_type eps)  
    { 
        construct(A, g, copyable, epsCG, max_iterations, iterCauchy, eps);
    }
    void construct( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, unsigned max_iterations, unsigned iterCauchy, value_type eps)
    {      
        m_A = A;
        m_epsCG = epsCG;
        m_max_iter = max_iterations;
        m_iterCauchy = iterCauchy;
        m_eps = eps;
        m_b = copyable;
        m_e1H.assign(max_iterations, 0.);
        m_e1H[0] = 1.;
        m_yH.assign(max_iterations, 1.);
        m_TinvH.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_cauchysqrtH.construct(m_TinvH, m_e1H, epsCG, false, false);
        m_mcg.construct(copyable, max_iterations);
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmax = 1./(1.-A.alpha()*hxhy*(1.0 + 1.0)); //EVs of inverse Helmholtz
        m_EVmin = 1./(1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()))); //EVs of inverse Helmholtz
#ifdef DG_DEBUG
        std::cout << "min EV = "<<m_EVmin <<"  max EV = "<<m_EVmax << "\n";
#endif //DG_DEBUG
    }
    /**
     * @brief Solve the system \f$\sqrt{A}*x = b \f$ for x using PCG method and sqrt ODE solve
     * 
     * @param x Contains an initial value
     * @param b The right hand side vector. (is multiplied by weights)
     * 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium.
      */
    std::array<unsigned,2> operator()(Container& x, const Container& b)
    {
        if( sqrt(dg::blas2::dot(m_A.weights(), b)) == 0)
        {
            dg::blas1::copy( b, x);
            return {0,0};
        }
        //multiply weights
        dg::blas2::symv(m_A.weights(), b, m_b);
        //Compute x (with initODE with gemres replacing cg invert)
        m_TH = m_mcg(m_A, x, m_b, m_A.inv_weights(), m_A.weights(), m_eps, 1., false); 
        unsigned iter = m_mcg.get_iter();
        m_TridiaginvH.resize(iter);
        m_TinvH = m_TridiaginvH(m_TH); 

        m_e1H.resize(iter, 0.);
        m_e1H[0] = 1.;
        m_yH.resize( iter, 0.);
        m_cauchysqrtH.new_size(iter); //resize  vectors in sqrtODE solver
        m_cauchysqrtH.set_A(m_TinvH);
        
        m_cauchysqrtH(m_e1H, m_yH, m_EVmin, m_EVmax, m_iterCauchy); //(minEV, maxEV) estimated // y= T^(-1/2) e_1  

        m_mcg.Ry(m_A, m_TH, m_A.inv_weights(), m_A.weights(), m_yH, x, m_b,  iter); // x =  R T^(-1/2) e_1  

        m_mcg.set_iter(m_max_iter);
        return {iter,m_iterCauchy};
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter, m_iterCauchy;
    value_type m_epsCG,  m_eps, m_EVmin, m_EVmax;
    Container m_b;
    HVec m_e1H, m_yH;
    dg::SqrtCauchyInt<HDiaMatrix, HVec> m_cauchysqrtH; 
    dg::TridiagInvDF<HVec, HDiaMatrix, HCooMatrix> m_TridiaginvH;
    dg::MCG< Container > m_mcg;
    HCooMatrix  m_TinvH;     
    HDiaMatrix m_TH;
};

} //namespace dg
