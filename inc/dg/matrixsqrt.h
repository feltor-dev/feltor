#include <cmath>
#include "blas.h"
#include "functors.h"

#include "adaptive.h"
#include "lanczos.h"
#include "helmholtz.h"
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
        m_A = A;
        Container temp = dg::evaluate(dg::zero,g);
        m_rhs.construct(A, temp, epsCG, true, true);
        m_epsTimerel = epsTimerel;
        m_epsTimeabs = epsTimeabs;
        m_rhs.set_weights(m_A.weights());
        m_rhs.set_inv_weights(m_A.inv_weights());
        m_rhs.set_precond(m_A.precond());
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
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        unsigned counter =  dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., x, 1., b, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs);
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "# b = sqrt(A) x took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
        return counter;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    dg::SqrtODE<dg::Helmholtz<Geometry,  Matrix, Container>, Container> m_rhs;  
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
     */
    DirectSqrtCauchySolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG, unsigned iterCauchy)
    {
        construct(A, g, epsCG, iterCauchy);
    }
    void construct(const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG, unsigned iterCauchy)
    {
        m_A = A;
        Container temp = dg::evaluate(dg::zero,g);
        m_iterCauchy = iterCauchy;
        m_cauchysqrtint.construct(m_A, temp, epsCG, true, true);
        m_cauchysqrtint.set_weights(m_A.weights());
        m_cauchysqrtint.set_inv_weights(m_A.inv_weights());
        m_cauchysqrtint.set_precond(m_A.precond());
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmin = 1.-A.alpha()*hxhy*(1+1);
        m_EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));
    }
    /**
     * @brief Compute \f$b \approx \sqrt{A} x  \f$ via sqrt Cauchy integral solve
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx \sqrt{A} x  \f$
     * @return number of timesteps of sqrt ODE solve
     */    
    void operator()(const Container& x, Container& b)
    {
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        m_cauchysqrtint(x, b, m_EVmin, m_EVmax, m_iterCauchy);
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "# b = sqrt(A) x took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
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
template< class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container, class SubContainer>
struct KrylovSqrtODESolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    KrylovSqrtODESolve() {}
    /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
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
        m_e1.assign(max_iterations, 0.);
        m_e1[0] = 1.;
        m_y.assign(max_iterations, 1.);
        m_T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_rhs.construct(m_T, m_e1, epsCG, true, false);
        m_lanczos.construct(copyable, max_iterations);
    }
    /**
     * @brief Compute \f$b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f$ via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f$
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        //Lanczos solve first         
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        value_type xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        if( xnorm == 0)
        {
            dg::blas1::copy( x,b);
            return 0;
        }
        m_T  = m_lanczos(m_A, x, b, m_A.inv_weights(), m_A.weights(),  m_eps); 
        m_e1.resize(m_lanczos.get_iter(), 0.);
        m_e1[0] = 1.;
        m_y.resize( m_lanczos.get_iter());
        
        m_rhs.new_size(m_lanczos.get_iter()); //resize  vectors in sqrtODE
        m_rhs.set_A(m_T); //set T in sqrtODE 

        unsigned counter = dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., m_e1, 1., m_y, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs); // y = T^(1/2) e_1

        m_lanczos.normMxVy(m_A, m_T, m_A.inv_weights(), m_A.weights(),  m_y,  b, x, xnorm, m_lanczos.get_iter());          // b = ||x|| V T^(1/2) e_1    
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        {
            std::cout << "# b = sqrt(A) x with " << m_lanczos.get_iter()  << " iterations took "<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        //reset max iterations if () operator is called again
        m_lanczos.set_iter(m_max_iter);
        return counter;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter;
    value_type m_epsCG,m_epsTimerel, m_epsTimeabs,  m_eps;
    SubContainer m_e1, m_y;
#ifdef MPI_VERSION
    SubContainer m_b;
#endif
    dg::SqrtODE<DiaMatrix, SubContainer> m_rhs;  
    dg::Lanczos< Container, SubContainer, DiaMatrix, CooMatrix > m_lanczos;
    DiaMatrix m_T; 
};

/*! 
 * @brief Shortcut for \f$b \approx \sqrt{A} x  \f$ solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly cauchy solve
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note The approximation relies on Projection \f$b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the Lanczos solve and \f$e_1\f$ is the normalized unit vector. The vector \f$\sqrt{T} e_1\f$ is computed via the sqrt ODE solve.
 */
template< class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container, class SubContainer>
struct KrylovSqrtCauchySolve
{
   public:
    using value_type = dg::get_value_type<Container>;
    ///@brief empty object ( no memory allocation)
    KrylovSqrtCauchySolve() {}
    /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver in Cauchy integral solver
     * @param max_iterations Max iterations of Lanczos method
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
        m_e1.assign(max_iterations, 0.);
        m_e1[0] = 1.;
        m_y.assign(max_iterations, 1.);
        m_T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_cauchysqrt.construct(m_T, m_e1, epsCG, false, true);
        m_lanczos.construct(copyable, max_iterations);
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmin = 1.-A.alpha()*hxhy*(1.0 + 1.0); //EVs of Helmholtz
        m_EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny())); //EVs of Helmholtz
#ifdef DG_DEBUG
        std::cout << "min EV = "<<m_EVmin <<"  max EV = "<<m_EVmax << "\n";
#endif //DG_DEBUG
    }
    /**
     * @brief Compute \f$b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f$ via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f$b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f$
     * @param iterCauchy iterations of sum of cauchy integral
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b)
    {
        //Lanczos solve first         
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        value_type xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        if( xnorm == 0)
        {
            dg::blas1::copy( x,b);
            return 0;
        }
        m_T = m_lanczos(m_A, x, b, m_A.inv_weights(), m_A.weights(), m_eps); 
        //TODO for a more rigorous eps multiply with sqrt(max_val(m_A.weights())/min_val(m_A.weights()))*sqrt(m_EVmin)
        m_e1.resize(m_lanczos.get_iter(), 0.);
        m_e1[0] = 1.;
        m_y.resize(m_lanczos.get_iter());        

        m_cauchysqrt.new_size(m_lanczos.get_iter()); //resize vectors in cauchy
        m_cauchysqrt.set_A(m_T);         //set T in cauchy
        m_cauchysqrt(m_e1, m_y, m_EVmin, m_EVmax, m_iterCauchy); //(minEV, maxEV) estimated from Helmholtz operator, which are close to the min and max EVs of T
        m_lanczos.normMxVy(m_A, m_T, m_A.inv_weights(), m_A.weights(),  m_y,  b, x, xnorm, m_lanczos.get_iter());
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif // MPI_VERSION
        {
            std::cout << "# b = sqrt(A) x with " << m_lanczos.get_iter()  << " iterations took "<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        //reset max iterations if () operator is called again
        m_lanczos.set_iter(m_max_iter);
        return m_iterCauchy;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter, m_iterCauchy;
    value_type m_eps, m_EVmin, m_EVmax;
    SubContainer m_e1, m_y, m_b;
    DiaMatrix m_T; 
    dg::SqrtCauchyInt<DiaMatrix, SubContainer> m_cauchysqrt;  
    dg::Lanczos< Container, SubContainer, DiaMatrix, CooMatrix> m_lanczos;
};

/*! 
 * @brief Shortcut for \f$x \approx \sqrt{A}^{-1} b  \f$ solve via exploiting first a Krylov projection achieved by the PCG method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper.
 *
 *@ingroup matrixfunctionapproximation 
 * 
 * @note The approximation relies on Projection \f$x = \sqrt{A}^{-1} b  \approx  R \sqrt{T^{-1}} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the PCG solve and \f$e_1\f$ is the normalized unit vector. The vector \f$\sqrt{T^{-1}} e_1\f$ is computed via the sqrt ODE solve.
 */
template<class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container, class SubContainer>
class KrylovSqrtODEinvert
{
  public:
    using value_type = dg::get_value_type<Container>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    KrylovSqrtODEinvert(){}
    ///@copydoc construct()
 /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver in adaptive ODE solver
     * @param epsTimerel relative accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param epsTimeabs absolute accuracy of adaptive ODE solver (Dormand-Prince-7-4-5)
     * @param max_iterations max number of iterations
     * @param eps accuracy of CGtridiag method
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
        m_e1.assign(max_iterations, 0.);
        m_e1[0] = 1.;
        m_y.assign(max_iterations, 1.);
        m_Tinv.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_R.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_rhs.construct(m_Tinv, m_e1, epsCG, false, false);
        m_cgtridiag.construct(copyable, max_iterations);
    }
    /**
     * @brief Solve the system \f$\sqrt{A}*x = b \f$ for x using PCG method and sqrt ODE solve
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. (is multiplied by weights)
     * @param P The preconditioner to be used
     * @param S (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium.
      */
    unsigned operator()(Container& x, const Container& b)
    {
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        if( sqrt(dg::blas2::dot(m_A.weights(), b)) == 0)
        {
            dg::blas1::copy( b, x);
#ifdef DG_BENCHMARK
            t.toc();
            std::cout << "# x = sqrt(A)^(-1) b with 0 iterations took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
            return 0;
        }
        //multiply weights
        dg::blas2::symv(m_A.weights(), b, m_b);
        //Compute x (with initODE with gemres replacing cg invert)
        m_TinvRpair = m_cgtridiag(m_A, x, m_b, m_A.inv_weights(), m_A.weights(), m_eps, 1.); 
        m_Tinv = m_TinvRpair.first; 
        m_R    = m_TinvRpair.second;   
        
        m_e1.resize(m_cgtridiag.get_iter(), 0.);
        m_e1[0] = 1.;
        m_y.resize( m_cgtridiag.get_iter(), 0.);
        m_rhs.new_size(m_cgtridiag.get_iter()); //resize  vectors in sqrtODE solver
        m_rhs.set_A(m_Tinv);

        //could be replaced by Cauchy sqrt solve
        unsigned time_iter = dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., m_e1, 1., m_y, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs);
        std::cout << "Time iterations  " << time_iter  << "\n";
        dg::blas2::gemv(m_R, m_y, x);  // x =  R T^(-1/2) e_1
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "# x = sqrt(A)^(-1) b with " << m_cgtridiag.get_iter()  << " iterations took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
        m_cgtridiag.set_iter(m_max_iter);
        return time_iter;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter;
    value_type m_epsCG, m_epsTimerel, m_epsTimeabs, m_eps;
    Container m_b;
    SubContainer m_e1, m_y;
    dg::SqrtODE<DiaMatrix, SubContainer> m_rhs;  
    dg::CGtridiag< Container, SubContainer, DiaMatrix, CooMatrix > m_cgtridiag;
    CooMatrix m_R, m_Tinv;     
    std::pair<CooMatrix, CooMatrix> m_TinvRpair;   
};



/*! 
 * @brief Shortcut for \f$x \approx \sqrt{A}^{-1} b  \f$ solve via exploiting first a Krylov projection achieved by the PCG method and and secondly a sqrt cauchy solve
 * 
 * @ingroup matrixfunctionapproximation
 * 
 * @note The approximation relies on Projection \f$x = \sqrt{A}^{-1} b  \approx  R \sqrt{T^{-1}} e_1\f$, where \f$T\f$ and \f$V\f$ is the tridiagonal and orthogonal matrix of the PCG solve and \f$e_1\f$ is the normalized unit vector. The vector \f$\sqrt{T^{-1}} e_1\f$ is computed via the sqrt ODE solve.
 */
template<class Geometry, class Matrix, class DiaMatrix, class CooMatrix, class Container, class SubContainer>
class KrylovSqrtCauchyinvert
{
  public:
    using value_type = dg::get_value_type<Container>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    KrylovSqrtCauchyinvert(){}
    ///@copydoc construct()
 /**
     * @brief Construct KrylovSqrtSolve
     *
     * @param A Helmholtz operator
     * @param g grid
     * @param epsCG accuracy of conjugate gradient solver in Cauchy integral solver
     * @param max_iterations max number of iterations
     * @param iterCauchy iterations of cauchy integral
     * @param eps accuracy of CGtridiag method
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
        m_e1.assign(max_iterations, 0.);
        m_e1[0] = 1.;
        m_y.assign(max_iterations, 1.);
        m_Tinv.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_R.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_cauchysqrt.construct(m_Tinv, m_e1, epsCG, false, false);
        m_cgtridiag.construct(copyable, max_iterations);
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
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. (is multiplied by weights)
     * @param P The preconditioner to be used
     * @param S (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium.
      */
    unsigned operator()(Container& x, const Container& b)
    {
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        if( sqrt(dg::blas2::dot(m_A.weights(), b)) == 0)
        {
            dg::blas1::copy( b, x);
#ifdef DG_BENCHMARK
            t.toc();
            std::cout << "# x = sqrt(A)^(-1) b with 0 iterations took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
            return 0;
        }
        //multiply weights
        dg::blas2::symv(m_A.weights(), b, m_b);
        //Compute x (with initODE with gemres replacing cg invert)
        m_TinvRpair = m_cgtridiag(m_A, x, m_b, m_A.inv_weights(), m_A.weights(), m_eps, 1.); 
        m_Tinv = m_TinvRpair.first; 
        m_R    = m_TinvRpair.second;               
        
        m_e1.resize(m_cgtridiag.get_iter(), 0.);
        m_e1[0] = 1.;
        m_y.resize( m_cgtridiag.get_iter(), 0.);
        m_cauchysqrt.new_size(m_cgtridiag.get_iter()); //resize  vectors in sqrtODE solver
        m_cauchysqrt.set_A(m_Tinv);
        
        m_cauchysqrt(m_e1, m_y, m_EVmin, m_EVmax, m_iterCauchy); //(minEV, maxEV) estimated
        dg::blas2::gemv(m_R, m_y, x);  // x =  R T^(-1/2) e_1  
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "# x = sqrt(A)^(-1) b with " << m_cgtridiag.get_iter()  << " iterations took "<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
        m_cgtridiag.set_iter(m_max_iter);
        return m_iterCauchy;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter, m_iterCauchy;
    value_type m_epsCG,  m_eps, m_EVmin, m_EVmax;
    Container m_b;
    SubContainer m_e1, m_y;
    dg::SqrtCauchyInt<DiaMatrix, SubContainer> m_cauchysqrt; 
    dg::CGtridiag< Container, SubContainer, DiaMatrix, CooMatrix > m_cgtridiag;
    CooMatrix m_R, m_Tinv;     
    std::pair<CooMatrix, CooMatrix> m_TinvRpair;   
};

} //namespace dg
