#include <cmath>

#include "blas.h"
#include "functors.h"

#include "adaptive.h"
#include "lanczos.h"
#include "sqrt_cauchy.h"
#include "sqrt_ode.h"

#ifdef DG_BENCHMARK
#include "backend/timer.h"
#endif //DG_BENCHMARK

///@brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve directly via sqrt ODE solve with adaptive ERK class as timestepper
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
     * @brief Compute \f[b \approx \sqrt{A} x  \f] via sqrt ODE solve
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \f]
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
        std::cout << "# Square root matrix computation took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
        return counter;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    Rhs<dg::Helmholtz<Geometry,  Matrix, Container>, Container> m_rhs;  
    value_type m_epsTimerel, m_epsTimeabs;
};

///@brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve directly via sqrt Cauchy integral solve
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
    DirectSqrtCauchySolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG)
    {
        construct(A, g, epsCG);
    }
    void construct(const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, value_type epsCG)
    {
        m_A = A;
        Container temp = dg::evaluate(dg::zero,g);
        m_cauchysqrtint.construct(m_A, temp, epsCG, true);
        m_cauchysqrtint.set_weights(m_A.weights());
        m_cauchysqrtint.set_inv_weights(m_A.inv_weights());
        m_cauchysqrtint.set_precond(m_A.precond());
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmin = 1.-A.alpha()*hxhy*(1+1);
        m_EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));
    }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x  \f] via sqrt Cauchy integral solve
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \f]
     * @return number of timesteps of sqrt ODE solve
     */    
    void operator()(const Container& x, Container& b, const unsigned& iterCauchy)
    {
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        m_cauchysqrtint(x, b, m_EVmin, m_EVmax, iterCauchy);
#ifdef DG_BENCHMARK
        t.toc();
        std::cout << "# Square root matrix computation took \t"<<t.diff()<<"s\n";
#endif //DG_BENCHMARK
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    CauchySqrtInt<dg::Helmholtz<Geometry,  Matrix, Container>, Container> m_cauchysqrtint;
    value_type m_EVmin,m_EVmax;
};

/*! 
 * @brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[b \approx \sqrt{A} x \approx b \approx ||x||_M V \sqrt{T} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the Lanczos solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T} e_1\f] is computed via the sqrt ODE solve.
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
    KrylovSqrtODESolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, value_type epsTimerel, value_type epsTimeabs, unsigned max_iterations, value_type eps)  
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
        m_xnorm = 0.;
        m_e1.assign(max_iterations, 0.);
        m_e1[0] = 1.;
        m_y.assign(max_iterations, 1.);
        m_T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_V.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_rhs.construct(m_T, m_e1, epsCG, true, false);
        m_lanczos.construct(copyable, max_iterations);
    }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f] via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f]
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
        m_xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        if( m_xnorm == 0)
        {
            dg::blas1::copy( x,b);
            return 0;
        }
        m_TVpair = m_lanczos(m_A, x, b, m_A.weights(), m_A.inv_weights(), m_eps); 
        m_T = m_TVpair.first; 
        m_V = m_TVpair.second;   
        m_e1.resize(m_lanczos.get_iter(),0);
        m_e1[0] = 1.;
        m_y.resize( m_lanczos.get_iter());
        m_rhs.new_size(m_lanczos.get_iter()); //resize  vectors in sqrtODE solver
        m_rhs.set_A(m_T); //set T in sqrtODE solver

        unsigned counter = dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., m_e1, 1., m_y, 0., dg::pid_control, dg::l2norm, m_epsTimerel, m_epsTimeabs); // y = T^(1/2) e_1
#ifdef MPI_VERSION
        dg::blas2::symv(m_V, m_y, m_b);
        dg::assign(m_b, b);
#else //ifndef MPI_VERSION
        dg::blas2::symv(m_V, m_y, b);
#endif //MPI_VERSION
        dg::blas1::scal(b, m_xnorm);             // b = ||x|| V T^(1/2) e_1    
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        {
            std::cout << "# Square root matrix computation took \t"<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        //reset max iterations if () operator is called again
        m_lanczos.set_iter(m_max_iter);
        return counter;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter;
    value_type m_epsCG,m_epsTimerel, m_epsTimeabs, m_xnorm, m_eps;
    SubContainer m_e1, m_y;
#ifdef MPI_VERSION
    SubContainer m_b;
#endif
    Rhs<DiaMatrix, SubContainer> m_rhs;  
    dg::Lanczos< Container, SubContainer, DiaMatrix, CooMatrix > m_lanczos;
    DiaMatrix m_T; 
    CooMatrix m_V;
    std::pair<DiaMatrix, CooMatrix> m_TVpair;     
};

/*! 
 * @brief Shortcut for \f[b \approx \sqrt{A} x  \f] solve via exploiting first a Krylov projection achived by the M-lanczos method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the Lanczos solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T} e_1\f] is computed via the sqrt ODE solve.
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
     * @param epsCG accuracy of conjugate gradient solver
     * @param max_iterations Max iterations of Lanczos method
     */
    KrylovSqrtCauchySolve( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable, value_type epsCG, unsigned max_iterations, value_type eps)
    { 
        construct(A, g, copyable, epsCG, max_iterations, eps);
    }
    void construct( const dg::Helmholtz<Geometry,  Matrix, Container>& A, const Geometry& g, const Container& copyable,value_type epsCG, unsigned max_iterations, value_type eps)
    {      
        m_A = A;
        m_max_iter = max_iterations;
        m_eps = eps;
        m_xnorm = 0.;
        m_e1.assign(max_iterations, 0.);
        m_e1[0] = 1.;
        m_y.assign(max_iterations, 1.);
        m_T.resize(max_iterations, max_iterations, 3*max_iterations-2, 3);
        m_T.diagonal_offsets[0] = -1;
        m_T.diagonal_offsets[1] =  0;
        m_T.diagonal_offsets[2] =  1;
        m_V.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_cauchysqrt.construct(m_T, m_e1, epsCG, false);
        m_lanczos.construct(copyable, max_iterations);
        value_type hxhy = g.lx()*g.ly()/(g.n()*g.n()*g.Nx()*g.Ny());
        m_EVmin = 1.-A.alpha()*hxhy*(1+1);
        m_EVmax = 1.-A.alpha()*hxhy*(g.n()*g.n() *(g.Nx()*g.Nx() + g.Ny()*g.Ny()));
    }
    /**
     * @brief Compute \f[b \approx \sqrt{A} x \approx  ||x||_M V \sqrt{T} e_1\f] via sqrt ODE solve.
     *
     * @param x input vector
     * @param b output vector. Is approximating \f[b \approx \sqrt{A} x  \approx  ||x||_M V \sqrt{T} e_1\f]
     * @param iterCauchy iterations of sum of cauchy integral
     * 
     * @return number of time steps in sqrt ODE solve
     */    
    unsigned operator()(const Container& x, Container& b, const unsigned& iterCauchy)
    {
        //Lanczos solve first         
#ifdef DG_BENCHMARK
        dg::Timer t;
        t.tic();
#endif //DG_BENCHMARK
        m_xnorm = sqrt(dg::blas2::dot(m_A.weights(), x)); 
        if( m_xnorm == 0)
        {
            dg::blas1::copy( x,b);
            return 0;
        }
        m_TVpair = m_lanczos(m_A, x, b, m_A.weights(), m_A.inv_weights(), m_eps); 
        //for a more rigorous eps multiply with sqrt(max_val(m_A.weights())/min_val(m_A.weights()))*sqrt(m_EVmin)
        m_T = m_TVpair.first; 
        m_V = m_TVpair.second;   
        m_e1.resize(m_lanczos.get_iter(),0);
        m_e1[0]=1.;
        m_y.resize(m_lanczos.get_iter());        

        m_cauchysqrt.new_size(m_lanczos.get_iter()); //resize vectors in cauchy sqrt solver
        m_cauchysqrt.set_A(m_T);         //set T in cauchy sqrt solver
#ifdef DG_DEBUG
        std::cout << "min EV = "<<m_EVmin <<"  max EV = "<<m_EVmax << "\n";
#endif //DG_DEBUG
        m_cauchysqrt(m_e1, m_y, m_EVmin, m_EVmax, iterCauchy); //(minEV, maxEV) estimated from Helmholtz operator, which are close to the min and max EVs of T
        dg::blas2::gemv(m_V, m_y, b);
        dg::blas1::scal(b, m_xnorm);             // b = ||x|| V T^(1/2) e_1     
#ifdef DG_BENCHMARK
        t.toc();
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
#endif //MPI
        {
            std::cout << "# Square root matrix computation took \t"<<t.diff()<<"s\n";
        }
#endif //DG_BENCHMARK
        //reset max iterations if () operator is called again
        m_lanczos.set_iter(m_max_iter);
        return iterCauchy;
    }
  private:
    dg::Helmholtz<Geometry,  Matrix, Container> m_A;
    unsigned m_max_iter;
    value_type m_eps, m_xnorm, m_EVmin, m_EVmax;
    SubContainer m_e1, m_y;
    DiaMatrix m_T; 
    CooMatrix m_V;
    std::pair<DiaMatrix, CooMatrix> m_TVpair; 
    CauchySqrtInt<DiaMatrix, SubContainer> m_cauchysqrt;  
    dg::Lanczos< Container, SubContainer, DiaMatrix, CooMatrix> m_lanczos;
};

/*! 
 * @brief Shortcut for \f[x \approx \sqrt{A}^{-1} b  \f] solve via exploiting first a Krylov projection achieved by the PCG method and and secondly a sqrt ODE solve with the adaptive ERK class as timestepper. 
 * 
 * @note The approximation relies on Projection \f[x = \sqrt{A}^{-1} b  \approx  R \sqrt{T^{-1}} e_1\f], where \f[T\f] and \f[V\f] is the tridiagonal and orthogonal matrix of the PCG solve and \f[e_1\f] is the normalized unit vector. The vector \f[\sqrt{T^{-1}} e_1\f] is computed via the sqrt ODE solve.
 */
template< class ContainerType, class SubContainerType, class DiaMatrix, class CooMatrix>
class CGsqrt
{
  public:
    using container_type = ContainerType;
    using value_type = dg::get_value_type<ContainerType>; //!< value type of the ContainerType class
    ///@brief Allocate nothing, Call \c construct method before usage
    CGsqrt(){}
    ///@copydoc construct()
    CGsqrt( const ContainerType& copyable, unsigned max_iterations)
    {
          construct(copyable, max_iterations);
    }
    ///@brief Set the maximum number of iterations
    ///@param new_max New maximum number
    void set_max( unsigned new_max) {m_max_iter = new_max;}
    ///@brief Get the current maximum number of iterations
    ///@return the current maximum
    unsigned get_max() const {return m_max_iter;}
    /**
     * @brief Allocate memory for the pcg method
     *
     * @param copyable A ContainerType must be copy-constructible from this
     * @param max_iterations Maximum number of iterations to be used
     */
    void construct( const ContainerType& copyable, unsigned max_iterations)
    {
        m_v0.assign(max_iterations,0.);
        m_vp.assign(max_iterations,0.);
        m_vm.assign(max_iterations,0.);
        m_e1.assign(max_iterations,0.);
        m_y.assign(max_iterations,0.);
        m_ap = m_p = m_r = m_rh = copyable;
        m_max_iter = max_iterations;
        m_R.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
        m_Tinv.resize(copyable.size(), max_iterations, max_iterations*copyable.size());
    }
    ///@brief Set the new number of iterations and resize Matrix T and V
    ///@param new_iter new number of iterations
    void set_iter( unsigned new_iter) {
        m_v0.resize(new_iter);
        m_vp.resize(new_iter);
        m_vm.resize(new_iter);
        m_e1.resize(new_iter,0.);
        m_e1[0]=1.;
        m_y.resize(new_iter,0.);
        m_R.resize(m_rh.size(), new_iter, new_iter*m_rh.size());
        m_Tinv.resize(new_iter, new_iter, new_iter*new_iter);
        m_invtridiag.resize(new_iter);
    }
    /**
     * @brief Solve the system \f[\sqrt{A}*x = b \f] for x using PCG method and sqrt ODE solve
     * 
     * @param A A symmetric, positive definit matrix (e.g. not normed Helmholtz operator)
     * @param x Contains an initial value
     * @param b The right hand side vector. 
     * @param P The preconditioner to be used
     * @param S (Inverse) Weights used to compute the norm for the error condition
     * @param eps The relative error to be respected
     * @param nrmb_correction the absolute error \c C in units of \c eps to be respected
     * 
     * @return Number of iterations used to achieve desired precision
     * @note So far only ordinary convergence criterium of CG method. Should be adapted to square root criterium.
      */
    template< class MatrixType, class ContainerType0, class ContainerType1, class Preconditioner, class SquareNorm>
    unsigned operator()( MatrixType& A, ContainerType0& x, const ContainerType1& b, Preconditioner& P, SquareNorm& S, value_type eps = 1e-12, value_type nrmb_correction = 1)
    {
        //Do CG iteration do get R and T matrix
        value_type nrmb = sqrt( dg::blas2::dot( S, b));
    #ifdef DG_DEBUG
    #ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if(rank==0)
    #endif //MPI
        {
        std::cout << "# Norm of b "<<nrmb <<"\n";
        std::cout << "# Residual errors: \n";
        }
    #endif //DG_DEBUG


        if( nrmb == 0)
        {
            dg::blas1::copy( b, x);
            return 0;
        }
        dg::blas2::symv( A, x, m_r);
        dg::blas1::axpby( 1., b, -1., m_r);
        dg::blas2::symv( P, m_r, m_p );//<-- compute p_0
        dg::blas1::copy( m_p, m_rh);

        //note that dot does automatically synchronize
        value_type nrm2r_old = dg::blas1::dot( m_rh, m_r); //and store the norm of it
        value_type nrm2r_new;
        
        unsigned counter = 0;
        for( unsigned i=0; i<m_max_iter; i++)
        {
            for( unsigned j=0; j<m_rh.size(); j++)
            {            
                m_R.row_indices[counter]    = j;
                m_R.column_indices[counter] = i; 
                m_R.values[counter]         = m_rh.data()[j];
                counter++;
            }
            dg::blas2::symv( A, m_p, m_ap);
            m_alpha = nrm2r_old /dg::blas1::dot( m_p, m_ap);
            dg::blas1::axpby( -m_alpha, m_ap, 1., m_r);
    #ifdef DG_DEBUG
    #ifdef MPI_VERSION
            if(rank==0)
    #endif //MPI
            {
                std::cout << "# Absolute r*S*r "<<sqrt( dg::blas2::dot(S,m_r)) <<"\t ";
                std::cout << "#  < Critical "<<eps*nrmb + eps <<"\t ";
                std::cout << "# (Relative "<<sqrt( dg::blas2::dot(S,m_r) )/nrmb << ")\n";
            }
    #endif //DG_DEBUG
            if( sqrt( dg::blas2::dot( S, m_r)) < eps*(nrmb + nrmb_correction)) //TODO change this criterium  for square root matrix
            {
                dg::blas2::symv(P, m_r, m_rh);
                nrm2r_new = dg::blas1::dot( m_rh, m_r);
                
                m_vp[i] = -nrm2r_new/nrm2r_old/m_alpha;
                m_vm[i] = -1./m_alpha;
                m_v0[i+1] = -m_vp[i];
                m_v0[i] -= m_vm[i];
                
                m_max_iter=i+1;
                break;
            }
            dg::blas2::symv(P, m_r, m_rh);
            nrm2r_new = dg::blas1::dot( m_rh, m_r);
            dg::blas1::axpby(1., m_rh, nrm2r_new/nrm2r_old, m_p );
                       
            m_vp[i] = -nrm2r_new/nrm2r_old/m_alpha;
            m_vm[i] = -1./m_alpha;
            m_v0[i+1] = -m_vp[i];
            m_v0[i] -= m_vm[i];
            nrm2r_old=nrm2r_new;
        }

        set_iter(m_max_iter);
        //Compute inverse of tridiagonal matrix
        m_Tinv = m_invtridiag(m_v0,m_vp,m_vm);

        //Compute x (with initODE with gemres replacing cg invert)
        Rhs<CooMatrix, SubContainerType> m_rhs(m_Tinv, m_e1, eps, false, false);        

        m_rhs.set_A(m_Tinv);

        //could be replaced by Cauchy sqrt solve
        unsigned time_iter = dg::integrateERK( "Dormand-Prince-7-4-5", m_rhs, 0., m_e1, 1., m_y, 0., dg::pid_control, dg::l2norm, 1e-8, 1e-10);
        std::cout << "Time iterations  " << time_iter  << "\n";
        dg::blas2::gemv(m_R, m_y, x);  // x =  R T^(-1/2) e_1   

        return m_max_iter;
    }
  private:
    value_type m_alpha;
    std::vector<value_type> m_v0, m_vp, m_vm;
    ContainerType m_r, m_ap, m_p, m_rh;
    SubContainerType m_e1, m_y;
    unsigned m_max_iter;
    CooMatrix m_R, m_Tinv;      
    dg::InvTridiag<SubContainerType, DiaMatrix, CooMatrix> m_invtridiag;
};
