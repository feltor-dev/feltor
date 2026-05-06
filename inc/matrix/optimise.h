#pragma once
#include "dg/algorithm.h"
#include "tridiaginv.h" // lapack wrapper

namespace dg{
namespace mat{
/**
 * @brief Newton iteration
 *
 * @tparam Gradient a callable with signature <tt>void operator()(const ContainerType& x0, ContainerType& grad)</tt>
 * @tparam InvHessian a callable with signature <tt>void operator()(const ContainerType& x0, const ContainerType& grad, ContainerType& p)</tt>
 * @param grad The gradient of the target function
 * @param invhess The inverse Hessian matrix
 * @param x0 initial guess on input, solution on output
 * @param tol succes condition is \f$ ||\nabla f(x_k)|| < \epsilon\f$
 * @param max_iter Maximum number of allowed iterations
 * @return Number of iterations
 *
 * A minimization algorithm based on the recursion
 *   \f[
 *   x_{k+1} = x_k - H^{-1}(x_k) \vec g (x_k)
 *   \f]
 * where \f$H\f$ is the Hessian matrix and \f$\vec g(x_k) = \nabla f |_{x_k}\f$ is the gradient of \f$f(\vec x)\f$.
 * The Newton iterate is essentially a root finding routine for the gradient of \f$f(\vec x)\f$.
 *
 * This already highlights the problem of using Newton for minimization:
 * - it can also find maxima and (more importantly)
 * - **it finds saddle points**.
 *
 * In essence Newton only works **if the Hessian is positive definite** at all search points **including the initial guess**.
 * Neither of these conditions is met often in practise, i.e. what we observe often is that *
 * the Newton method is attracted by saddle points and moves in the wrong diretion
 *
 * @section wolfe (Failed, not implemented) Line search and the Wolfe conditions
 * The Newton iterate produces a direction
 * \f[
 * p_k = - H_k^{-1} \vec g_k
 * \f]
 * The idea of a **line search algorithm** is to introduce a step-length \f$\alpha\f$ into the iterates:
 * \f[
 * x_{k+1} = x_k + \alpha_k p_k
 * \f]
 * and to find \f$\alpha\f$ through the condition
 * \f[
 * \alpha_k = \text{arg min}_{\alpha} f(\vec x_k + \alpha \vec p_k) \equiv \text{arg min}_{\alpha} \phi_k(\alpha)
 * \f]
 *
 * Typically, this minimization is not solved exactly. It is solved only to the
 * degree that the **Wolfe conditions** are satisfied
 * \f[
 * \begin{align}
 * \phi(\alpha) &\leq \phi(0) + c_1 \alpha \phi'(0)\\
 * \phi'(\alpha) &\geq c_2  \phi'(0)
 * \end{align}
 * \f]
 * with \f$0<c_1<c_2<1\f$ and in practise \f$c_1 = 10^{-4}\f$ and \f$c_2 = 0.9\f$.
 * A slightly stronger version are **the strong Wolfe conditions**
 *
 * \f[
 * \begin{align}
 * \phi(\alpha) &\leq \phi(0) + c_1 \alpha \phi'(0)\\
 * |\phi'(\alpha)| &\leq |c_2  \phi'(0)|
 * \end{align}
 * \f]
 *
 * The problem with this is that
 * - it is quite tedious to come up with an efficient line search that comes up
 *   with an efficient step length satisfying the Wolfe conditins
 * - for example: negative Eigenvalues in \f$H\f$ may lead to \f$\vec p_k =
 *   -H^{-1}\vec g_k \f$ **not a descent direction** (but \f$p_k\f$ being a descent
 *   direction is necessary for the existence of \f$\alpha_k >0\f$ satisfying the
 *   Wolfe conditions)
 *
 *
 * @ingroup opt
 * @sa "Numerical optimization" by Nocedal & Wright (Springer 2006)
 * @sa https://arxiv.org/pdf/1406.2572
 * @sa https://www.scientific.net/AMM.347-350.2586
 */
template<class Gradient, class InvHessian, class ContainerType>
unsigned newton( Gradient grad, InvHessian invhess,
    ContainerType& x0, double tol = 1e-5, unsigned max_iter = 1000)
{
    ContainerType jj(x0), p(x0), test(x0);
    for ( unsigned i=0; i<max_iter; i++)
    {
        grad( x0, jj);
        invhess( x0, jj, p);
        double alpha = 1.;
        dg::blas1::axpby( -alpha, p, 1., x0);
        double err = sqrt(dg::blas1::dot( jj, jj));
        if (err < tol)
            return i;
    }
    return max_iter;
}

/**
 * @brief The Levenberg Marquardt algorithm
 *
 * @tparam Func Callable with signature <tt>void operator()(ContainerType0& x0, ContainerType1& rs)</tt>
 * @tparam Jacobian Callable with signature <tt>void operator()(const
 * ContainerType0& x0, std::vector<ContainerType1>& jacs)</tt> (no need to resize the given jacs, its size is <tt>x0.size()</tt>)
 * @tparam ContainerType0 Typically a host container
 * (Determines the 1st argument type of \c Func and \c Jacobian)
 * @tparam ContainerType1 Determine the target architecture to run the algorithm
 * on (and the 2nd argument type of \c Func and \c Jacobian)
 * @param fun Compute the vector of residuals \f$ r_j\f$ given parameters \c x
 * @param jac Compute the gradients of the residuals: each element i of the outer
 * \c std::vector contains the gradient \f$ \partial \vec r/ \partial x_i\f$
 * @param x0 The initial guess on input, solution on output. The size
 * of \c x0 determines the number of free parameters \f$n \f$ to optimize.
 * @param copyable Determine the size \f$m\f$ and type of the second arguments
 * of \c fun and \c jac The contents are irrelevant, just the size and type are
 * important.
 * @param tol Tolerance determines termination condition
        <tt>if( ||p_k|| <= tol*(||x_k|| + 1.))</tt>
 * @param max_iter Maximum number of iterations
 * @return Number of iterations
 *
 *
 * @section trust Trust region algorithms

 * A line search algorithm first chooses an appropriate descent
 * direction and then tries to find the optimal step length. A trust region
 * algorithm first chooses a (maximum) step length and then tries to find the
 * direction that minimizes the function. Both algorithms (at least of Newton
 * type) assume a Taylor expansion of the function
 * \f[
 * f(x_k + p) \approx f(x_k) +  g_k^T p + \frac{1}{2} p^T H_k p \equiv m_k(p)
 * \f]
 * A trust region algorithm is an algorithm that minimizes \f$m_k(p)\f$ over
 * \f$p\f$ with the condition that \f$||p|| \leq \Delta_k\f$ where
 * \f$\Delta_k\f$ is the trust region radius:
 * \f[
 * \begin{align}
 * p &= \text{arg min}_p m_k(p) \\
 * ||p || &\leq \Delta_k
 * \end{align}
 * \f]
 *
 * It turns out somewhat surprisingly that even though \f$m_k(p)\f$ is a
 * quadratic function a closed solution is somewhat non-trivial (at least the
 * literature immediately goes on to only finding approximate solutions)
 *
 * Let us first state an important result: **The trust region minimization has
 * no solution** \f$p\f$ **with** \f$||p|| = \Delta_k\f$ **if and only if** \f$H\f$ **is
 * positive definite and** \f$|| H_k^{-1} g_k|| < \Delta_k\f$. This result
 * means we can focus on the case were the minimization has a solution \f$||p||
 * = \Delta_k\f$.


 * @section lsq Least squares problem
 *
 * Assume our objective function (the minimum of which we seek) has the form
 * \f[
 * f( x) = \frac{1}{2}\sum_{j=1}^m r_j(x)^2
 * \f]
 * where \f$r_j(x)\f$ (the residuals) are smooth functions and \f$ x\f$ are the **parameters**
 * (of the fit function) and we assume
 * \f$m>n\f$ with \f$n\f$ the number of parameters (or dimension of \f$x\f$).
 * Typically, this appears if \f$ f\f$ is an error function and \f$ x\f$ are a
 * set of \f$n\f$ parameters that we try to optimize to reduce the residuals at
 * various test points j.  In the literature minimization of this form is
 * treated under **nonlinear least squares problems** (because the \f$ r_j\f$
 * depend nonlinearly on the parameters \f$x\f$, if the residuals depend
 * linearly on the parameters then the problem is a linear least squares problem
 * that can be solved straighforwardly).
 * @sa dg::least_squares
 *
 * Let us define
 * \f[
 * J = \frac{\partial r_j}{\partial x_i}
 * \f]
 * Then (with \f$r(x) = (r_1(x), r_2(x) , ...)\f$)
 * \f[
 * \begin{align}
 * g(x) =& \nabla f(x) = J^T r(x)\\
 * H(x) =& J^T J + \sum_{j=1}^m r_j(x) \nabla\nabla r_j(x)
 * \end{align}
 * \f]
 * Often, the second term in the Hessian \f$H(x)\f$ can be neglected, which is
 * the distinctive feature of least squares problems.
 *
 * In fact, Newton's method where \f$H\f$ is replaced with \f$J^T J\f$ is
 * called **Gauss-Newton** method and the search direction is the solution to
 * the linear least squares problem (at each step \f$k\f$)
 * \f[
 * \text{min}_p || J p + r||^2
 * \f]
 * Of course, in practice Gauss-Newton still needs to be combined with a line-search method.
 *

 *
 * @section lm Levenberg-Marquardt algorithm (for nonlinear least squares problems)
 *
 * The LM algorith is a trust region algorithm where the Hessian is replaced by
 * \f$H\approx J^T J\f$ and the Jacobian is the \f$m\times n\f$ derivatives of
 * the \f$r_i(x)\f$. Doing so, all Eigenvalues are \f$\lambda_i \geq 0\f$.
 *
 * The way to solve a constraint minimization problem (the trust region
 * minimization above) is through Lagrange multipliers
 * \f[
 * L(p, \lambda) = r^T J p + \frac{1}{2} p^T J^T J p + \lambda( ||p||_W^2 -
 * \Delta^2)/2 \simeq ||Jp +r||^2/2 + \lambda( ||p||_W^2 - \Delta^2)/2
 * \f]
 * The Euler Lagrange equations read
 * \f[
 * \begin{align}
 * J^T r + J^T J p + \lambda W p =& 0 \\
 * p^T W p =& \Delta^2
 * \end{align}
 * \f]
 * The proposed search direction is therefore
 * \f[
 * p = -( J^T J + \lambda W)^{-1} J^T r
 * \f]
 * with \f$\lambda > 0\f$. The case \f$\lambda = 0\f$ is equivalent to a
 * Gauss-Newton step.  For \f$\lambda\ll 1\f$ the search direction converges to
 * that of Gauss-Newton while for \f$\lambda\gg 1\f$ the \f$p\f$ is the
 * steepest descent direction.  The basis for this algorithm is the result that
 * there exists a \f$\Delta\f$ related to \f$\lambda\f$ such that \f$p\f$ is
 * the solution to the minimization
 * \f[
 * \begin{align}
 * p =& \text{arg min}_p ||J p + r||^2/2 \\
 * ||p||_W\leq &\Delta(\lambda)
 * \end{align}
 * \f]
 * Note that there is an inverse relation between \f$\Delta\f$ and
 * \f$\lambda\f$, i.e. if \f$\lambda\gg 1\f$ then \f$\Delta\ll 1\f$.  From a
 * (generalized) Eigenvalue decomposition of \f$H\equiv J^T J = WE_H \Lambda_H
 * E_H^{-1}\f$ (columns of \f$ E_H\f$ are Eigenvectors, \f$ E_H^T W E_H = 1\f$ and \f$ E_H^{-1} = E_H^T W\f$)
 * we get with \f$\bar g = E_H^T J^T r \equiv E_H^T g\f$ (and \f$H v_j = \lambda_j W v_j\f$)
 * \f[
 * \begin{align}
 * p =& -E_H(\Lambda + \lambda 1)^{-1} E_H^T J^T r \equiv E_H \bar p\\
 * p^T W p =& \sum_j \frac{\bar g_j^2}{(\lambda_j + \lambda)^2}
 * \end{align}
 * \f]
 * Finally, note that in practice one needs to take care to make the algorithm
 * scale invariant by using \f$|| W p ||\leq \Delta\f$ in the constrained
 * minimization.
 *
 * Two ideas seem to exist going forward. The original paper by Marquardt
 * suggests to use \f$\lambda\f$ directly as the adaptive parameter (i.e. a
 * substitute for the trust radius). Some notes suggest that in this case the
 * dogleg method is preferable. Later, a paper by Moré (1977) "The
 * Levenberg-Marquardt algorithm: Implementation and Theory" suggests to
 * consider \f$\Delta\f$ as given and iteratively finding \f$\lambda\f$ such
 * that
 * \f[
 * ||p||_W = \Delta
 * \f]
 * (which is attributed to Hebden). They suggest to use QR decomposition and
 * cleverly combine that with a root finding for \f$\lambda\f$. In our own
 * implementation we use lapack's Eigenvalue decomposition of \f$J^T J\f$ and
 * combine it with a 1d Newton root finding, to find \f$\lambda\f$. The
 * disadvantage to compute \f$J^TJ\f$ is not too high compared to computing QR
 * of \f$J\f$ for low dimensions. Nocedal&Wright suggest to use the target
 * function
 * \f[
 * \phi(\lambda)= 1/\Delta - 1/||p(\lambda)||_W
 * \f]
 * for the root finding because \f$\phi(\lambda)\simeq \lambda\f$ near the
 * optimum, however one must take care because the target function is not
 * differentiable at \f$\lambda = -\lambda_i\f$. \f$\lambda_0 > -\lambda_1\f$
 * where \f$\lambda_1\f$ is the smallest Eigenvalue should be a good starting
 * value.
 *
 * For this to work with simple Newton iteration we need the derivative
 * \f[
 * \partial_\lambda \phi = - \partial_\lambda ||p||_W = - ||p||_W^{-3} \sum_j \frac{\bar g_j^2}{(\lambda_j + \lambda)^3} < 0 \text{ for } \lambda \geq 0
 * \f]
 * We here see that if \f$ \phi(0) <= 0\f$ we must immediately choose \f$
 * \lambda = 0\f$ since no \f$\lambda > 0 \f$ exists such that \f$\phi(\lambda)
 * = 0\f$. In this case we have \f$||p||_W <\Delta\f$. In the case \f$\phi(0)>0\f$
 * we can use Newton iteration to find \f$ \lambda>0\f$ such that 
 * \f$ | ||p||_W  - \Delta| \leq \sigma \Delta\f$. i.e.
 * \f[ |\phi(\lambda) | \leq \frac{\sigma}{||p||_W}
 * \f]
 * where we choose \f$ \sigma = 10^{-4}\f$ as the stopping criterion
 *
 * @subsection adapt Adaptive choice of trust region radius
 * If we implement for given \f$\Delta_k\f$ the above procedure we get a search
 * vector \f$p_k\f$.  Similar to adaptive timestep algorithms we can define an
 * error quantity comparing \f$f(x+p_k)\f$ with \f$m_k(p_k)\f$.  In practice
 * this is done using the ratio
 * \f[
 * \begin{align}
 * \rho_k = \frac{f(x_k) - f(x_k+p_k)}{m_k(0) - m_k(p_k)}
 * \end{align}
 * \f]
 * with \f$ m_k(0) - m_k(p_k) = -0.5 p^T J^T J p - g_k^T p = -0.5 \bar p \Lambda \bar p + (J^T J p + \lambda W p)^T p = 0.5\bar p \Lambda \bar p +\lambda ||p||_W^2\f$.
 * If \f$\rho_k\approx 1\f$ it means that \f$f(x)\f$ is very well approximated
 * by \f$m_k\f$ and we can extend the trust radius \f$\Delta_k\f$
 * in the next step.
 * If \f$\rho_k \leq 0\f$ or \f$\rho_k \ll 1\f$ it means that \f$m_k\f$ was
 * such a poor representation of \f$f\f$ that we must reject the proposed step
 * and reduce the trust region.
 * We use Algorithm 4.1. of [Nocedal&Wright]
 * @subsection weights Choice of weights
 * The weights are chosen according to [Nocedal&Wright, More1967] as the decreasing sequence
 * \f[
 * \begin{align}
 *  W^{k} =& \text{diag}( W_0^k, ..., W_{n-1}^k)\\
 *  W_i^k = & \text{max}( || J_i||^2, W_i^{k-1})
 *  \end{align}
 *  \f]
 *
 *
 * @ingroup opt
 * @sa "Numerical optimization" by Nocedal & Wright (Springer 2006)
 */
template<class Func, class Jacobian, class ContainerType0, class ContainerType1>
unsigned levenberg_marquardt( Func fun, Jacobian jac,
    ContainerType0& x0,
    const ContainerType1& copyable, // size of return of fun
    double tol = 1e-8, unsigned max_iter = 1000)
{
    unsigned num_p = x0.size();
    auto x1 = x0, W = x0;
    auto rs(copyable), rs1(rs);
    std::vector<ContainerType1> jacs(num_p, copyable);
    dg::SquareMatrix<double> HH(num_p, 0.), evHH( HH), evHH_T(HH), WW(HH), syWW(WW);
    thrust::host_vector<double> evs( num_p), grad(num_p), gradbar(num_p),
        pk(num_p), pkbar(num_p);
    thrust::host_vector<double> work( 3*num_p-1);
    // init loop
    fun( x0, rs);
    jac( x0, jacs);
    double delta = 0;
    for( unsigned p=0; p<num_p; p++)
    {
        WW(p,p) = W[p] = dg::blas1::dot( jacs[p], jacs[p]);
    }
    for( unsigned p=0; p<num_p; p++)
    {
        grad[p] = dg::blas1::dot( jacs[p], rs);
        delta += grad[p]*grad[p]/W[p];
    }
    //std::cout << "Norm gT g/W^2 " << sqrt(delta)<<"\n";
    double f0 = dg::blas1::dot( rs, rs);
    delta = 0.25*f0/sqrt(delta);
    //std::cout << "initial delta " << delta<<"\n";
    double normx0 = sqrt(dg::blas1::dot( x0, x0));
    for ( unsigned k=0; k<max_iter; k++)
    {
        // 1. Solve (J^T J + lambda W)p = -J^T r with lambda : ||p||_W leq Delta
        // W[p] = max{ W_{k-1}, H_pp}
        // In: jacs, rs; Out: pk, normpk, lambda
        for( unsigned l=0; l<num_p; l++)
        {
            for( unsigned j=l; j<num_p; j++)
                HH(j,l) = HH(l,j) = dg::blas1::dot( jacs[l], jacs[j]);
            WW(l,l) = std::max( WW(l,l), HH(l,l)); // After Mor´e the weights must not decrease
        }
        // !!! sygv destroys WW on output (so use copy syWW instead) !!!
        syWW = WW;
        lapack::sygv( 1, 'V', 'U', num_p, HH.data(), num_p, syWW.data(), num_p, evs, work);
        evHH_T = HH;
        evHH = HH.transpose();
        //std::cout << "#########Iteration "<<k<<"\n";
        //std::cout << "Eigenvalues are \n";
        //for( unsigned p=0; p<num_p; p++)
        //    std::cout << "p EV "<<evs[p]<<"\n";
        //std::cout << "Weights are \n";
        //for( unsigned p=0; p<num_p; p++)
        //    std::cout << "W "<<WW(p,p)<<"\n";
        dg::blas2::gemv( evHH_T, grad, gradbar); // !! sygv gives A = WE_A Lambda E_A^TW
        double normp=0;
        auto target = [&]( double lambda)
        {
            // safeguard against 0 Eigenvalue!
            for( unsigned p=0; p<num_p; p++)
                pkbar[p]= -gradbar[p]/(evs[p] +lambda == 0 ? 1e-16 : evs[p]+lambda);
            normp = sqrt(dg::blas1::dot( pkbar, pkbar));
            return 1./delta - 1./normp;
        };
        auto dtarget = [&]( double lambda)
        {
            // safeguard against 0 Eigenvalue!
            double dnorm =0;
            for( unsigned p=0; p<num_p; p++)
                dnorm += pkbar[p]*pkbar[p]/(evs[p]+lambda == 0 ? 1e-16 : evs[p]+lambda);
            return - dnorm/normp/normp/normp;
        };
        double lambda = 0;
        const double sigma = 1e-4; // tolerance for Newton algorithm
        const unsigned max_newton = 100;
        double phi = target(lambda);
        if( phi > 0) // positive lambda only exist if phi(0) > 0 since dlambda < 0 for all lambda >= 0
        {
            for( unsigned i=0; i<max_newton; i++) // Safeguard
            {
                // if ||p|| leq delta ( 1+sigma)
                if ( fabs(phi) <= sigma/normp || i == max_newton-1)
                    break;
                // The first step is always to the right ... (phi > 0 , lambda < 0)
                lambda += - phi/dtarget(lambda);
                phi = target(lambda);
            }
        }
        dg::blas2::gemv(evHH, pkbar, pk);
        // target(lambda) updates grad and normp
        // 2. Check termination
        //std::cout << "Real Norm p "<<sqrt(dg::blas1::dot( pk, pk))<<" normx0 "<<normx0<<"\n";
        if( sqrt(dg::blas1::dot( pk,pk)) <= tol*(normx0 + 1.))
        {
            dg::blas1::axpby( 1., pk , 1., x0);
            return k;
        }
        // 3. Compute Ratio rhok
        // don't overwrite (x0, rs) because we might reject
        dg::blas1::axpby( 1., pk , 1., x0, x1);
        //for( unsigned l=0; l<num_p; l++)
        //    std::cout << "x0 "<<l<<" "<<x0[l]<<"\n";
        //for( unsigned l=0; l<num_p; l++)
        //    std::cout << "x1 "<<l<<" "<<x1[l]<<"\n";
        double f1 = 0;
        try{
            fun( x1, rs1);
            f1 = dg::blas1::dot( rs1, rs1);
        }
        catch( dg::Error& e)
        {
            // The only reason to throw an Error here is if there is a NaN or Inf
            // In this case will rs1 with huge values such the step is rejected
            dg::blas1::copy( 1e300, rs1);
            f1 = 1e300;
            //std::cout << "############################# STEP CONTAINS NAN OR INF! REJECT!\n";
        }
        // compute (Jp)^2
        double jp = dg::blas2::dot( pkbar, evs, pkbar);
        //std::cout<< "f0 "<<f0<<" f1 "<<f1<<"\n";
        double rhok = (1.-f1/f0)/( jp/f0 + 2*lambda*normp*normp/f0);
        //std::cout << "Actual    reduction "<<f0-f1<<"\n";
        //std::cout << "Predicted reduction "<<( jp + 2*lambda*normp*normp)<<"\n";
        //std::cout << "Ratio "<<rhok<<"\n";
        // Algorithm 4.1 from Nocedal & Wright
        if( rhok < 0.25)
            delta = 0.25*delta;
        else
        {
            // Mor´e has slightly different conditions
            if( rhok > 0.75 && lambda > 0) // steps lies on the trust region boundary
                delta = 2*delta;
            // else delta remains unchanged
        }
        const double eta = 1e-4; // when step is rejected
        if ( rhok > eta)
        {
            x0 = x1;
            // update all quantities
            f0 = f1;
            using std::swap;
            swap( rs, rs1);
            jac( x0, jacs);
            normx0 = sqrt(dg::blas1::dot( x0, x0));
            for( unsigned l=0; l<num_p; l++)
            {
                grad[l] = dg::blas1::dot( jacs[l], rs);
                //std::cout << "grad "<<l<<" "<<grad[l]<<"\n";
            }
        }
        else
        {
        //    std::cout << "REJECTED\n";
        // else step is rejected
        }
    }
    return max_iter;
}
} //namespace mat
} //namespace dg
