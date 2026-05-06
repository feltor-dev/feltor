#pragma once
#include "dg/algorithm.h"
#include "optimise.h"

/**
* @brief Functions for optimizing Contours
*/


namespace dg{
namespace mat{

///@cond
//The following is an implementation of the python notebook

double f_alpha( double alphabar, double lambda = 1.)
{
    return 1-exp( -lambda*alphabar);
}
double finv_alpha( double alpha, double lambda = 1.)
{
    return - log( 1. - alpha)/lambda;
}
double df_alpha( double alphabar, double lambda = 1)
{
    return lambda*exp( -alphabar*lambda);
}
double ddf_alpha( double alphabar, double lambda = 1)
{
    return -lambda*lambda*exp( -alphabar*lambda);
}

// There are 2*N real params and there will be N/2 nodes
std::pair<std::vector<thrust::complex<double>>,std::vector<thrust::complex<double>>>
    weights_and_nodes_talbot( unsigned N, const std::vector<double>& params)
{
    thrust::complex<double> I( 0,1);
    double h = M_PI/(double)N;
    unsigned n = N/2;
    std::vector<thrust::complex<double>> zk(N/2);
    std::vector<thrust::complex<double>> wk(N/2);
    double mu = params[0];
    double sigma = params[1];
    double nu = params[2];
    double alphabar = params[3];
    double alpha = f_alpha( alphabar);

    for( unsigned k=0; k<n; k++)
    {
        auto x = thrust::complex<double>( 2*h*k + h);
        zk[k] = thrust::complex<double>(N)*(-sigma + mu*x/tan(alpha*x) + nu*I*x);
        auto vk = thrust::complex<double>(N)*(mu/tan(alpha*x) - mu*x*alpha/sin(alpha*x)/sin(alpha*x) + nu*I);
        wk[k] = I*h/M_PI*vk;
    }
    return std::make_pair( zk, wk);
}

std::pair<std::vector<thrust::complex<double>>,std::vector<thrust::complex<double>>>
    jacobian_talbot( unsigned N, const std::vector<double>& params)
{
    thrust::complex<double> I( 0,1);
    double h = M_PI/(double)N;
    unsigned n = N/2;
    std::vector<thrust::complex<double>> dzk(4*n);
    std::vector<thrust::complex<double>> dwk(4*n);
    double mu = params[0];
    double alphabar = params[3];
    double alpha = f_alpha( alphabar);
    double dalpha = df_alpha( alphabar);
    for( unsigned k=0; k<n; k++)
    {
        auto Nc = thrust::complex<double>(N);
        auto x = thrust::complex<double>( 2*h*k + h);
        dzk[0*n+k] = Nc*( x/tan(alpha*x));
        dzk[1*n+k] = Nc*(-1. );
        dzk[2*n+k] = Nc*(I*x);
        dzk[3*n+k] = Nc*(-mu*x*x/sin(alpha*x)/sin(alpha*x));
        dzk[3*n+k] *= dalpha;
        dwk[0*n+k] = I*h/M_PI*Nc*(1./tan(alpha*x) - x*alpha/sin(alpha*x)/sin(alpha*x) );
        dwk[1*n+k] = 0;
        dwk[2*n+k] = I*h/M_PI*Nc*(I);
        dwk[3*n+k] = I*h/M_PI*Nc*(-2.*mu*x/sin(alpha*x)/sin(alpha*x) +
                             mu*x*x*alpha*2.*cos(alpha*x)/sin(alpha*x)/sin(alpha*x)/sin(alpha*x) );
        dwk[3*n+k] *= dalpha;

    }
    return std::make_pair( dzk, dwk);
}

std::vector<double> weights_and_nodes2params( const
std::pair<std::vector<thrust::complex<double>>,std::vector<thrust::complex<double>>>& zkwk)
{
    const auto& zk = zkwk.first;
    const auto& wk = zkwk.second;
    std::vector<double> params( 4*zk.size());
    unsigned n = zk.size();
    for( unsigned i=0; i<n; i++)
    {
        params[0*n+i] = zk[i].real();
        params[1*n+i] = zk[i].imag();
    }
    for( unsigned i=0; i<n; i++)
    {
        params[2*n+i] = wk[i].real();
        params[3*n+i] = wk[i].imag();
    }
    return params;
}


// There are 2*N real params and there will be N/2 nodes
std::pair<std::vector<thrust::complex<double>>,std::vector<thrust::complex<double>>>
    weights_and_nodes_identity( unsigned N, const std::vector<double>& params )
{
    if( N != params.size()/2)
        throw dg::Error(dg::Message(_ping_)<<"N "<<N<<" must match 0.5 params.size "<<params.size()/2<<"!");
    unsigned n = N/2;
    std::vector<thrust::complex<double>> zk(n);
    std::vector<thrust::complex<double>> wk(n);
    for( unsigned i=0; i<n; i++)
        zk[i] = thrust::complex<double>(params[0*n+i], params[1*n+i]);
    for( unsigned i=0; i<n; i++)
        wk[i] = thrust::complex<double>(params[2*n+i], params[3*n+i]);
    return std::make_pair( zk, wk);
}
std::pair<std::vector<thrust::complex<double>>,std::vector<thrust::complex<double>>>
    jacobian_identity( unsigned N, const std::vector<double>&)
{
    unsigned n = N/2;
    std::vector<thrust::complex<double>> dzk(4*n*n, {0.});
    std::vector<thrust::complex<double>> dwk(4*n*n, {0.});
    for( unsigned k=0; k<n; k++)
    {
        dzk[(0*n+k)*n + k] = thrust::complex<double>(1,0);
        dzk[(1*n+k)*n + k] = thrust::complex<double>(0,1);
        dwk[(2*n+k)*n + k] = thrust::complex<double>(1,0);
        dwk[(3*n+k)*n + k] = thrust::complex<double>(0,1);
    }
    return std::make_pair( dzk, dwk);
}


/////////////////////////////////////////////////////////////////////////////

// target_result
// target_error
//
struct LeastSquaresCauchyError
{
    /**! @brief
     *
     * @param func must accept complex values as arguments
     */
    template<class Generator, class UnaryFunction>
    LeastSquaresCauchyError( unsigned N, Generator generate, UnaryFunction func, const
        std::vector<double>& rrs, const std::vector<double>& lls):
        m_N(N), m_func( func), m_generate(generate),
        m_rrs(rrs), m_lls(lls), m_exact( lls.size()*rrs.size()),
        m_func_rrs( N/2*rrs.size())
        {
            m_nl= lls.size();
            m_nr= rrs.size();
            for( unsigned i=0; i<m_nl; i++)
                for( unsigned j=0; j<m_nr; j++)
                    m_exact[i*m_nr+j] = (func( -thrust::complex<double>(m_lls[i]*m_rrs[j]))).real();
        }
    void result( const std::vector<double>& params, std::vector<double>& result)
    {
        auto pair = m_generate( m_N, params);
        const auto& zk = pair.first;
        const auto& wk = pair.second;
        unsigned n = zk.size();
        dg::blas1::copy( 0, result);
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<m_nr; j++)
                m_func_rrs[k*m_nr + j] = wk[k]*m_func( m_rrs[j]*zk[k]);
        for( int k=n-1; k>=0; k--)
            for( unsigned i=0; i<m_nl; i++)
                for( unsigned j=0; j<m_nr; j++)
                    result[i*m_nr+j]+= 2*(m_func_rrs[k*m_nr+j]/(-m_lls[i] - zk[k])).real();
    }
    void error( const std::vector<double>& params, std::vector<double>& err)
    {
        result( params, err);
        dg::blas1::axpby( -1., m_exact, 1., err);
    }
    void set_order ( unsigned order){
        m_order = order;
    }

    void operator()( const std::vector<double>& params, std::vector<double>& res)
    {
        //dg::Timer t;
        //t.tic();
        result( params, res);
        //t.toc();
        //std::cout << "Computing result "<<t.diff()<<"\n";
        //t.tic();
        dg::blas1::axpby( -1., m_exact, 1., res);
        if ( m_order == 2)
            dg::blas1::pointwiseDot( res, res, res); // least squares converge better in r^4

        //t.toc();
        //std::cout << "Computing error  "<<t.diff()<<"\n";
    }
    private:
    unsigned m_N, m_nl, m_nr;
    std::function<thrust::complex<double>(thrust::complex<double>)> m_func;
    std::function<std::pair<std::vector<thrust::complex<double>>, std::vector<thrust::complex<double>>>(unsigned, const std::vector<double>&)> m_generate;
    std::vector<double> m_rrs, m_lls, m_exact;
    std::vector<thrust::complex<double>> m_func_rrs;
    unsigned m_order = 2;
};

// target_jacobian
struct LeastSquaresCauchyJacobian
{
    /**! @brief
     *
     * @param func must accept complex values as arguments
     */
    template<class Generator, class GeneratorJac, class UnaryFunction, class UnaryFunctionD>
    LeastSquaresCauchyJacobian( unsigned N, Generator generate, GeneratorJac generateJac,
        UnaryFunction func, UnaryFunctionD dxlnfunc, const std::vector<double>& rrs, const std::vector<double>& lls):
        m_N(N), m_func( func), m_dxlnfunc(dxlnfunc),
        m_generate(generate), m_generateJac( generateJac),
        m_rrs(rrs), m_lls(lls), m_exact( lls.size()*rrs.size()),
        m_result( m_exact),
        m_func_rrs(N/2*rrs.size())
        {
            m_nl= lls.size();
            m_nr= rrs.size();
            for( unsigned i=0; i<m_nl; i++)
                for( unsigned j=0; j<m_nr; j++)
                    m_exact[i*m_nr+j] = (func( -thrust::complex<double>(m_lls[i]*m_rrs[j]))).real();
        }

    void set_order( unsigned order) { m_order = order;}
    void operator()( const std::vector<double>& params, std::vector<std::vector<double>>& jac)
    {
        //dg::Timer t;
        //t.tic();
        auto pair = m_generate( m_N, params);
        const auto& zk = pair.first;
        const auto& wk = pair.second;
        auto Jacpair = m_generateJac( m_N, params);
        const auto& dzk = Jacpair.first;
        const auto& dwk = Jacpair.second;
        unsigned n = zk.size();
        //t.toc();
        //std::cout <<"Generating pairs took "<<t.diff()<<"\n";
        //t.tic();
        dg::blas1::copy( 0, m_result);
        for( unsigned k=0; k<n; k++)
            for( unsigned j=0; j<m_nr; j++)
                m_func_rrs[k*m_nr+j] = wk[k]*m_func( m_rrs[j]*zk[k]);
        for( int k=n-1; k>=0; k--)
            for( unsigned i=0; i<m_nl; i++)
                for( unsigned j=0; j<m_nr; j++)
                    m_result[i*m_nr+j]+= 2*(m_func_rrs[k*m_nr+j]/(-m_lls[i] - zk[k])).real();
        dg::blas1::axpby( -1., m_exact, 1., m_result);
        //t.toc();
        //std::cout <<"Computing result "<<t.diff()<<"\n";
        //t.tic();
        std::vector<thrust::complex<double>> tmp( m_func_rrs.size());
        dg::blas1::copy( 0, jac);
        for( unsigned p=0; p<params.size(); p++)
        {
            for( unsigned k=0; k<n; k++)
                for( unsigned j=0; j<m_nr; j++)
                    tmp[k*m_nr+j] =  dwk[p*n+k]/wk[k] +
                            m_rrs[j]*dzk[p*n+k]*m_dxlnfunc(m_rrs[j]*zk[k]);
            for( int k=n-1; k>=0; k--)
                for( unsigned i=0; i<m_nl; i++)
                    for( unsigned j=0; j<m_nr; j++)
                        jac[p][i*m_nr+j]+= 2*(m_func_rrs[k*m_nr+j]*(
                        dzk[p*n+k]/(-m_lls[i] - zk[k]) +
                        tmp[k*m_nr+j])/(-m_lls[i] - zk[k])).real();
            if( m_order == 2)
                dg::blas1::pointwiseDot( 2., jac[p], m_result, 0., jac[p]);
        }
        //t.toc();
        //std::cout <<"Computing jacobian "<<t.diff()<<"\n";
    }
    private:
    unsigned m_N, m_nl, m_nr;
    std::function<thrust::complex<double>(thrust::complex<double>)> m_func, m_dxlnfunc;
    std::function<std::pair<std::vector<thrust::complex<double>>, std::vector<thrust::complex<double>>>(unsigned, const std::vector<double>&)> m_generate, m_generateJac;
    std::vector<double> m_rrs, m_lls, m_exact, m_result;
    std::vector<thrust::complex<double>> m_func_rrs;
    unsigned m_order = 2;
};

std::vector<double> generate_range( double min, double max, unsigned per_order = 20, bool with_zero = false)
{
    unsigned orders = unsigned (log10(max) - log10(min));
    if ( orders == 0)
        orders = 1;
    std::vector<double> range(orders*per_order);
    unsigned N = range.size();
    double h = (log10(max) - log10(min))/double(N-1);
    for( unsigned i=0; i<N; i++)
        range[i] = pow( 10.0, log10(min) + i*h);
    if( with_zero)
        range.insert( range.begin(), 0);
    return range;
}
///////////////////////////////////////////////////////////////////////////////////////

// Not such a great idea:
// Mainly because of the lack of the accuracy of our blas1::dot functions
template<class ContainerType, class UnaryFunc>
void result_talbot( unsigned N, double rrs, double lls,
    const std::vector<ContainerType>& ps,
    UnaryFunc func,
    ContainerType& result)
{
    dg::blas1::subroutine( [N,rrs,lls,func]DG_DEVICE(
        double mu, double sigma, double nu, double alphabar, double& result)
    {
        thrust::complex<double> I( 0,1);
        double h = M_PI/(double)N;
        unsigned n = N/2;
        result = 0;
        double alpha = 1.-exp(-alphabar);
        for( int k=n-1; k>=0; k--)
        {
            double x = 2*h*k + h;
            double tanx = tan(alpha*x);
            double sinx = sin(alpha*x);
            thrust::complex<double> zk( N*(-sigma + mu*x/tanx), nu*x*N);
            thrust::complex<double> wk(-h/M_PI*N*nu, h*N/M_PI*(mu/tanx - mu*x*alpha/sinx/sinx));
            result+= 2*(wk*func(rrs*zk)/(-lls - zk)).real();
        }
    }, ps[0], ps[1], ps[2], ps[3], result);
}

template<class ContainerType, class UnaryFunc>
void error_talbot( unsigned N, const std::vector<double>& rrs, const std::vector<double>& lls,
    const std::vector<ContainerType>& ps,
    UnaryFunc f,
    ContainerType& error)
{
    dg::blas1::copy( 0, error);
    ContainerType tmp( error);
    for( unsigned i=0; i<lls.size();i++)
    for( unsigned j=0; j<rrs.size();j++)
    {
        result_talbot( N, rrs[j], lls[i], ps, f, tmp);
        dg::blas1::plus( tmp, -f(-thrust::complex<double>(lls[i]*rrs[j])).real());
        dg::blas1::pointwiseDot( 1., tmp, tmp, 1., error);
    }
}
///@endcond

///@cond

struct CauchyOptimizer
{
    CauchyOptimizer( )
    {
        re_init();
    }


    void set_verbose( bool verbose) { m_verbose = verbose;}

    /*!
     * @brief Compute optimal zk and wk based on given parameters
     *
     * Will only re-compute zk and wk if the currently used ones produces
     * an  error greater than the given tolerance (increase number of nodes)
     * or if the error is lower than the tolerance divided by e (decrease number of nodes).
     *
     * @param changed True if the returned nodes and weights are different
     * from the previous call to \c update_zkwk. False else. False on the first call.
     * @param lmin The minimum Eigenvalue greater than zero.
     * Use an estimate of the minimum Eigenvalue in case lmin is
     * zero (for the Laplacian a good estiamte is lmax / grid_size) (it is
     * exact if Nx = Ny), it is not so important to get the min EV right, if 0
     * is present
     * @param with_zero set to true if the actual min Eigenvalue is zero.
     * @param eps Maximum allowed absolute error. The optimizer will find
     * nodes and weights that keep the error smaller than \c eps
     * The optimizer may reduce the number of returned nodes and weights if the error
     * is much smaller than the desired \c eps
     * @return The pair <tt>(zk,wk)</tt>  The number of nodes is <tt>zkwk.first.size()</tt>
     */
    template<class UnaryFunc, class UnaryFuncD>
    const std::pair<std::vector<thrust::complex<double>>, std::vector<thrust::complex<double>>>& update_zkwk(
        bool& changed,
        UnaryFunc func, UnaryFuncD dxlnfunc, double lmin, double lmax,
        double dmin, double dmax, bool with_zero, double eps = 1e-4)
    {
#ifdef MPI_VERSION
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif //MPI
        //double error_tolerance_factor = 2.718; // == e
        double error_tolerance_factor = 10;
        // We assume that our optimal curve converges with eps \propto exp( -n
        // ) and thus an increase or decrease by one node decreases/increases
        // the error by e. However, there is a danger that in corner cases the
        // error interval [eps/e ; eps] cannot be reached so we increase the
        // tolerance to a safer 10!
        unsigned max_nodes = 36; // exp( - 36 ) Approx 1e-16

        // 1. Generate range
        auto rrs = dg::mat::generate_range( dmin, dmax, 20);
        auto lls = dg::mat::generate_range( lmin, lmax, 20, with_zero );
        std::vector<double> results( lls.size()*rrs.size());
        // 2. Test if currently used nodes are enough
        unsigned n = m_zkwk.first.size();
        dg::mat::LeastSquaresCauchyError
            Icauchy( 2*n, dg::mat::weights_and_nodes_identity, func, rrs, lls);
        Icauchy.set_order(1);
        Icauchy.error( m_paramsI, results);
        double current_eps = dg::blas1::reduce( results, -1e300, thrust::maximum<double>(), dg::ABS<double>() );
        if( current_eps > 100*eps ||  current_eps < eps/error_tolerance_factor) // far out of reach
            re_init(); // re-init
        changed = false;
        while( current_eps  > eps  && n < max_nodes )
        {
            // First optimize the Talbot curve
            dg::mat::LeastSquaresCauchyError
                 cauchy( 2*n, dg::mat::weights_and_nodes_talbot, func, rrs, lls);
            dg::mat::LeastSquaresCauchyJacobian
                 jac( 2*n, dg::mat::weights_and_nodes_talbot, dg::mat::jacobian_talbot, func, dxlnfunc, rrs, lls);
            // One can play between 1 and 2 here (2 seems to be slightly better)
            cauchy.set_order(2);
            jac.set_order(2);

            unsigned steps = levenberg_marquardt( cauchy, jac, m_params, results, 1e-5, 1000);
            if( m_verbose )
            {
                DG_RANK0 std::cout << "# Current n "<<n<<"\n";
                DG_RANK0 std::cout << "# Num steps in Levenberg Marquardt "<<steps<<"\n";
                cauchy.error( m_params, results);
                double cauchy_error = dg::blas1::dot( results, results);
                DG_RANK0 std::cout << "# Talbot error "<<cauchy_error<<" ";
                DG_RANK0 std::cout << "#  with params "<<m_params[0]<<" "<<m_params[1]<<" "<<m_params[2]<<" "<<m_params[3]<<"\n";
                double abs_max = dg::blas1::reduce( results, -1e300, thrust::maximum<double>(), dg::ABS<double>());
                DG_RANK0 std::cout << "# Abs max error "<< abs_max<<"\n";
            }

            // Second: from there find better values
            dg::mat::LeastSquaresCauchyError
                Icauchy( 2*n, dg::mat::weights_and_nodes_identity, func, rrs, lls);
            dg::mat::LeastSquaresCauchyJacobian
                Ijac( 2*n, dg::mat::weights_and_nodes_identity, dg::mat::jacobian_identity, func, dxlnfunc, rrs, lls);
            Icauchy.set_order(2);
            Ijac.set_order(2);
            // convert params to paramsI
            auto zkwk = dg::mat::weights_and_nodes_talbot( 2*n, m_params);
            m_paramsI = dg::mat::weights_and_nodes2params( zkwk);
            unsigned stepsI = levenberg_marquardt( Icauchy, Ijac, m_paramsI, results, 1e-5, 1000);
            Icauchy.error( m_paramsI, results);
            current_eps = dg::blas1::reduce( results, -1e300, thrust::maximum<double>(), dg::ABS<double>() );
            if( m_verbose)
            {
                double cauchy_error = dg::blas1::dot( results, results);
                DG_RANK0 std::cout << "# Num steps in Levenberg Marquardt Id "<<stepsI<<"\n";
                DG_RANK0 std::cout << "# Cauchy I error "<<cauchy_error<<"\n";
                DG_RANK0 std::cout << "# Abs max I error "<<current_eps<<"\n";
            }
            m_zkwk = dg::mat::weights_and_nodes_identity( 2*n, m_paramsI);
            changed = true;
            n++;
        }
        if( n >= max_nodes)
        {
            re_init();
            throw dg::Error( dg::Message(_ping_)<<"Error! Maximum number of nodes (36) reached! Tolerance "<<eps<<" cannot be reached! Current eps "<<current_eps);
        }
        m_called_previously = true;
        return m_zkwk;
    }
    unsigned num_nodes() const {return m_zkwk.first.size();}
    private:
    void re_init(){
        m_called_previously = false;
        // init zk and wk with default Talbot parameters
        m_params = {0.5017,0.6122,0.2645,dg::mat::finv_alpha(0.6407)};
        m_zkwk = dg::mat::weights_and_nodes_talbot( 2*2, m_params);
        m_paramsI = dg::mat::weights_and_nodes2params( m_zkwk);
    }

    bool m_called_previously = false;
    std::vector<double> m_params, m_paramsI;
    std::pair<std::vector<thrust::complex<double>>, std::vector<thrust::complex<double>>> m_zkwk; // zkwk == paramsI
    bool m_verbose = false;
};

///@endcond

} //namespace mat
} //namespace dg

