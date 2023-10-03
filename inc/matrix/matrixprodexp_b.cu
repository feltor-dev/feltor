// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"

#include "lanczos.h"
#include "mcg.h"
#include "matrixfunction.h"

#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>

#include <cusp/lapack/lapack.h>

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
//const double m=3./2.;
//const double n=4.;
const double m=1./2.;
const double n=1.;
const double ms=1./2.;
const double ns=2.;
const double alpha = 1./2.;
const double ell_fac = (m*m+n*n);
const double ell_facs = (ms*ms+ns*ns);

const double amp=20.0;
const double bgamp=1.0;

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}
double lhss( double x, double y){ return sin(x*ms)*sin(y*ns);}
using Matrix = dg::DMatrix;
using Container = dg::DVec;

int main(int argc, char * argv[])
{
    dg::Timer t;

    unsigned n, Nx, Ny;
    std::cout << "# Type n, Nx and Ny! \n";
    std::cin >> n >> Nx >> Ny;
    std::cout <<"# You typed\n"
              <<"n:  "<<n<<"\n"
              <<"Nx: "<<Nx<<"\n"
              <<"Ny: "<<Ny<<std::endl;
    unsigned iter = 0;

    unsigned max_iter = 1;
    std::cout << "# Type max_iter of tridiagonalization (500) ?\n";
    std::cin >> max_iter ;
    std::cout << "# Type in eps of tridiagonalization (1e-7)\n";
    double eps = 1e-7; //# of pcg iter increases very much if
    std::cin >> eps;
    std::cout <<"# You typed\n"
              <<"max_iter: "<<max_iter<<"\n"
              <<"eps: "<<eps<<std::endl;

    double erel = 0;

    dg::Grid2d g( 0, lx, 0, ly,n, Nx, Ny, bcx, bcy);
    const Container w2d = dg::create::weights( g);
    Container w2d_scaled(w2d);

    double max_weights = dg::blas1::reduce(w2d, 0., dg::AbsMax<double>() );
    double min_weights = dg::blas1::reduce(w2d, max_weights, dg::AbsMin<double>() );
    std::cout << "#   min(W)  = "<<min_weights <<"  max(W) = "<<max_weights << "\n";
    dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( {g, dg::centered, 1.0});

    std::vector< std::function<double (double)>> funcs{
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},      
    };
    std::vector<std::string> outs = {
            "K_0(-alpha A)",
            "K_0(d, -alpha A)",
            "K_0(-alpha A, d)",
    };
    
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        Container b_h(b);
        Container one = dg::evaluate(dg::ONE(), g);
        
        //note that d must fulfill boundary conditions and should be positive definite!
        Container d = dg::evaluate(dg::Cauchy(lx/2., ly/2., 3., 3., amp), g);
        //add constant background field to d
        dg::blas1::plus(d, bgamp);
        
        std::cout << outs[u] << ":\n";

        dg::mat::UniversalLanczos<Container> krylovfunceigen( x, max_iter);
        dg::mat::UniversalLanczos<Container> krylovfunceigend( x, max_iter);

        auto func = dg::mat::make_FuncEigen_Te1( funcs[u]);
        double time = t.diff();
        unsigned iter_sum=0;
        
        //MLanczos-universal
        if (u==0)
        {
            t.tic();
            iter= krylovfunceigen.solve(x, func, A, b, w2d, eps, 1., "universal");
            t.toc();
            time = t.diff();
        }
        if (u==1)
        {
            t.tic();
            //Tridiagonalize A first to T with the stopping condition for the function exp(-max(d)*alpha A)
            auto Tf = krylovfunceigen.tridiag(func, A,  b, w2d, eps, 1.,  "universal");
            iter = krylovfunceigen.get_iter();
            
            //make eigendecomposition of f(d T) e_1 = E_T f(d eval_T) E_T^T e_1
            cusp::array2d< double, cusp::host_memory> evecs(iter,iter);
            cusp::array1d< double, cusp::host_memory> evals(iter);
            cusp::lapack::stev(Tf.values.column(1), Tf.values.column(2), evals, evecs, 'V');
            
            //Compute c[l], v[l] and utilize them for x
            std::vector<Container> c{iter,d}, v{iter,d};
            dg::HVec e_l(iter, 0.); //unit vector e_l
            Container fd(d); // helper variable
            dg::blas1::scal(x,0.0);
            for( unsigned l=0; l<iter; l++)
            {
                dg::blas1::copy( 0, c[l]); // init sum
                dg::blas1::copy( 0, v[l]); // init sum
                //e_l
                dg::blas1::scal(e_l, 0.0);
                e_l[l] = 1.;
                //Compute c[l]
                for( unsigned j=0; j<iter; j++)
                {
                    dg::blas1::axpby( evals[j], d, 0., fd);
                    dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
                    dg::blas1::axpby( evecs(0,j)*evecs(l,j), fd, 1., c[l]);
                }
                //compute v[l]
                krylovfunceigen.normMbVy(A, Tf, e_l, v[l], b, krylovfunceigen.get_bnorm()); //v[l]=  ||b|| V e_l
                //compute x+=v[l] p. c[l]
                dg::blas1::pointwiseDot(1.0, v[l], c[l], 1.0, x);
            }
            t.toc();
            time = t.diff();
        }
        if (u==2)
        {
            t.tic();
            //Tridiagonalize diagonal matrix d
            auto Rf = krylovfunceigend.tridiag(func, d,  b, w2d, 1e-12, 1.,  "universal");
            unsigned iter_Rf = krylovfunceigend.get_iter();
            iter_sum+=iter_Rf;
            std::cout << "#    universal-iter-Rf: "<<std::setw(3)<< iter_Rf << "\n";
            
            //make eigendecomposition of Rf = E_Rf  eval_Rf E_Rf^T 
            cusp::array2d< double, cusp::host_memory> evecs_Rf(iter_Rf,iter_Rf);
            cusp::array1d< double, cusp::host_memory> evals_Rf(iter_Rf);
            cusp::lapack::stev(Rf.values.column(1), Rf.values.column(2), evals_Rf, evecs_Rf, 'V');
            cusp::coo_matrix<int, double, cusp::host_memory> E_Rf, E_Rf_t;
            cusp::convert(evecs_Rf, E_Rf);
            cusp::transpose(E_Rf, E_Rf_t);       
            
            dg::HVec e_1(iter_Rf,0.), e_k(e_1), y(e_1); //unit vector e_1
            Container fd(d); 
            e_1[0] = 1.;
            
            dg::blas2::symv(E_Rf_t, e_1, y); //y = E_Rf^T e_1
            dg::blas1::scal(x, 0.0);
            for( unsigned k=0; k<iter_Rf; k++)
            {
                dg::blas1::scal(e_k, 0.0);
                e_k[k] = 1.;
                dg::blas1::pointwiseDot(e_k, y, e_1); //y = e_k * (E_Rf^T e_1) = 1_k E_Rf^T e_1 
                dg::blas2::symv(E_Rf, e_1, e_k);        //h_k =E_Rf (e_k * (E_Rf^T e_1)) =E_Rf 1_k E_Rf^T e_1      
                krylovfunceigend.normMbVy(d, Rf, e_k, fd, b, krylovfunceigend.get_bnorm()); //v_k=  ||b||_M V_Rf h_k
                
                 
                //Solve 
                //is it important that v_k = fd respects the boundary conditions?, takes now much longer because the test function is no longer an eigenfunction
                A.set_chi(evals_Rf[k]);
                iter= krylovfunceigen.solve(x_h, func, A, fd,  w2d, eps, 1., "universal"); // x_h = ||v_k||_M V_Tf f(Tf lambda_Rf,k) v_k
                iter_sum+=iter;
                dg::blas1::axpby(1.0, x_h, 1.0, x);
                std::cout << "#    universal-iter-Tf_"<< k <<": "<<std::setw(3)<< krylovfunceigen.get_iter() << "      eval_Rf: " << evals_Rf[k] << "\n";
            }
            t.toc();
            time = t.diff();
        }
        //Compute errors
        if (u==0)
        {
            dg::blas1::scal(x_exac, funcs[u](ell_fac));
        }
        else 
        {
            Container fd(d); // helper variable
            //Compute absolute and relative error in adjointness 
            if (u==2 )
            {
                x_h = dg::evaluate(lhss, g); // -> g
                dg::blas1::axpby(ell_facs, d, 0.0, fd);
                dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
                dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(ms^2+ns^2) d) sin(x*ms) cos(y*ms) \equiv exp(d,-alpha A) g
                x_h = dg::evaluate(lhs, g); // -> f
                double fOg = dg::blas2::dot( x_h, w2d, x_exac); //<f,exp(d,-alpha A) g>
                std::cout << "#    <f, exp(d,-alpha A) g> = " << fOg << std::endl;
                x_h = dg::evaluate(lhss, g); // -> g
                double gOadjf = dg::blas2::dot( x, w2d, x_h); //<exp(-alpha A, d)f, g>
                std::cout << "#    <exp(-alpha A, d)f, g> = " << gOadjf << std::endl;

                double eabs_adj = fOg-gOadjf; // <f,exp(d,-alpha A) g> -<exp(-alpha A, d)f, g>
                std::cout << "#    Errors in adjointness"<< "\n";
                std::cout << "#    universal-abserror: "<< eabs_adj  << "\n"; 
                std::cout << "    universal-error: "<< eabs_adj/fOg  << "\n";
            }
            //Compute exact error for product exponential (is used also for adjoint product exponential since we have no analytical solution there)
            x_h = dg::evaluate(lhs, g);
            dg::blas1::axpby(ell_fac, d, 0.0, fd);
            dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
            dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(m^2+n^2) d) sin(m x) cos(n y)
        }        
        std::cout << "    universal-time: "<<time<<"s \n";
        if (u==0 || u==1) {
            dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
            erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
            std::cout << "    universal-error: "<<erel  << "\n";
            std::cout << "    universal-iter: " <<std::setw(3)<< iter << "\n";            
        }
        else std::cout << "    universal-iter: " <<std::setw(3)<< iter_sum << "\n";
    }

    return 0;
}
