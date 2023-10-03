// #define DG_DEBUG

#include <iostream>
#include <iomanip>

#include "dg/algorithm.h"
#include "dg/file/file.h"

#include "lanczos.h"
#include "mcg.h"
#include "matrixfunction.h"

#include "gl_quadrature.h"

#include <cusp/transpose.h>
#include <cusp/array1d.h>
#include <cusp/array2d.h>
#include <cusp/print.h>

#include <cusp/lapack/lapack.h>

const double lx = 2.*M_PI;
const double ly = 2.*M_PI;
dg::bc bcx = dg::DIR;
dg::bc bcy = dg::PER;
// const double m=3./2.;
// const double n=4.;
// const double m=1./2.;
// const double n=1.;
const double m=3./2.;
const double n=2.;
const double ms=1./2.;
const double ns=2.;
const double alpha = 1./2.;
const double ell_fac = (m*m+n*n);
const double ell_facs = (ms*ms+ns*ns);

const double amp=10.0;
const double bgamp=1.0;

double lhs( double x, double y){ return sin(x*m)*sin(y*n);}
double lhss( double x, double y){ return sin(x*ms)*sin(y*ns);}
double sin2( double x, double y){ return amp*sin(x*m)*sin(y*n)*sin(x*m)*sin(y*n);}
using Matrix = dg::DMatrix;
using Container = dg::DVec;

// compute y = delta*A*gamma*x
template<class ContainerType>
class Wrapper
{
    template<class MatrixType>
    Wrapper( MatrixType& A, const ContainerType& gamma, const ContainerType& delta):
        m_gamma(gamma), m_delta( delta), m_temp(gamma){
        m_A = [&]( const ContainerType& x, ContainerType& y){
            return dg::apply( A, x, y);
        };
    }
    template<class ContainerType0, class ContainerType1>
    void operator()( const ContainerType0 x, ContainerType1& y)
    {
        dg::blas1::pointwiseDot( m_gamma, x, m_temp);
        dg::apply( m_A, m_temp, y);
        dg::blas1::pointwiseDot( m_delta, y, y);
    }

    private:
    std::function< void( const ContainerType&, ContainerType&)> m_A;
    ContainerType m_gamma, m_delta, m_temp;

};

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
    const Container w2d = dg::create::weights( g);//=M
    
    dg::Elliptic<dg::CartesianGrid2d, Matrix, Container> A( {g, dg::centered, 1.0});

    std::vector< std::function<double (double)>> funcs{
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
//         [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},
//         [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);}        
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);},   
        [](double x) { return dg::mat::GyrolagK<double>(0.,-alpha)(x);}  
    };
    std::vector<std::string> outs = {
            "K_0(-alpha A)",
            "K_0(d, -alpha A)",
            "K_0(-alpha A, d)",
//             "K_0_naive(d, -alpha A)",
//             "K_0_naive(-alpha A, d)",
            "K_0(-alpha d A)",
            "K_0(-alpha A d )"
    };
    
    //Plot into netcdf file
    size_t start = 0;
    dg::file::NC_Error_Handle err;
    int ncid;
    err = nc_create( "visual11.nc", NC_NETCDF4|NC_CLOBBER, &ncid);
    int dim_ids[5], tvarID;
    err = dg::file::define_dimensions( ncid, dim_ids, &tvarID, g);

//     std::string names[5] = {"K0","K0_prod","K0_prodadj","K0_prod_naive","K0_prodadj_naive"};
    std::string names[5] = {"K0","K0_prod","K0_prodadj","K0_app","K0_appadj"};
    int dataIDs[5];
    for( unsigned i=0; i<5; i++){
    err = nc_def_var( ncid, names[i].data(), NC_DOUBLE, 3, dim_ids, &dataIDs[i]);}

    dg::HVec transferH(dg::evaluate(dg::zero, g));
        
    for( unsigned u=0; u<funcs.size(); u++)
    {
        std::cout << "\n#Compute x = "<<outs[u]<<" b " << std::endl;

        Container x = dg::evaluate(lhs, g), x_exac(x), x_h(x), b(x), error(x);
        Container b_h(b);
        Container one = dg::evaluate(dg::ONE(), g);
        
        //note that d must fulfill boundary conditions and and must be >0
        //initialize constant d
//         Container d = dg::evaluate(dg::ZERO(), g);   
        
        //initialize d = amp*(sin( x/2) sin(y))^2
//         Container d = dg::evaluate(sin2, g);
        
        //initialize d = heaviside bump function
        Container d = dg::evaluate(dg::Cauchy(lx/2., ly/2., 3., 3., amp), g);
//         b_h = dg::evaluate(dg::SinXSinY(amp, 0.0, 4.0, 4.0), g); //superimpose sinxsiny
//         dg::blas1::pointwiseDot(b_h,b_h,b_h);      
//         dg::blas1::pointwiseDot(d,b_h,d);
        
        //add constant background field to d
        dg::blas1::plus(d, bgamp);
        
        Container w2d_AD = dg::create::weights( g); 
        Container w2d_DA = dg::create::weights( g); 
        dg::blas1::pointwiseDot( w2d, d, w2d_AD); //scale norm for A D self adjoint in the scaled norm M D , requires d\neq 0
        dg::blas1::pointwiseDivide( w2d, d, w2d_DA); //scale norm for D A self adjoint in the scaled norm M D^{-1}, requires d\neq 0
    
        
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
            
            //Compute c[l], v[l] and utlize them for x
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
//             //algorithm 1 (not converging)
//             t.tic();
//              //Tridiagonalize A first to T with the stopping condition for the function exp(-max(d)*alpha A)
//             auto Tf = krylovfunceigen.tridiag(func, A,  b, w2d, eps, 1,  "universal");
//             iter = krylovfunceigen.get_iter();            
//             //make eigendecomposition of f(d T) e_1 = E_T f(d eval_T) E_T^T e_1
//             cusp::array2d< double, cusp::host_memory> evecs(iter,iter);
//             cusp::array1d< double, cusp::host_memory> evals(iter);
//             cusp::lapack::stev(Tf.values.column(1), Tf.values.column(2), evals, evecs, 'V');            
//             //Compute c[l], v[l] and utlize them for x
//             std::vector<Container> v{iter,d}, c{iter,d};
//             dg::HVec e_k(iter, 0.); //unit vector e_k
//             Container fd(d); // helper variable
//             dg::blas1::scal(x, 0.0);            
//             //precompute v_k
//             for( unsigned k=0; k<iter; k++)
//             {
//                 dg::blas1::copy( 0, v[k]); // init sum
//                 dg::blas1::copy( 0, c[k]); // init sum
//                 //e_l
//                 dg::blas1::scal(e_k, 0.0);
//                 e_k[k] = 1.;
//                 //compute v[l]
// //                 krylovfunceigen.normMbVy(A, Tf, e_k, v[k], b, 1.0); //v_k=  V e_k //set bnorm = 1.0; the latter is not the same than the two lines below, why ? 
//                 krylovfunceigen.normMbVy(A, Tf, e_k, v[k], b, krylovfunceigen.get_bnorm()); //v_k= ||b||_M V e_k 
//                 dg::blas1::scal( v[k], 1./krylovfunceigen.get_bnorm()); //v_k = V e_k
//             }
//             //Compute v[k]
//             for( unsigned l=0; l<iter; l++)
//             {
//                 for( unsigned i=0; i<iter; i++)
//                 {
//                     dg::blas1::axpby( evals[i], d, 0., fd); //fd = lambda_i d
//                     dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha)); //fd =  f(lambda_i d)
//                     for( unsigned k=0; k<iter; k++)
//                     {
//                         dg::blas1::pointwiseDot(evecs(l,i)*evecs(i,k), fd, v[k], 1.0, c[l]); //c_l += (eps_{l,i} eps_{k,i}) f(lambda_i d) * v_k
//                     }
//                 }
//                 dg::blas1::axpby(dg::blas2::dot(c[l], w2d, b),  v[l], 1., x); //x += (c_l.M b) v_l 
//                 
//             }
//             //Compute errors in b and d approximation
//             dg::blas1::scal(b_h,0.);
//             dg::blas1::scal(fd,0.);
//             for( unsigned l=0; l<iter; l++)
//             {
//                 dg::blas1::axpby(fabs(dg::blas2::dot(v[l], w2d, d)),  v[l], 1., b_h); 
//                 dg::blas1::axpby(dg::blas2::dot(v[l], w2d, b),  v[l], 1., fd); 
//             }
//             
//             dg::blas1::axpby(1.0, b_h, -1.0, d, error);
//             std::cout << "    error_abs d = " << sqrt(dg::blas2::dot( w2d, error)) << std::endl;
//             std::cout << "    error d = " << sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, d)) << std::endl;
//             dg::blas1::axpby(1.0, fd, -1.0, b, error);
//             std::cout << "    error b = " << sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, b)) << std::endl;
//             t.toc();
//             time = t.diff();
//             
            //algorithm 2 
            t.tic();
            //Tridiagonalize diagonal matrix D            
            auto Rf = krylovfunceigend.tridiag(func, d,  b, w2d,  1e-12, 1.,  "universal");
            unsigned iter_Rf = krylovfunceigend.get_iter();
            std::cout << "    universal-iter-Rf: "<<std::setw(3)<< iter_Rf << "\n";
            
            //make eigendecomposition of Rf = E_Rf  eval_Rf E_Rf^T 
            cusp::array2d< double, cusp::host_memory> evecs_Rf(iter_Rf,iter_Rf);
            cusp::array1d< double, cusp::host_memory> evals_Rf(iter_Rf);
            cusp::lapack::stev(Rf.values.column(1), Rf.values.column(2), evals_Rf, evecs_Rf, 'V');
            cusp::coo_matrix<int, double, cusp::host_memory> E_Rf, E_Rf_t;

            cusp::convert(evecs_Rf, E_Rf);
            cusp::transpose(E_Rf, E_Rf_t);           

            //Compute h_k
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
                A.set_chi(evals_Rf[k]);
                iter= krylovfunceigen.solve(x_h, func, A, fd, w2d, eps, 1., "universal"); // x_h = ||v_k||_M V_Tf f(Tf lambda_Rf,k) v_k
                dg::blas1::axpby(1.0, x_h, 1.0, x);
                std::cout << "    universal-iter-Tf: "<<std::setw(3)<< krylovfunceigen.get_iter() << "\n";

            }
            t.toc();
            time = t.diff();
            A.set_chi(one);
        }
//         if (u==3) 
//         {
//             t.tic();
//             double lambda_d = 0.;
//             for( unsigned k=0; k<x.size(); k++)
//             {
//                 lambda_d = d[k];
//                 A.set_chi(lambda_d);
//                 iter = krylovfunceigen.solve(x_h, func, A, b, w2d, eps, 1., "universal");
//                 iter_sum+=iter;
//                 x[k] = x_h[k];
//             }
//             t.toc();
//             time = t.diff();
//             A.set_chi(one);
//         }
//         if (u==4) 
//         {
//             dg::blas1::scal(x, 0.0);
//             double lambda_d = 0.;
//             iter_sum=0;
//             Container fd(b); // helper variable
//             t.tic();
//             for( unsigned k=0; k<x.size(); k++)
//             {
//                 lambda_d = d[k];
//                 A.set_chi(lambda_d);
//                 dg::blas1::scal(b_h, 0.0);
//                 b_h[k] = b[k]; //instead of b
//                 iter = krylovfunceigen.solve(x_h, func, A, b_h, w2d, eps, 1.0, "universal");
//                 iter_sum+=iter;
//                 dg::blas1::axpby(1.0, x_h, 1.0, x);
//             }
//             t.toc();m_precond
//             time = t.diff();
//            A.set_chi(one);
//         }
        if (u==3)
        {
            Wrapper wrap( A, one, d);
            t.tic();
            iter= krylovfunceigen.solve(x, func, wrap, b, w2d_DA, eps, 1., "universal"); 
            t.toc();
            time = t.diff();
        }
        if (u==4)
        {
            Wrapper wrap( A, d, one);
            t.tic();            
            iter= krylovfunceigen.solve(x, func, wrap, b, w2d_AD, eps, 1., "universal"); 
            //weights of adjoint missing?
            t.toc();
            time = t.diff();
        }
        //write solution into file
        dg::assign( x, transferH);
        dg::file::put_vara_double( ncid, dataIDs[u], start, g, transferH);      
        //Compute errors
        if (u==0)
        {
            dg::blas1::scal(x_exac, funcs[u](ell_fac));
        }
        else 
        {
            Container fd(d); // helper variable
            //Compute absolute and relative error in adjointness 
            if (u==2 || u==4)
            {
                x_h = dg::evaluate(lhss, g); // -> g
                dg::blas1::axpby(ell_facs, d, 0.0, fd);
                dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
                dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(ms^2+ns^2) d) sin(x*ms) cos(y*ms) \equiv exp(d,-alpha A) g
                x_h = dg::evaluate(lhs, g); // -> f
                double fOg = dg::blas2::dot( x_h, w2d, x_exac); //<f,exp(d,-alpha A) g>
                std::cout << "    <f, exp(d,-alpha A) g> = " << fOg << std::endl;
                x_h = dg::evaluate(lhss, g); // -> g
                double gOadjf = dg::blas2::dot( x, w2d, x_h); //<exp(-alpha A, d)f, g>
                std::cout << "    <exp(-alpha A, d)f, g> = " << gOadjf << std::endl;

                double eabs_adj = fOg-gOadjf; // <f,exp(d,-alpha A) g> -<exp(-alpha A, d)f, g>
                std::cout << "    universal-abserror-adjointness: "<< eabs_adj  << "\n"; 
                std::cout << "    universal-relerror-adjointness: "<< eabs_adj/fOg  << "\n";
            }
            //Compute exact error for product exponential (is used also for adjoint product exponential since we have no analytical solution there)
            x_h = dg::evaluate(lhs, g);
            dg::blas1::axpby(ell_fac, d, 0.0, fd);
            dg::blas1::transform(fd, fd, dg::mat::GyrolagK<double>(0.,-alpha));
            dg::blas1::pointwiseDot(fd, x_h, x_exac); //x_exac = f(-alpha*(m^2+n^2) d) sin(m x) cos(n y)
        }        
        dg::blas1::axpby(1.0, x, -1.0, x_exac, error);
        erel = sqrt(dg::blas2::dot( w2d, error) / dg::blas2::dot( w2d, x_exac));
        std::cout << "    universal-iter: "<<std::setw(3)<< iter << "\n";
        if (u==3 || u==4) std::cout << "    universal-iter_sum: "<<std::setw(3)<<iter_sum << "\n";
        std::cout << "    universal-time: "<<time<<"s \n";
        std::cout << "    universal-error: "<<erel  << "\n";
    }
    err = nc_close(ncid);

    return 0;
}
