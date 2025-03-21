#include <iostream>
#include <iomanip>

#include "tridiaginv.h"

using value_type = double;
using CooMatrix =  dg::SquareMatrix<double>;
using DiaMatrix =  dg::TriDiagonal<thrust::host_vector<double>>;
using Container = dg::HVec;

double mu(double s, unsigned i, unsigned n) {
    return (1.0+1.0/s*(1.0-1.0/pow(1.0 + s,n-i-1.0)));
}

int main()
{
    dg::Timer t;
    unsigned size = 50;
    std::cout << "#Specify size of vectors (50)\n";
    std::cin >> size;
    unsigned max_outer =300;
    unsigned max_inner = 300;
    unsigned restarts = 30000;
//     std::cout << "# max_outer, max_inner and restarts of lgmres (30,10,10000) \n";
//     std::cin >> max_outer >> max_inner >> restarts;

    std::cout << "#Constructing and filling vectors\n";
    std::vector<value_type> a(size,1.);
    std::vector<value_type> b(size,1.);
    std::vector<value_type> c(size,1.);
    std::vector<value_type> a_sym(size,1.);
    std::vector<value_type> b_sym(size,1.);
    std::vector<value_type> c_sym(size,1.);
    double s= 1.1;
    for (unsigned i=0;i<size; i++)
    {
        //vectors of non-symmetric tridiagonal matrix
        a[i] = 1.0;
        b[i] = -1.0/(2.0+s);
        c[i] = -(1.0+s)/(2.0+s);
        //vectors of symmetric tridiagonal matrix
        if (i<size-1) {
            a_sym[i] = 4.0*(i+1)*(i+1)*(i+1)/(4.0*(i+1)*(i+1)-1.0);
        }
        else {
            a_sym[i] = size*size/(2.0*size-1.0);
        }
        b_sym[i] = -1.0*(i+1)*((i+1)+1.0)/(2.0*(1+i)+1.0);
        c_sym[i] = i==0 ? 0 : b_sym[i-1];
    }
    std::cout << "#Constructing and filling containers\n";
    const Container d(size,1.);
    Container x(size,0.), x_symsol(x), x_sol(x), err(x);
    std::cout << "#Constructing Matrix inversion and linear solvers\n";
    value_type eps= 1e-20;
    t.tic();
    dg::PCG pcg( x,  size*size+1);
    t.toc();
    std::cout << "#Construction of CG took "<< t.diff()<<"s \n";
    t.tic();
    dg::LGMRES lgmres( x, max_outer, max_inner, restarts);
    t.toc();
    std::cout << "#Construction of LGMRES took "<< t.diff()<<"s \n";
    t.tic();
    dg::BICGSTABl bicg( x,size*size,4);
    t.toc();
    std::cout << "#Construction of BICGSTABl took "<< t.diff()<<"s \n";
    t.tic();
    dg::mat::TridiagInvDF<double> tridiaginvDF(a);
    t.toc();
    std::cout << "#Construction of Tridiagonal inversion DF routine took "<< t.diff()<<"s \n";
    t.tic();
    dg::mat::TridiagInvD<double> tridiaginvD(a);
    t.toc();
    std::cout << "#Construction of Tridiagonal inversion D routine took "<< t.diff()<<"s \n";

    //Create Tridiagonal and fill matrix
    DiaMatrix T, Tsym;
    T.resize(size);
    Tsym.resize(size);

    for( unsigned i=0; i<size; i++)
    {
        T.O[i]   =  a[i];  // 0 diagonal
        T.M[i]   =  c[i];  // -1 diagonal
        T.P[i]   =  b[i];  // +1 diagonal
        Tsym.O[i]   =  a_sym[i];  // 0 diagonal
        Tsym.M[i]   =  c_sym[i];  // -1 diagonal
        Tsym.P[i]   =  b_sym[i];  // +1 diagonal
    }

    //Create Inverse of tridiagonal matrix
    CooMatrix Tinv, Tsyminv, Tinv_sol(size), Tsyminv_sol(size);
    for( unsigned i=0; i<size; i++) //row index
    {
        for( unsigned j=0; j<size; j++) //column index
        {
            if (i>= j)
            {
                Tinv_sol(i,j) = (2.0+s)/(1.0+s)*mu(s,i+1,size+1)*mu(s,size+1-(j+1),size+1)/mu(s,0,size+1);
                Tsyminv_sol(i,j) = (j+1.0)/(i+1.0);
            }
            else
            {
                Tsyminv_sol(i,j) = (i+1.0)/(j+1.0);
            }
        }
    }
    for( unsigned i=0; i<size; i++) //row index
    {
        for( unsigned j=0; j<size; j++) //column index
        {
            if (i<j)
            {
                Tinv_sol(i,j) = pow(1.0/(1.0+s),j-i)*Tinv_sol(j,i);
            }
        }
    }
    dg::blas2::gemv(Tinv_sol, d, x_sol);
    dg::blas2::gemv(Tsyminv_sol, d, x_symsol);

    //Do inversions
    std::cout << "####Compute inverse of symmetric tridiagonal matrix\n";
    std::cout << "CG:" << std::endl;
    dg::blas1::scal(x, 0.);
    t.tic();
    pcg.solve( Tsym, x, d, 1., 1., eps);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_symsol, err );
    std::cout << "    time: "<< t.diff()<<"s \n";
    std::cout << "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_symsol,x_symsol)) << "\n";
    std::cout << "InvtridiagDF(v_sym):" << std::endl;
    t.tic();
    tridiaginvDF(a_sym,b_sym,c_sym, Tsyminv);
    t.toc();
    dg::blas2::gemv(Tsyminv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_symsol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_symsol,x_symsol)) << "\n";
    std::cout << "InvtridiagDF(Tsym):" << std::endl;
    t.tic();
    tridiaginvDF(Tsym, Tsyminv);
    t.toc();
    dg::blas2::gemv(Tsyminv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_symsol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_symsol,x_symsol)) << "\n";
    std::cout <<  "    #error_rel in T_{m,1}: " << fabs(Tsyminv(size-1,0) - Tsyminv_sol(size-1,0))/fabs(Tsyminv_sol(size-1,0)) << "\n";
    if(  size < 150)
    {
    std::cout << "InvtridiagD(v_sym):" << std::endl;
    t.tic();
    tridiaginvD(a_sym,b_sym,c_sym, Tsyminv);
    t.toc();
    dg::blas2::gemv(Tsyminv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_symsol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_symsol,x_symsol)) << "\n";
    std::cout << "InvtridiagD(Tsym):" << std::endl;
    t.tic();
    tridiaginvD(Tsym, Tsyminv);
    t.toc();
    dg::blas2::gemv(Tsyminv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_symsol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_symsol,x_symsol)) << "\n";
    std::cout <<  "    #error_rel in T_{m,1}: " << fabs(Tsyminv(size-1,0) - Tsyminv_sol(size-1,0))/fabs(Tsyminv_sol(size-1,0)) << "\n";
    }
    std::cout << "# Test Thomas algorithm\n";
    t.tic();
    double T1m = dg::mat::compute_Tinv_m1(Tsym, size);
    t.toc();
    std::cout <<  "    # time: "<< t.diff()<<"s \n";
    dg::HVec e1( size, 0.), em ( e1), y(e1);
    e1[0] = 1.;
    em[size-1] = 1.;
    dg::blas2::symv( Tsyminv, e1, y);
    double T1mana = dg::blas1::dot( em, y);
    //std::cout << "    # result "<<T1mana<<" "<<T1m<<"\n";
    std::cout << "    # error_rel "<<fabs(T1mana - T1m)/T1m<<"\n";
    t.tic();
    dg::mat::compute_Tinv_y(Tsym, x, d);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_symsol, err );
    std::cout <<  "    # time: "<< t.diff()<<"s \n";
    //std::cout << "    # result "<<T1mana<<" "<<T1m<<"\n";
    std::cout <<  "    # error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_symsol,x_symsol)) << "\n";



    std::cout << "\n####Compute inverse of non-symmetric tridiagonal matrix\n";
    std::cout << "lGMRES:" << std::endl;
    dg::blas1::scal(x, 0.);
    t.tic();
    lgmres.solve( T, x, d , d, d, eps, 1);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    std::cout << "BICGSTABl:" << std::endl;

    dg::blas1::scal(x, 0.);
    t.tic();
    bicg.solve( T, x, d , d, d, eps, 1);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    std::cout << "InvtridiagDF(v):" << std::endl;
    t.tic();
    tridiaginvDF(a,b,c,Tinv);
    t.toc();
    dg::blas2::gemv(Tinv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    std::cout << "InvtridiagDF(T):" << std::endl;
    t.tic();
    tridiaginvDF(T,Tinv);
    t.toc();
    dg::blas2::gemv(Tinv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    if( size < 150)
    {
    std::cout << "InvtridiagD(v):" << std::endl;
    t.tic();
    tridiaginvD(a,b,c,Tinv);
    t.toc();
    dg::blas2::gemv(Tinv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    std::cout << "InvtridiagD(T):" << std::endl;
    t.tic();
    tridiaginvD(T,Tinv);
    t.toc();
    dg::blas2::gemv(Tinv, d, x);
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "    time: "<< t.diff()<<"s \n";
    std::cout <<  "    error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";
    }

    std::cout << "# Test Thomas algorithm\n";
    t.tic();
    T1m = dg::mat::compute_Tinv_m1(T, size);
    t.toc();
    std::cout <<  "    # time: "<< t.diff()<<"s \n";
    dg::blas2::symv( Tinv, e1, y);
    T1mana = dg::blas1::dot( em, y);
    std::cout << "    # error_rel "<<fabs(T1mana - T1m)/T1m<<"\n";
    t.tic();
    dg::mat::compute_Tinv_y(T, x, d);
    t.toc();
    dg::blas1::axpby(1.0, x, -1.0, x_sol, err );
    std::cout <<  "    # time: "<< t.diff()<<"s \n";
    //std::cout << "    # result "<<T1mana<<" "<<T1m<<"\n";
    std::cout <<  "    # error_rel: " << sqrt(dg::blas1::dot(err,err)/dg::blas1::dot(x_sol,x_sol)) << "\n";

    return 0;
}
