
#include <iostream>
#include "functors.h"

#include "contours.h"


struct Sanify
{
    double operator()DG_DEVICE(double x){
        if( m_isnsane(x))
            return 1e100;
        else
            return x;
    }
    private:
    dg::ISNSANE<double> m_isnsane;
};
int main()
{
    unsigned n = 5;
    std::vector<std::vector<double>> ps(4);
    ps[0] = dg::mat::generate_range( 0.1, 10, 5);
    ps[1] = dg::mat::generate_range( 0.1, 10, 5);
    ps[2] = dg::mat::generate_range( 0.1, 10, 5);
    ps[3] = dg::mat::generate_range( 0.1, 10, 5);
    std::vector<dg::HVec> ps_cpu( 4, dg::HVec( ps[0].size()*ps[1].size()*ps[2].size()*ps[3].size()));
    std::vector<dg::DVec> ps_gpu( 4);
    dg::Timer t;
    t.tic();
    for( unsigned i=0; i<ps[0].size(); i++)
    for( unsigned j=0; j<ps[1].size(); j++)
    for( unsigned k=0; k<ps[2].size(); k++)
    for( unsigned l=0; l<ps[3].size(); l++)
    {
        ps_cpu[0][((i*ps[1].size() + j)*ps[2].size()+k)*ps[3].size()+l] = ps[0][i];
        ps_cpu[1][((i*ps[1].size() + j)*ps[2].size()+k)*ps[3].size()+l] = ps[1][j];
        ps_cpu[2][((i*ps[1].size() + j)*ps[2].size()+k)*ps[3].size()+l] = ps[2][k];
        ps_cpu[3][((i*ps[1].size() + j)*ps[2].size()+k)*ps[3].size()+l] = ps[3][l];
    }
    t.toc();
    std::cout << "Generating params took "<<t.diff()<<"\n";
    auto func = dg::mat::GyrolagK<thrust::complex<double>>(0,1);
    auto dxfunc = dg::mat::DGyrolagK<thrust::complex<double>>(0,1);

    // TCV
    auto rrs = dg::mat::generate_range( 0.0625, 12.5, 20);
    auto lls = dg::mat::generate_range( 21, 23401947, 20);

    // A difficult example
    //auto rrs = dg::mat::generate_range( 0.1, 11, 20);
    //auto lls = dg::mat::generate_range( 1, 11800, 20);

    // matrixprodexp
    //auto rrs = dg::mat::generate_range( 1, 11, 20);
    //auto lls = dg::mat::generate_range( 0.25, 118, 20);
    dg::HVec errors_cpu(ps_cpu[0].size());


    dg::mat::LeastSquaresCauchyError cauchyTCV( 2*n, dg::mat::weights_and_nodes_talbot,
        func, rrs, lls);
    dg::mat::LeastSquaresCauchyJacobian jacTCV( 2*n, dg::mat::weights_and_nodes_talbot,
        dg::mat::jacobian_talbot,
        func, dxfunc, rrs, lls);
    std::vector<double> results( lls.size()*rrs.size());
    dg::mat::LeastSquaresCauchyError IcauchyTCV( 2*n, dg::mat::weights_and_nodes_identity,
        func, rrs, lls);
    dg::mat::LeastSquaresCauchyJacobian IjacTCV( 2*n, dg::mat::weights_and_nodes_identity,
        dg::mat::jacobian_identity,
        func, dxfunc, rrs, lls);
    // One can play between 1 and 2 here
    cauchyTCV.set_order(1);
    jacTCV.set_order(1);
    IcauchyTCV.set_order(1);
    IjacTCV.set_order(1);

    dg::assign( ps_cpu, ps_gpu);
    dg::DVec errors(ps_gpu[0].size());
    t.tic();
    error_talbot( 2*n, rrs, lls, ps_gpu, func, errors);
    t.toc();
    std::cout << "Computing all erros took "<<t.diff()<<"\n";
    dg::blas1::transform( errors,errors, Sanify());
    errors_cpu = errors;
    //double min = dg::blas1::reduce( errors, 1e300, thrust::minimum<double>(), Sanify());
    for( unsigned i=0; i <1; i++) // it always finds the same point
    {
        unsigned min_idx = std::min_element( errors_cpu.begin(), errors_cpu.end())-errors_cpu.begin();

        std::cout << "Minimum is "<<errors_cpu[min_idx]<<" with params "<<ps_cpu[0][min_idx]<<" "<<ps_cpu[1][min_idx]<<" "<<ps_cpu[2][min_idx]<<" "<<ps_cpu[3][min_idx]<<"\n";

        errors_cpu[min_idx] = 1e300;
        std::vector<double> params = {ps_cpu[0][min_idx], ps_cpu[1][min_idx], ps_cpu[2][min_idx], ps_cpu[3][min_idx]};
        unsigned steps = levenberg_marquardt( cauchyTCV, jacTCV, params, results, 1e-8, 1000);
        std::cout << "Found optimum in "<<steps<<" steps\n";
        cauchyTCV.set_order(1);
        cauchyTCV.error( params, results);
        std::cout << "Cauchy error "<<dg::blas1::dot( results, results)<<" ";
        std::cout << " with params "<<params[0]<<" "<<params[1]<<" "<<params[2]<<" "<<params[3]<<"\n";
        std::cout << "Abs max error "<<dg::blas1::reduce( results, -1e300, thrust::maximum<double>(), dg::ABS<double>())<<"\n";
        auto zkwk = dg::mat::weights_and_nodes_talbot( 2*n, params);
        auto paramsI = dg::mat::weights_and_nodes2params( zkwk);
        steps = levenberg_marquardt( IcauchyTCV, IjacTCV, paramsI, results, 1e-8, 1000);
        std::cout << "Found I optimum in "<<steps<<" steps\n";
        IcauchyTCV.set_order(1);
        IcauchyTCV.error( paramsI, results);
        std::cout << "Cauchy I error "<<dg::blas1::dot( results, results)<<"\n";
        std::cout << "Abs max I error "<<dg::blas1::reduce( results, -1e300, thrust::maximum<double>(), dg::ABS<double>())<<"\n";
    }

    return 0;
}
