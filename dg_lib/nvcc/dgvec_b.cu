#include <iostream>

#include "dgvec.cuh"
#include "blas/thrust_vector.cuh"
#include "timer.cuh"


using namespace dg;
using namespace std;

typedef double real;


const unsigned P = 3;
const unsigned N = 1e5;

typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> HVec;
int main()
{
    cout << "Array size is: "<<P<<"\n";
    cout << "Vector size is (P*N): "<< P*N<<"\n";
    ArrVec1d<P> hx( N, 3.), hy( N, 7.);
    DVec dx( hx.data()), dy( hy.data());
    Timer t;

    double dot;
    t.tic(); 
    dot = dg::BLAS1<DVec>::ddot( dx,dy); 
    t.toc();
    cout << "GPU dot took "<<t.diff() <<"s\n";
    cout << "Result "<<dot<<"\n";
    t.tic(); 
    dot = dg::BLAS1<HVec>::ddot( hx.data(),hy.data());
    t.toc();
    cout << "CPU dot took "<<t.diff() <<"s\n";
    cout << "Result "<<dot<<"\n";

    t.tic(); 
    dg::BLAS1<DVec>::daxpby( 3., dx, 7., dy); 
    t.toc();

    cout << "GPU daxpby took " << t.diff() <<"s\n";
    cout << "Result : \n" << dy[ dy.size() -1] << "\n";

    t.tic(); 
    dg::BLAS1<HVec>::daxpby( 3., hx.data(), 7., hy.data()); 
    t.toc();

    cout << "CPU daxpby took " << t.diff() <<"s\n";
    cout << "Result : \n" << hy(N-1, 2) << "\n";

    return 0;
}



