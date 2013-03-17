#include <iostream> 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

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
typedef ArrVec1d< double, P, HVec> HArrVec;
typedef ArrVec1d< double, P, DVec> DArrVec;
int main()
{
    cout << "Array size is:             "<<P<<"\n";
    cout << "Number of intervals is:    "<< N<<"\n\n";
    Timer t;
    HArrVec hx( N, 3.), hy( N, 7.);
    t.tic();
    DVec dx( hx.data()), dy( hy.data());
    t.toc();
    cout << "Copy of data host2device took: "<<t.diff()<<"s\n";
    typename DArrVec::View dx_v( dx);
    typename DArrVec::View dy_v( dy);

    double dot;
    t.tic(); 
    dot = dg::BLAS1<DVec>::ddot( dx_v.data(),dy); 
    t.toc();
    cout << "GPU dot(x,y) took          "<<t.diff() <<"s\n";
    cout << "Result "<<dot<<"\n";
    t.tic(); 
    dot = dg::BLAS1<HVec>::ddot( hx.data(),hy.data());
    t.toc();
    cout << "CPU dot(x,y) took          "<<t.diff() <<"s\n";
    cout << "Result "<<dot<<"\n\n";

    t.tic(); 
    dg::BLAS1<DVec>::daxpby( 3., dx, 7., dy_v.data()); 
    t.toc();

    cout << "GPU daxpby took            " << t.diff() <<"s\n";
    cout << "Result : " << dy[ dy.size() -1] << "\n";

    t.tic(); 
    dg::BLAS1<HVec>::daxpby( 3., hx.data(), 7., hy.data()); 
    t.toc();

    cout << "CPU daxpby took            " << t.diff() <<"s\n";
    cout << "Result : " << hy(N-1, 2) << "\n";

    return 0;
}



