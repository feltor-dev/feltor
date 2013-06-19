#include <iostream> 
#include <cusp/blas.h>

#include "blas.h"
#include "arrvec1d.cuh" 
//#include "../gcc/timer.h" 
#include "timer.cuh"

using namespace dg; 
using namespace std; 

typedef double real;


const unsigned P = 3;
const unsigned N = 1e7; //GPU starts to pay off > 1e4 points

typedef thrust::device_vector<double> DVec;
typedef thrust::host_vector<double> HVec;
typedef ArrVec1d< double, P, HVec> HArrVec;
typedef ArrVec1d< double, P, DVec> DArrVec;
int main()
{
    cout << "# of Legendre coefficients "<<P<<"\n";
    cout << "# of grid cells            "<< N<<"\n";
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
    dot = dg::blas1::dot( dx_v.data(), dy); 
    t.toc();
    cout << "GPU dot(x,y) took              "<<t.diff() <<"s\n";
    //cout << "Result "<<dot<<"\n";
    t.tic(); 
    dot = dg::blas1::dot( hx.data(),hy.data());
    t.toc();
    cout << "CPU dot(x,y) took              "<<t.diff() <<"s\n";
    cout << "Result "<<dot<<"\n";

    t.tic(); 
    dg::blas1::axpby( 3., dx, 7., dy_v.data()); 
    t.toc();

    cout << "GPU daxpby took                " << t.diff() <<"s\n";
    //cout << "Result : " << dy[ dy.size() -1] << "\n";

    t.tic(); 
    dg::blas1::axpby( 3., hx.data(), 7., hy.data()); 
    t.toc();

    cout << "CPU daxpby took                " << t.diff() <<"s\n";
    //cout << "Result : " << hy(N-1, 2) << "\n";
    t.tic(); 
    dg::blas1::pointwiseDot( dx, dy, dy); 
    t.toc();
    cout << "GPU pointwiseDot took          " << t.diff() <<"s\n";

    t.tic(); 
    dg::blas1::pointwiseDot( hx.data(), hy.data(), hy.data()); 
    t.toc();
    cout << "CPU pointwiseDot took          " << t.diff() <<"s\n";

    return 0;
}



