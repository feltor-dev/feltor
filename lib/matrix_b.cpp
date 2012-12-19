#include <iostream>

#include "timer.h"
#include "matrix.h"
#include <complex>

using namespace toefl;
using namespace std;


size_t rows = 500, cols = 4*500;

int main()
{
    Matrix<double > a( rows, cols);
    Matrix<double > b( rows, cols);
    Timer t;

    t.tic();
    for( size_t i =0; i<rows; i++)
        for( size_t j=0; j<cols; j++)
            a(i,j) = 17 + i +j;
    t.toc();
    cout << "Assignment took " << t.diff() << " seconds\n";
    t.tic();
    b = a;
    t.toc();
    cout << "Member function took " << t.diff()<<" seconds\n";
    return 0;
}






