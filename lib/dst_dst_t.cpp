#include <iostream>
#include <iomanip>
#include <cmath>
#include "dst_dst.h"

using namespace std;
using namespace toefl;


int main()
{
    const size_t rows = 4, cols = 9;
    Matrix<double, TL_NONE> test( rows, cols);
    Matrix<double, TL_NONE> test_T( cols, rows);

    DST_DST dst_dst( rows, cols, FFTW_RODFT00);
    for( size_t i = 0; i < rows; i++)
        for( size_t j = 0; j < cols; j++)
            test(i,j) = sin( M_PI*(i+1)/(rows+1.))*sin( 2*M_PI*(j+1)/(cols+1.));
    cout << setprecision(2) << fixed;
    cout << "Testmatrix is\n" << test<<endl;

    dst_dst.r2r_T( test, test_T);
    cout << "transformed matrix is\n" << test_T<<endl;
    dst_dst.r_T2r( test_T, test);
    cout << "backtransformed is (should be input times 200)\n" << test << endl;


    return 0 ;
}
