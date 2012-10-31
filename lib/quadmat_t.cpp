
#include "quadmat.h"
#include <complex>
#include <iostream>

using namespace toefl;
using namespace std;

int main()
{
  {
    QuadMat<double, 2> m; 
    QuadMat<double, 3> n; 
    m(0, 0) = 1; 
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;
    for( size_t i=0; i<3; i++)
        for( size_t j=0; j<3; j++)
            n( i, j) = i + j;
    n( 2,2 ) = 17;

    cout << "Test of QuadMat class\n";
    cout << "Output operations m and n:\n";
    cout << m <<endl;
    cout << n <<endl;
    QuadMat<double, 3> k(n);
    cout << "k(n)\n" << k<<endl;
    
    cout << "invert m and n\n";
    try{
        invert<double>( m);
        invert( n);
    }
    catch ( Message& message) {message.display();}
    cout << m <<endl;
    cout << n <<endl;
    k = n;
    cout << "k = n\n" << k<<endl;
    cout << "comparison: \n k == n: " << (k==n) << " and k !=n " << (k!=n) <<endl;
    n(1,1) = 0;
    cout << "comparison with changed n: \n k == n: " << (k==n) << " and k !=n " << (k!=n) <<endl;


  }
  {
    cout<< "Test of complex class\n";
    QuadMat< complex<double>, 2> m; 
    QuadMat< complex<double>, 3> n; 
    m(0, 0) = 1; 
    m(0, 1) = 2;
    m(1, 0) = 3;
    m(1, 1) = 4;
    for( size_t i=0; i<3; i++)
        for( size_t j=0; j<3; j++)
            n( i, j) = i + j;
    n( 2,2 ) = 17;

    cout << "Test of QuadMat class\n";
    cout << m <<endl;
    cout << n <<endl;
    cout << "invert m and n\n";
    try{
        invert<std::complex<double>>( m);
        invert( n);
    }
    catch ( Message& message) {message.display();}
    cout << m <<endl;
    cout << n <<endl;
  }
    return 0;

}


