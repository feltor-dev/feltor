#include <iostream>
#include "arakawa.h"
#include "matrix.h"

using namespace std;
using namespace toefl;

const double h = 1./100.;
const double c = 1./(12.*h*h);
int main()
{
    Matrix<double> lhs( 5, 5), rhs( 5, 5), jac( 5, 5);
    jac.zero(), lhs.zero(), rhs.zero();
    Matrix<double> jac2(jac);
    Arakawa<TL_PER, TL_PER> arakawa(h);

    for( int i=1; i<4; i++)
        for( int j=1; j<4; j++)
        {
            lhs( i, j) = i + 7*j;
            rhs( i, j) = 2*i +3*j;
        }
    for( int i=1; i<4; i++)
        for( int j=1; j<4; j++)
        {
            jac( i, j) = interior( i, j, lhs, rhs);
        }
    cout << "arakawa by interior routine\n";
    cout << "lhs\n" <<lhs << endl <<"rhs\n"<< rhs <<endl;
    cout << jac << endl <<endl;
    jac2(1,1) = corner<1,1>( 1, 1, lhs, rhs);
    jac2(3,3) = corner<-1,-1>( 3,3, lhs, rhs);
    jac2(1,3) = corner<1,-1>( 1, 3, lhs, rhs);
    jac2(3,1) = corner<-1,1>( 3,1, lhs, rhs);
    jac2(1,2) = edge<1,0>(1,2,lhs, rhs);
    jac2(2,1) = edge<0,1>(2,1,lhs, rhs);
    jac2(3,2) = edge<-1,0>(3,2,lhs, rhs);
    jac2(2,3) = edge<0,-1>(2,3,lhs, rhs);
    jac2(2,2) = boundary( 2,2, lhs, rhs);
    if( jac == jac2) 
        cout << "corner and edge test passed\n";
    else 
    {
        cout << "corner and edge test failed! See jac2: \n";
        cout << jac2 <<endl;
    }



    return 0;
}
