#include <iostream>
#include "arakawa.h"
#include "matrix.h"

using namespace std;
using namespace toefl;

const double h = 1.;
const double c = 1./(12.*h*h);
int main()
{
    Matrix<double> lhs( 5, 5), rhs( 5, 5), jac( 5, 5);
    Matrix<double> lhs2( 3,3), rhs2( 3,3), jac2( 3,3);
    jac.zero(), lhs.zero(), rhs.zero();
    Matrix<double> jac3(jac);
    Arakawa arakawa(h);

    for( int i=1; i<4; i++)
        for( int j=1; j<4; j++)
        {
            lhs2( i-1, j-1) = lhs( i, j) = i + 7*j;
            rhs2( i-1, j-1) = rhs( i, j) = 2*i +3*j;
        }
    for( int i=1; i<4; i++)
        for( int j=1; j<4; j++)
        {
            jac( i, j) = c* interior( i, j, lhs, rhs);
        }
    cout << "arakawa by interior routine\n";
    cout << "lhs\n" <<lhs << endl <<"rhs\n"<< rhs <<endl;
    cout << "jac\n" <<jac << endl <<endl;
    jac3(1,1) = c*corner<1,1>( 1, 1, lhs, rhs);
    jac3(3,3) = c*corner<-1,-1>( 3,3, lhs, rhs);
    jac3(1,3) = c*corner<1,-1>( 1, 3, lhs, rhs);
    jac3(3,1) = c*corner<-1,1>( 3,1, lhs, rhs);
    jac3(1,2) = c*edge<1,0>(1,2,lhs, rhs, 0);
    jac3(2,1) = c*edge<0,1>(2,1,lhs, rhs, 0);
    jac3(3,2) = c*edge<-1,0>(3,2,lhs, rhs, 0);
    jac3(2,3) = c*edge<0,-1>(2,3,lhs, rhs, 0);
    jac3(2,2) = c*boundary( 2,2, lhs, rhs);
    if( jac == jac3) 
        cout << "corner and edge test passed\n";
    else 
    {
        cout << "corner and edge test failed! See jac2: \n";
        cout << jac3 <<endl;
    }
    try{
        arakawa.dir_dir( lhs2, rhs2, jac2);}
    catch( Message& message) { message.display(); }

    {
        Matrix<double> lhs( 5, 5), rhs( 5, 5), jac( 5, 5);
        jac.zero(), lhs.zero(), rhs.zero();
        Matrix<double> jac2(jac);
        Arakawa arakawa(h);


        for( int i=0; i<5; i++)
            for( int j=0; j<5; j++)
            {
                lhs( i, j) = i + 7*j;
                rhs( i, j) = 2*i +3*j;
            }
        cout << "test of periodic BC\n";
        cout << "lhs\n" << lhs<<"rhs\n"<<rhs<<endl;
        for( int i=0; i<5; i++)
            for( int j=0; j<5; j++)
            {
                jac( i, j) = c* boundary( i, j, lhs, rhs);
            }
        cout << "jac\n"<<jac<<endl;
        try{
            arakawa.per_per( lhs, rhs, jac2);}
        catch( Message& message) { message.display(); }
        if( jac == jac2) 
            cout << "periodic test passed\n";
        else 
        {
            cout << "periodic test failed! See jac2: \n";
            cout << jac2 <<endl;
        }
    }
    {
        Matrix<double> lhs( 5, 5), rhs( 5, 5), jac( 5, 5);
        Matrix<double> lhs2( 5, 3), rhs2( 5, 3), jac2( 5, 3);
        jac.zero(), lhs.zero(), rhs.zero();
        Arakawa arakawa(h);


        for( int i=0; i<5; i++)
            for( int j=1; j<4; j++)
            {
                lhs2( i, j -1) = lhs( i, j) = i + 7*j;
                rhs2( i, j-1)  = rhs( i, j) = 2*i +3*j;
            }
        cout << "test of periodic BC\n";
        cout << "lhs\n" << lhs<<"rhs\n"<<rhs<<endl;
        for( int i=0; i<5; i++)
            for( int j=1; j<4; j++)
            {
                jac( i, j) = c* boundary( i, j, lhs, rhs);
            }
        cout << "jac\n"<<jac<<endl;
        try{
            arakawa.dir_per( lhs2, rhs2, jac2);}
        catch( Message& message) { message.display(); }
        cout << "jac2\n";
        cout << jac2 <<endl;
    }
    {
        Matrix<double> lhs( 5, 5), rhs( 5, 5), jac( 5, 5);
        Matrix<double> lhs2( 3, 5), rhs2( 3, 5), jac2( 3, 5);
        jac.zero(), lhs.zero(), rhs.zero();
        Arakawa arakawa(h);

        for( int i=1; i<4; i++)
            for( int j=0; j<5; j++)
            {
                lhs2( i-1, j )  = lhs( i, j) = i + 7*j;
                rhs2( i-1, j )  = rhs( i, j) = 2*i +3*j;
            }
        cout << "test of periodic_dirichlet BC\n";
        cout << "lhs\n" << lhs<<"rhs\n"<<rhs<<endl;
        for( int i=1; i<4; i++)
            for( int j=0; j<5; j++)
            {
                jac( i, j) = c* boundary( i, j, lhs, rhs);
            }
        cout << "jac\n"<<jac<<endl;
        try{
            arakawa.per_dir( lhs2, rhs2, jac2);}
        catch( Message& message) { message.display(); }
        cout << "jac2\n";
        cout << jac2 <<endl;
    }



    return 0;
}
