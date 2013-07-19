#include <iostream>
#include <iomanip>
#include "dlt.h"
#include "operators.h"

using namespace std;
using namespace dg;

#define P 4
int main()
{
    Operator<double, 2> Forward2(  DLT<2>::forward);
    Operator<double, 2> Backward2( DLT<2>::backward);
    Operator<double, 3> Forward3(  DLT<3>::forward);
    Operator<double, 3> Backward3( DLT<3>::backward);
    Operator<double, 4> Forward4(  DLT<4>::forward);
    Operator<double, 4> Backward4( DLT<4>::backward);
    Operator<double, 5> Forward5(  DLT<5>::forward);
    Operator<double, 5> Backward5( DLT<5>::backward);

    cout << "Hopefully Forward times Backward gives One\n";
    cout << Forward2*Backward2<<endl;
    cout << Forward3*Backward3<<endl;
    cout << Forward4*Backward4<<endl;
    cout << Forward5*Backward5<<endl;

    double x=0;
    cout << "Dlt trafo of y(x) = x should give one element\n";
    Operator<double, P> Forward ( DLT<P>::forward);
    Operator<double, P> Backward( DLT<P>::backward);
    for( unsigned i=0; i<P; i++)
    {
        x=0;
        for( unsigned j=0; j<P; j++)
            x+= Forward(i,j)*DLT<P>::abscissa[j];
        cout << x <<endl;
    }

    return 0;
}
