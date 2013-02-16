#include <iostream>
#include <iomanip>
#include "dlt.h"
#include "operators.h"

using namespace std;
using namespace dg;

int main()
{
    Operator<double, 2> Forward2(  DLT<2>::forward);
    Operator<double, 2> Backward2( DLT<2>::backward);
    Operator<double, 3> Forward3(  DLT<3>::forward);
    Operator<double, 3> Backward3( DLT<3>::backward);
    Operator<double, 4> Forward4(  DLT<4>::forward);
    Operator<double, 4> Backward4( DLT<4>::backward);

    cout << "Hopefully Forward times Backward gives One\n";
    cout << Forward2*Backward2<<endl;
    cout << Forward3*Backward3<<endl;
    cout << Forward4*Backward4<<endl;

    double x=0;
    for( unsigned i=0; i<2; i++)
    {
        x=0;
        for( unsigned j=0; j<2; j++)
            x+= Forward2(i,j)*DLT<2>::abscissa[j];
        cout << x <<endl;
    }




    return 0;
}
