#include <iostream>
#include "operators.h"


using namespace std;
using namespace dg;


int main()
{
    Operator<double, 3> A(1);
    Operator<double, 3> B(2);


    cout << "Test of Operator class\n";
    A+=B; 

    cout << A <<endl <<B<<endl<< A+B << endl;
    cout << A-2.*B<<endl;
    A-=B;
    cout << A*B;



    return 0;
}
