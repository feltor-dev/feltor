#include <iostream>
#include "operators.cuh"
#include "array.cuh"


using namespace std;
using namespace dg;

double initializer( unsigned i, unsigned j)
{
    return i+2*j;
}

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

    cout << "Operator and its transpose:\n";
    Operator<double, 4> C( initializer);
    cout << C <<endl;
    cout << C.transpose()<<endl;
    cout << "Negation -C \n";
    cout << -C<<endl;
    cout << "Access operator C(1,2) \n";
    cout << C(1,2)<<endl;



    return 0;
}
