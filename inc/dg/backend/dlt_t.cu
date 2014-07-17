#include <iostream>
#include <iomanip>

#include "dlt.cuh"
#include "operator.h"


using namespace std;
using namespace dg;


int main()
{
    unsigned n;
    cout << "Type n! \n";
    cin >> n;
    DLT<double> dlt( n);

    Operator<double> forward( dlt.forward());
    Operator<double> backward( dlt.backward());
    cout << "Forward * Backward should give Delta: \n";
    cout << setprecision(2);
    cout << forward*backward;
    cout << setprecision(16);
    cout << "Abscissas:\n";
    for( unsigned i=0; i<n; i++)
        cout << dlt.abscissas()[i]<<" ";
    cout << endl;
    cout << "Weights:\n";
    for( unsigned i=0; i<n; i++)
        cout << dlt.weights()[i]<<" ";
    cout << endl;
    return 0;
}
