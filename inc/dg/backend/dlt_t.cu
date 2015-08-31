#include <iostream>
#include <iomanip>

#include "dlt.h"
#include "operator.h"


int main()
{
    //Test if forward * backwards gives a Delta function and print abscissas and weights
    unsigned n;
    std::cout << "Type # of polynomial coefficients n! \n";
    std::cin >> n;
    dg::DLT<double> dlt( n);

    dg::Operator<double> forward( dlt.forward());
    dg::Operator<double> backward( dlt.backward());
    std::cout << "Forward * Backward should give Delta: \n";
    std::cout << std::setprecision(2);
    std::cout << forward*backward;
    std::cout << std::setprecision(16);
    std::cout << "Abscissas:\n";
    for( unsigned i=0; i<n; i++)
        std::cout << dlt.abscissas()[i]<<" ";
    std::cout << std::endl;
    std::cout << "Weights:\n";
    for( unsigned i=0; i<n; i++)
        std::cout << dlt.weights()[i]<<" ";
    std::cout << std::endl;
    return 0;
}
