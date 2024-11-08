#include <iostream>
#include <iomanip>

#include "dlt.h"
#include "operator.h"


int main()
{
    //Test if forward * backwards gives a Delta function and print abscissas and weights
    for( unsigned n=1; n<21; n++)
    {
        std::cout << "# of polynomial coefficients n! \t"<<n<<"\n";
        dg::Operator<double> forward( dg::DLT<double>::forward(n));
        dg::Operator<double> backward( dg::DLT<double>::backward(n));
        auto abs = dg::DLT<double>::abscissas(n);
        auto weights = dg::DLT<double>::weights(n);
        auto delta = forward*backward;
        std::cout << "Forward * Backward should give Delta: \n";
        for( unsigned k=0; k<n; k++)
        for( unsigned l=0; l<n; l++)
        {
            if( k == l)
            {
                if( fabs(delta(k,l) -1)  > 1e-15)
                {
                    std::cerr << "FAILED\n";
                    return -1;
                }
            }
            else
            {
                if( fabs(delta(k,l) )  > 1e-15)
                {
                    std::cerr << "FAILED\n";
                    return -1;
                }
            }
        }
        std::cout << std::fixed<<std::setprecision(0);
        std::cout << forward*backward;
        std::cout << std::setprecision(16);

        std::cout << "Abscissas:\n";
        double sum =0;
        for( unsigned i=0; i<n; i++)
        {
            sum += abs[i];
            std::cout << abs[i]<<" ";
        }
        std::cout << std::endl;
        std::cout << std::setprecision(6);
        std::cout<< "Sum of all abscissas\t"<<sum<<" (0)\n\n";
        std::cout << std::setprecision(16);
        std::cout << "Weights:\n";
        sum = 0;
        for( unsigned i=0; i<n; i++)
        {
            sum += weights[i];
            std::cout << weights[i]<<" ";
        }
        std::cout << std::endl;
        std::cout << std::setprecision(6);
        std::cout<< "Sum of all weights\t"<<sum<<" (2)\n\n";
    }
    return 0;
}
