#include <iostream>
#include "projection_functions.h"
#include "operators.h"

using namespace dg; 
using namespace std;

#define P 4  //can go arbitrarily high
int main()
{
    Operator<double, P> l( lilj);
    cout << "L is: \n";
    cout << l<<endl;
    Operator<double, P> r( rirj);
    cout << "R is: \n";
    cout << r<<endl;
    Operator<double, P> lr( lirj);
    cout << "LR is: \n";
    cout << lr<<endl;
    Operator<double, P> rl( rilj);
    cout << "RL is: \n";
    cout << rl<<endl;
    Operator<double, P> d( pidxpj);
    cout << "D is: \n";
    cout << d<<endl;
    Operator<double, P> s( pipj);
    cout << "S is: \n";
    cout << s<<endl;
    Operator<double, P> t( pipj_inv);
    cout << "T is: \n";
    cout << t<<endl;

    cout << "S*T is \n";
    cout << s*t<<endl;
    cout << "D*D is \n";
    cout << d*d<<endl;
    return 0;
}
