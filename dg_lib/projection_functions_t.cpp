#include <iostream>
#include "projection_functions.h"
#include "operators.h"

using namespace dg; 
using namespace std;

#define P 2  //can go arbitrarily high
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
    Operator<double, P> A = (d+l)*t*(d+l).transpose() + lr*t*rl ;
    Operator<double, P> B = -(d+l)*t*rl ;
    array<double,P> a{{0,0}}, b{{2*2,2*2}}, c{{4*4,2*4}};
    a = t*B.transpose()*a; 
    b = t*A*b;
    c = t*B*c;
    cout << a[0] + b[0] + c[0] <<endl;
    cout << a[1] + b[1] + c[1] <<endl;

    




    return 0;
}
