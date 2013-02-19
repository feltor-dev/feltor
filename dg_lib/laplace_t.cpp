#include <iostream>
#include "laplace.h"

using namespace dg;
using namespace std;

int main()
{
    dg::Laplace<2> l(1./100.);
    dg::Laplace_Dir<2> ld(1./100.);
    cout << l.get_a()<<endl;
    cout << l.get_b()<<endl;
    cout << ld.get_ap() <<endl;
    cout << ld.get_bp() <<endl;

    return 0;
}
