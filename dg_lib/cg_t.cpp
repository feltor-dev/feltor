#include <iostream>

#include "cg.h"
#include "evaluation.h"
#include "laplace.h"

#define P 2
const unsigned N = 100;

typedef std::vector<std::array<double,P>> ArrVec;
int main()
{
    ArrVec v(100);
    dg::CG<Laplace<P>, ArrVec, T> cg( v)

    return 0;
}
