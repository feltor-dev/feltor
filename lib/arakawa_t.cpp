#include <iostream>
#include <iomanip>
#include "ghostmatrix.h"
#include "arakawa.h"
#include "matrix.h"

using namespace std;
using namespace toefl;

const double h = 1.;
const double c = 1./(12.*h*h);
int main()
{
    int rows = 3, cols = 5;
    GhostMatrix<double> lhs( rows, cols), rhs( rows, cols);
    Matrix<double> jac( rows, cols);
    Arakawa arakawa(h);

    for( int i=0; i<rows; i++)
        for( int j=0; j<cols; j++)
        {
            lhs( i, j) = i + 7*j;
            rhs( i, j) = 2*i*i*i +3*j*j;
        }
    //Make Dirichlet BC
    for( int j = -1; j < cols + 1; j++)
    {
        rhs.at(-1, j) = lhs.at(-1, j) = 1;
        rhs.at(rows,j) = lhs.at(rows,j)= 1;
    }
    for( int i = 0; i < rows ; i++)
    {
        rhs.at( i, -1) = lhs.at(i, -1) = 1;
        rhs.at( i, cols) = lhs.at(i,cols)= 1;
    }
    cout << setprecision(2) << fixed;
    cout << lhs << endl << rhs <<endl;
    cout << "The Matrices including ghost Cells\n";
    lhs.display(cout);
    cout << endl;
    rhs.display(cout);
    cout << endl;
    arakawa( lhs, rhs, jac);
    cout << jac <<endl;
    return 0;
}
