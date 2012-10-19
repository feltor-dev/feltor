#include "matrix.h"


int main()
{
    Matrix m1(7,8);
    Matrix m2(7,8,9);

    Matrix m3(m2);
    m1 = m3;
    cout << m1 << "\n" << m2 << "\n" << m3 << endl;





    return 0;
}
