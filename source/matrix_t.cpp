#include "matrix.h"
#include "numerics/fehler.h"

using namespace std;
typedef Matrix<double> DoubMat;

int main()
{
    cout << "Test der TL DoubMatklasse\n";
    cout << "Construction m1\n";
    DoubMat m1(2,8);
    cout << "Construktion m2\n";
    DoubMat m2(2,8,9);

    cout << "Copy m3(m2)\n";
    DoubMat m3(m2);
    cout << "swap m1 m2\n";
    swap( m1, m2);
    cout << "const DoubMat\n";
    const DoubMat mc( m1);
    cout << "try Error handling on wrong index\n";
    try{ 
        cout << mc(9,2);
    }
    catch ( Fehler& fehler){ 
        fehler.anzeigen();}
    cout << "Assignment m1 = m3\n";
    m2 = m3;
    cout << "cout m1 m2 m3 sollten alle gleich sein\n";
    cout << m1 << "\n" << m2 << "\n" << m3 << endl;
    cout << "Assignment m1 = 0, m2=3 \n";
    m1 = 0, m2 = 3;
    cout << m1 << "\n" << m2 << "\n" << m3 << endl;
    cout << "Permute clockwise\n";
    permute_cw( m1, m2, m3);
    cout << m1 << "\n" << m2 << "\n" << m3 << endl;
    return 0;
}
