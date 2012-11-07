#include <iostream>
#include "ghostmatrix.h"


using namespace std;
using namespace toefl;

int main()
{
    GhostMatrix< double, TL_NONE> gm( 3,4);
    gm.zero();
    for( int j = -1; j < 5; j++)
    {
        gm.at( -1, j) = 10;
        gm.at( 3, j) = 15;
    }
    gm(1,1) = 9;
    for( int i = 0; i < 3; i++)
    {
        gm.at( i, -1) = 4;
        gm.at( i, 4) = 7;
    }

    for( int i = -1; i < 4; i++)
    {
        for( int j = -1; j < 5; j++)
            cout << gm.at(i,j) << " ";
        cout << endl;
    }

    return 0;
}
