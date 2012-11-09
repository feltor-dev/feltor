#include <iostream>
#include "timer.h"
#include "arakawa.h"
#include "matrix.h"
#include "ghostmatrix.h"

using namespace std;
using namespace toefl;

const double h = 1.;
const double c = 1./(12.*h*h);
const double hysq = -c;
const int nymax = 2049;
const int nxmax = 513;
    int rows = 500, cols = 2*2000;
    unsigned loop = 1;
    int nx1 = rows, ny1 = cols;
void arakawa2(double (*uuu)[nymax], double (*vvv)[nymax], double (*www)[nymax])
{ 
  int i0,j0,ip,jp,im,jm;
  double xxx;

  for (i0=1; i0<=nx1; ++i0)
    {
      ip = (i0==nx1) ? 0   : i0+1;  
      im = (i0==0)   ? nx1 : i0-1;

      for (j0=1; j0<=ny1; ++j0)
	{
	  jp = (j0==ny1) ? 0   : j0+1; 
	  jm = (j0==0)   ? ny1 : j0-1;
	  
	  xxx = vvv[i0][jm]*( uuu[ip][j0] -uuu[im][j0] -uuu[im][jm] +uuu[ip][jm] );
	  xxx+= vvv[i0][jp]*(-uuu[ip][j0] +uuu[im][j0] -uuu[ip][jp] +uuu[im][jp] );
	  xxx+= vvv[ip][j0]*( uuu[i0][jp] -uuu[i0][jm] +uuu[ip][jp] -uuu[ip][jm] );
	  xxx+= vvv[im][j0]*(-uuu[i0][jp] +uuu[i0][jm] +uuu[im][jm] -uuu[im][jp] );
	  xxx+= vvv[ip][jm]*( uuu[ip][j0] -uuu[i0][jm]);
	  xxx+= vvv[ip][jp]*( uuu[i0][jp] -uuu[ip][j0]);
	  xxx+= vvv[im][jm]*( uuu[i0][jm] -uuu[im][j0]);
	  xxx+= vvv[im][jp]*( uuu[im][j0] -uuu[i0][jp]);

	  www[i0][j0] = -hysq*xxx;
	};
    }
}
int main()
{
    Timer t;
    //cin >> rows >> cols;
    Matrix<double> lhs0( rows + 2, cols + 2), rhs0( rows + 2, cols + 2); //Matrix with ghostcells
    GhostMatrix<double> lhs( rows, cols), rhs( rows, cols);
    GhostMatrix<double> jac( rows, cols);
    Matrix<double> jac0( rows, cols);
    //double uuu[nxmax][nymax], vvv[nxmax][nymax], www[nxmax][nymax];
    
    Arakawa arakawa(h);

    for( int i=0; i<rows; i++)
        for( int j=0; j<cols; j++)
        {
            lhs( i, j) = i + 7*j;
            rhs( i, j) = 2*i*i*i +3*j*j;
        }

    for( int i=0; i<rows; i++)
        for( int j=0; j<cols; j++)
        {
            /*uuu[i+1][j+1] =*/ lhs0( i + 1, j + 1) = i + 7*j;
            /*vvv[i+1][j+1] =*/ rhs0( i + 1, j + 1) = 2*i*i*i +3*j*j;
        }

    //Make Dirichlet BC
    cout << "Completely with interior function\n";
    t.tic();
    for( unsigned k = 0; k < loop; k++)
    {
        for( int j = -1; j < cols + 1; j++)
        {
            /*uuu[0][j+1] = vvv[0][j+1] =*/ rhs0(-1+1, j+1)  = lhs0(-1+1, j+1) = 1;
            /*uuu[rows+1][j+1] = vvv[rows+1][j+1] =*/ rhs0(rows+1,j+1) = lhs0(rows+1,j+1)= 1;
        }
        for( int i = 0; i < rows ; i++)
        {
            /*uuu[i+1][0] = vvv[i+1][0] =*/ rhs0( i+1, -1+1)   = lhs0(i+1, -1+1) = 1;
            /*uuu[i+1][cols+1] = uuu[i+1][cols+1] =*/ rhs0( i+1, cols+1) = lhs0(i+1,cols+1)= 1;
        }
        for( size_t i = 1; i < (size_t)rows+1; i++)
            for( size_t j = 1; j < (size_t)cols +1; j++)
                jac0( i-1, j-1) = c*interior(i,j,lhs0, rhs0);
    }
    t.toc();
    cout << "Arakawa scheme took " <<t.diff() <<" seconds\n";
    //Make Dirichlet BC
    cout << "The Matrices including ghost Cells\n";
    t.tic();
    for( unsigned i = 0; i < loop; i++)
    {
        for( int j = -1; j < cols + 1; j++)
        {
            rhs.at(-1, j)  = lhs.at(-1, j) = 1;
            rhs.at(rows,j) = lhs.at(rows,j)= 1;
        }
        for( int i = 0; i < rows ; i++)
        {
            rhs.at( i, -1)   = lhs.at(i, -1) = 1;
            rhs.at( i, cols) = lhs.at(i,cols)= 1;
        }
        arakawa( lhs, rhs, jac);
    }
    t.toc();
    cout << "Arakawa scheme took " <<t.diff() <<" seconds\n";
    if( jac!=jac0)
        cerr << "An error occured!\n" << jac << "\n"<<jac0;

    //cout << "Completely with boundary function\n";
    //t.tic();
    //for( unsigned k = 0; k < loop; k++)
    //{
    //    for( size_t i = 0; i < (size_t)rows; i++)
    //        for( size_t j = 0; j < (size_t)cols; j++)
    //            jac( i, j) = c*boundary(i,j,lhs, rhs);
    //}
    //t.toc();
    //cout << "Arakawa scheme took " <<t.diff() <<" seconds\n";
    //if( jac!=jac0)
    //    cerr << "An error occured!\n" << jac << "\n"<<jac0;
    /*
    cout << "Alexanders function\n";
    t.tic();
    for( unsigned k = 0; k < loop; k++)
    {
        arakawa2( uuu, vvv, www);
    }
    t.toc();
    cout << "Arakawa scheme took " <<t.diff() <<" seconds\n";
    */

    /*
    cout << jac << endl;
    for( int i=1; i<=nx1; i++)
    {
        for( int j=1; j<= ny1; j++)
            cout << www[i][j] << " ";
        cout << endl;
    }
    */



    return 0;
}
