#include <iostream>

#include "cusp_eigen.h"


#include "laplace.cuh"
#include "arrvec1d.cuh"

const unsigned n = 2;
const unsigned N = 5; //minimum 3

typedef thrust::host_vector< double>     HVec;
typedef dg::ArrVec1d< double, n, HVec>  HArrVec;

typedef cusp::coo_matrix<int, double, cusp::host_memory> HMatrix;


using namespace std;
int main()
{
    HMatrix hmatrix = dg::create::laplace1d_per<double, n>( N, 1.); 
    Eigen::SparseMatrix<double> ematrix = dg::convert( hmatrix);
    
    cout << ematrix <<endl;


    return 0;
}
