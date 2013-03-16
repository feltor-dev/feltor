#include <cusp/array1d.h>
#include <cusp/print.h>
#include <thrust/device_vector.h>


typedef thrust::device_vector<double> DVector;
int main()
{
    DVector dv( 10, 2);
    cusp::array1d_view<DVector::iterator> dv_view( dv.begin(), dv.end());
    cusp::print( dv_view);
    return 0;
}
