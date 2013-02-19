#include "matrix_array.h"
#include <array>
#include "quadmat.h"
#include "matrix.h"

using namespace toefl;
using namespace std;


class Container
{
    std::array< Matrix<double>, 3> a;
    public:
    Container( size_t rows, size_t cols): a(MatrixArray<double, TL_NONE,3>::construct(rows,cols)){}
};

int main()
{
    std::array<Matrix<double>,2> a {{Matrix<double>{ 3,4}, Matrix<double>{ 3,4}}};
    auto b =  MatrixArray<double, TL_NONE, 3>::construct(3,3) ;
    Container c(4,4);

    return 0;
}
