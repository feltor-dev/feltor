#include <iostream>
#include "operator.h"

int main()
{

    dg::Operator<double> lilj = dg::create::lilj<double>(3);
    dg::Operator<double> pidxpj = dg::create::pidxpj<double>(3);
    dg::Operator<double> rilj = dg::create::rilj<double>(3);

    dg::Operator<double> op = lilj + pidxpj;
    //dg::Operator<double> op = pidxpj - pidxpj.transpose();
    //op(0,0) = 1, op(0,1) = 2, op(0,2) = 0;
    //op(1,0) = 2, op(1,1) = 4, op(1,2) = 1;
    //op(2,0) = 2, op(2,1) = 1, op(2,2) = 0;
    std::cout << "Operator\n"<<op<<"\n";
    dg::Operator<double> inv_op = dg::create::inverse( op);
    std::vector<unsigned> pivot( op.size());
    dg::Operator<double> lu(op);
    double det= dg::create::lu_pivot( lu, pivot);
    std::cout << "Determinant "<<det<<"\n";
    std::cout << "lu decomposition\n"<<lu<<"\n";
    std::cout << "pivot\n";
    for( unsigned i=0; i<op.size(); i++)
        std::cout << pivot[i] <<" ";
    std::cout << "\n";
    std::cout << "Inverse Operator\n"<<inv_op<<"\n";
    std::cout << "Multiplication\n"<<inv_op*op<<"\n";

    //op.zero();
    op(0,2) = op(1,1) = op(2,0) = 0;// op(3,3)= 1;
    std::cout << "Operator\n"<<op<<"\n";
    inv_op = dg::create::inverse(op);
    lu = op;
    det= dg::create::lu_pivot( lu, pivot);
    std::cout << "Determinant "<<det<<"\n";
    std::cout << "lu decomposition\n"<<lu<<"\n";
    std::cout << "pivot\n";
    for( unsigned i=0; i<op.size(); i++)
        std::cout << pivot[i] <<" ";
    std::cout << "\n";
    std::cout << "Inverse Operator\n"<<inv_op<<"\n";
    std::cout << "Multiplication\n"<<op*inv_op<<"\n";



    return 0;
}
