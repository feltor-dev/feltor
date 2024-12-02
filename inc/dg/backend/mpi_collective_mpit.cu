#include <iostream>

#include <mpi.h>
#include "../blas1.h"
#include "mpi_collective.h"
#include "mpi_vector.h"


int main( int argc, char * argv[])
{
    MPI_Init( &argc, &argv);
    unsigned N = 900;
    int rank, size;
    MPI_Comm_rank( MPI_COMM_WORLD, &rank);
    MPI_Comm_size( MPI_COMM_WORLD, &size);
    if(size!=4 ){std::cerr <<"You run with "<<size<<" processes. Run with 4 processes!\n"; MPI_Finalize(); return 0;}

    if(rank==0)std::cout <<"N " <<N << std::endl;
    if(rank==0)std::cout <<"# processes =  " <<size <<std::endl;

    std::vector<std::array<int,2>> gIdx {
    {0, 1}, {6, 2}, {6,7}, {2,0,}, {1,0}, {1,2}, {1,1}, {6,7}, {0,1}, {0,4}, {2,0}};
    thrust::host_vector<std::array<int,2>> unique;
    thrust::host_vector<int> bufferIdx;
    dg::detail::global2bufferIdx( gIdx, bufferIdx, unique);

    auto send = dg:detail::lugi2sendTo( unique, 7);

    if(rank==0)std::cout<< "Found unique values \n";
    for( unsigned u=0; u<unique.size(); u++)
        if(rank==0)std::cout << unique[u][0]<<" "<<unique[u][1]<<"\n";
    if(rank==0)std::cout << std::endl;
    if(rank==0)std::cout<< "Found unique pids \n";
    for( unsigned u=0; u<send.size(); u++)
        if(rank==0)std::cout << "pid "<<u<<" "<<send[u]<<"\n";

    if(rank==0)std::cout << "Test BijectiveComm: scatter followed by gather leaves the input vector intact"<<std::endl;
    thrust::host_vector<double> v( N, 3*rank);
    thrust::host_vector<double> m( N);
    for( unsigned i=0; i<m.size(); i++)
    {
        m[i] = i%3 + rank/2;
        v[i] = v[i] + (double)(i + 17%(rank+1));
    }
    const thrust::host_vector<double> w(v);
    dg::BijectiveComm<thrust::host_vector<int>, thrust::host_vector<double> > c(m, MPI_COMM_WORLD);
    thrust::host_vector<double> receive = c.allocate_buffer();
    receive = c.global_gather( thrust::raw_pointer_cast(v.data()));
    MPI_Barrier( MPI_COMM_WORLD);
    c.global_scatter_reduce( receive, thrust::raw_pointer_cast(v.data()));
    bool equal = true;
    for( unsigned i=0; i<m.size(); i++)
        if( v[i] != w[i]) { equal = false; }
    {
        if( equal)
            std::cout <<"Rank "<<rank<<" PASSED"<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED"<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    {
    if(rank==0)std::cout << "Test bijective branch of SurjectiveComm:"<<std::endl;
    // First get localIndexMap of Bijective
    thrust::host_vector<double> seq( c.local_size()), pids(c.local_size(), rank);
    thrust::sequence( seq.begin(), seq.end());
    auto idx_map = dg::construct<thrust::host_vector<int>>(c.global_gather( &seq[0]));
    auto pid_map = dg::construct<thrust::host_vector<int>>(c.global_gather( &pids[0]));
    dg::SurjectiveComm<thrust::host_vector<int>, thrust::host_vector<double>> sur( seq.size(), idx_map, pid_map, MPI_COMM_WORLD);
    std::cout << "Rank "<<rank<<" Surjective is bijective? "<<std::boolalpha<<sur.isLocalBijective()<<" (true)\n";
    MPI_Barrier(MPI_COMM_WORLD);
    auto rec_bi = c.global_gather( thrust::raw_pointer_cast(v.data()));
    auto rec_su = sur.global_gather( thrust::raw_pointer_cast(v.data()));
    equal = true;
    if(rank==0)std::cout << "Bijective and Surjective do the same?"<<std::endl;
    for( unsigned i=0; i<rec_bi.size(); i++)
        if( rec_bi[i] != rec_su[i]) { equal = false; }
    {
        if( equal)
            std::cout <<"Rank "<<rank<<" PASSED"<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED"<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0)std::cout << "Surjective w/o reduce workds?"<<std::endl;
    sur.global_scatter_reduce( rec_su, thrust::raw_pointer_cast(v.data()));
    equal = true;
    for( unsigned i=0; i<m.size(); i++)
        if( v[i] != w[i]) { equal = false; }
    {
        if( equal)
            std::cout <<"Rank "<<rank<<" PASSED"<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED"<<std::endl;
    }
    MPI_Barrier(MPI_COMM_WORLD);
    }
    if(rank==0)std::cout << "Test GeneralComm and SurjectiveComm:"<<std::endl;
    if(rank==0)std::cout << "Test Norm Gv*Gv vs G^T(Gv)*v: "<<std::endl;
    N = 20;
    thrust::host_vector<double> vec( N, rank);
    for( int i=0; i<(int)(N); i++)
        vec[i]*=i;
    thrust::host_vector<int> idx( N+rank), pids( N+rank);
    for( unsigned i=0; i<N+rank; i++)
    {
        idx[i] = i%N;
        if(i>=5 && i<10) idx[i]+=3;
        pids[i] = rank;
        if( i>=N) pids[i] = (rank+1)%size;
    }
    dg::SurjectiveComm<thrust::host_vector<int>, thrust::host_vector<double>> sur( N, idx, pids, MPI_COMM_WORLD);
    std::cout << "Rank "<<rank<<" Surjective is bijective? "<<std::boolalpha<<sur.isLocalBijective()<<" (false)\n";
    dg::GeneralComm<thrust::host_vector<int>, thrust::host_vector<double> > s2( N, idx, pids, MPI_COMM_WORLD), s(s2);
    receive = s.global_gather( thrust::raw_pointer_cast(vec.data()));
    /// Test if global_scatter_reduce is in fact the transpose of global_gather
    dg::MPI_Vector<thrust::host_vector<double>> mpi_receive( receive, MPI_COMM_WORLD);
    double norm1 = dg::blas1::dot( mpi_receive, mpi_receive);
    thrust::host_vector<double> vec2(vec.size(), 7e6);
    s.global_scatter_reduce( receive, thrust::raw_pointer_cast(vec2.data()));
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec( vec, MPI_COMM_WORLD);
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec2( vec2, MPI_COMM_WORLD);
    double norm2 = dg::blas1::dot( mpi_vec, mpi_vec2);
    {
        if( fabs(norm1-norm2)<1e-14)
            std::cout <<"Rank "<<rank<<" PASSED "<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED "<<std::endl;
    }
    if(fabs(norm1-norm2)>1e-14 && rank==0)std::cout << norm1 << " "<<norm2<< " "<<norm1-norm2<<std::endl;
    if(rank==0) std::cout << "Test non-communicating versions\n";
    for( unsigned i=0; i<N+rank; i++)
        pids[i] = rank;
    dg::GeneralComm<thrust::host_vector<int>, thrust::host_vector<double>> s3( N, idx, pids, MPI_COMM_WORLD);
    /// Test if global_scatter_reduce is in fact the transpose of global_gather
    receive = s3.global_gather( thrust::raw_pointer_cast(vec.data()));
    dg::MPI_Vector<thrust::host_vector<double>> mpi_receive2( receive, MPI_COMM_WORLD);
    norm1 = dg::blas1::dot( mpi_receive2, mpi_receive2);
    s3.global_scatter_reduce( receive, thrust::raw_pointer_cast(vec2.data()));
    dg::MPI_Vector<thrust::host_vector<double>> mpi_vec3( vec2, MPI_COMM_WORLD);
    std::cout << "Rank "<<rank<<" is communicating? "<<s3.isCommunicating()<<" (false)\n";
    // Just to show that even if not communciating the size is not zero
    if(rank==0)std::cout << "Rank "<<rank<<" buffer size "<<s3.buffer_size()<<" \n";
    if(rank==0)std::cout << "Rank "<<rank<<" local  size "<<s3.local_size()<<" \n";
    norm2 = dg::blas1::dot( mpi_vec, mpi_vec3);
    {
        if( fabs(norm1-norm2)<1e-14)
            std::cout <<"Rank "<<rank<<" PASSED "<<std::endl;
        else
            std::cerr <<"Rank "<<rank<<" FAILED "<<std::endl;
    }
    if(fabs(norm1-norm2)>1e-14 && rank==0)std::cout << norm1 << " "<<norm2<< " "<<norm1-norm2<<std::endl;

    MPI_Finalize();

    return 0;
}
