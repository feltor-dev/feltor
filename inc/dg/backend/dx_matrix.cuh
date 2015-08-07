#include <iostream>
#include <vector>
#include <cmath>


#include "grid.h"
#include "functions.h"
#include "operator.h"
#include "creation.cuh"

#ifdef DG_DEBUG
#include <cassert>
#endif


#include <thrust/inner_product.h>

#include "vector_categories.h"
#include "matrix_traits.h"


namespace dg
{

/***********************************************************************************
 *
 * Implementation of data type, giving position in dx_matrix
 *
 ***********************************************************************************/
class dx_matrix_pos
{
public:
    dx_matrix_pos(unsigned int N, unsigned int P) : N_(N), P_(P), row_(0), col_(0) {};
    dx_matrix_pos(unsigned int N, unsigned int P, unsigned int row, unsigned int col) : N_(N), P_(P), row_(row), col_(col) {};
    dx_matrix_pos(const dx_matrix_pos &rhs) : N_(rhs.N_), P_(rhs.P_), row_(rhs.row_), col_(rhs.col_) {};

    bool operator==(dx_matrix_pos rhs) { return ( (rhs.get_row() == row_) && (rhs.get_col() == col_)); };
    dx_matrix_pos& operator++();
    dx_matrix_pos operator++(int);

    unsigned int get_row() const { return(row_); };
    unsigned int get_col() const { return(col_); };


    // Returns the smallest column with a non-zero entry in the current row
    // This may be a negative number! Accessing elements in dx_matrix takes care of
    // wrap-arounds
    inline int get_min_col() { return( (static_cast<int>(get_row_block()) - 1) * P_); };
    // Returns the largest column with a non-zero entry in the current row
    inline int get_max_col() { return( (static_cast<int>(get_row_block()) + 2) * P_); };

    std::vector<int> get_block_list();

    // Returns the current block row, where each block is PxP
    inline unsigned int get_row_block() { return floor(double(row_) / double(P_)); };

    // Returns the current block column, where each block is PxP
    inline unsigned int get_col_block() { return floor(double(col_) / double(P_)); };
private:
        const unsigned int N_;
        const unsigned int P_;
        unsigned int row_;
        unsigned int col_;
};


std::vector<int> dx_matrix_pos :: get_block_list()
{
    std::vector<int> block_list(3);
    if(get_row_block() == 0)
    {
        // Do block 0 first, then 1 and finally N - 1
        block_list[0] = 0;
        block_list[1] = 1;
        block_list[2] = N_ - 1;
    }
    else if (get_row_block() == N_ - 1)
    {
        // Do block N - 2 first, then N - 1 and finally 0
        block_list[0] = 0;
        block_list[1] = N_ - 2;
        block_list[2] = N_ - 1;
    }
    else
    {
        // Do all blocks in order
        block_list[0] = get_row_block() - 1;
        block_list[1] = get_row_block();
        block_list[2] = get_row_block() + 1;
    }
    return block_list;
}

// Iterates over the non-zero elements of a dx_matrix
dx_matrix_pos& dx_matrix_pos :: operator++()
{
    unsigned int block_row = get_row_block();
    std::vector<int> block_list = get_block_list();
    static int current_block_idx = 0;
    static int current_col_idx = 0;

    //std::cout << "Before: current_block_idx = " << current_block_idx;
    //std::cout << "  current_col_idx = " << current_col_idx;
    //std::cout << "  row_ = " << row_;
    //std::cout << "  col_ = " << col_;

    // Traverse 3 * P columns in each row.
    if(current_col_idx < 3 * P_ - 1)
    {
        current_col_idx++;
        current_block_idx = floor(current_col_idx / P_);
    } 
    else 
    {
        current_col_idx = 0;
        current_block_idx = 0;
        row_++;
    }
    
    col_ = block_list[current_block_idx] * P_ + current_col_idx % 3;

    //std::cout << std::endl;
    //std::cout << "After: current_block_idx = " << current_block_idx;
    //std::cout << "  current_col_idx = " << current_col_idx;
    //std::cout << "  row_ = " << row_;
    //std::cout << "  col_ = " << col_;
    //std::cout << std::endl;
    //std::cout << std::endl;

    return *this;
}

// The postfix operator++
dx_matrix_pos dx_matrix_pos :: operator++(int)
{
    dx_matrix_pos retval(*this);
    ++(*this);
    return (retval);
}

//class dx_matrix_iter;

/*******************************************************************
 *
 * Data type for a tridiagonal banded sparse matrix
 *
 * On the main, upper and lower diagonal are PxP matrices.
 * Total diagonal size is N blocks of P*P matrices
 *
 *     / A    B    0   ...      C \
 *     | C    A    B   ...      0 |
 *     | 0    C    A   ...      0 |
 * M = |   ... ...        ...   0 |
 *     |                        B |
 *     \ B                 C    A /
 *
 *
 * M is (P*N) x (P*N)
 * A, B, C are PxP matrices
 *
 * Indexing:
 * M(i, j): i -> row, j -> column
 *
 *  
 *
 *
 *
 *
 *
 * Iterators:
 *  The non-zero elements in row i are given by:
 *  non-zero = [(row_block - 1) % N] * P ... [row_block + 2) % N] * P
 *  e.g. non-zero in row 1 = [(0 - 1) % N] * P ... [(0 + 2) % N] * P = [5P + 1... 6P] ... [1 ... 2P] 
 *                         = C A B
 *  row_block is the current block row, i.e. row_block = i % P, where i = 0...(P * N-1)
 *
 *******************************************************************/


class dx_matrix
{
public:
    dx_matrix(const unsigned int N,
              const unsigned int P,
              const double h, 
              const dg::bc bcx,
              const dg::direction dir);

    double operator() (const unsigned int n, const unsigned int m) const;
    double operator() (const dx_matrix_pos) const;

    //friend class dx_matrix_iter;
    // Assign values in a given row to the argument vector
    // vector with values in a given row
    void get_row_values(unsigned int, thrust::host_vector<double>&) const;
    
    //dx_matrix_iter begin() const; 
    //dx_matrix_iter end() const; 

    unsigned int get_N() const {return (N_); };
    unsigned int get_P() const {return (P_); };

private:
    // Returns the current block row, where each block is PxP
    inline unsigned int get_row_block(unsigned int n) const { return floor(double(n) / double(P_)); };

    // Returns the current block column, where each block is PxP
    inline unsigned int get_col_block(unsigned int m) const { return floor(double(m) / double(P_)); };

    inline bool is_on_main_diagonal(const unsigned int n, const unsigned int m) const {return(m == n);};
    inline bool is_on_main_diagonal(const dx_matrix_pos p) const {return(p.get_row() == p.get_col());};

    inline bool is_on_upper_diagonal(const unsigned int n, const unsigned int m) const {return (m == (n - 1));};
    inline bool is_on_upper_diagonal(const dx_matrix_pos p) const {return (p.get_row() == (p.get_col() - 1));};

    inline bool is_on_lower_diagonal(const unsigned int n, const unsigned int m) const {return (m == (n + 1));};
    inline bool is_on_lower_diagonal(const dx_matrix_pos p) const {return (p.get_row() == (p.get_col() + 1));};

    inline bool is_upper_left_bc(const unsigned int n, const unsigned int m) const {return ((m == 0) && (n == N_ - 1));};
    inline bool is_upper_left_bc(const dx_matrix_pos p) const {return ((p.get_col() == 0) && (p.get_col() == N_ - 1));};

    inline bool is_lower_right_bc(const unsigned int n, const unsigned int m) const {return((m == N_ - 1) && (n == 0));};
    inline bool is_lower_right_bc(const dx_matrix_pos p) const {return((p.get_col()== N_ - 1) && (p.get_col() == 0));};

    // Block data main, upper, and lower diagonal
    thrust::host_vector<double> main_diag_data;
    thrust::host_vector<double> upper_diag_data;
    thrust::host_vector<double> lower_diag_data;

    // Values for upper left and lower right data blocks (boundary condition)
    thrust::host_vector<double> bc_ul_data;
    thrust::host_vector<double> bc_lr_data;

    // Values for upper right and lower left data blocks (only non-zero for periodic BC)
    thrust::host_vector<double> bc_ur_data;
    thrust::host_vector<double> bc_ll_data;

    // Store occuring row value permutations
    thrust::host_vector<double> row_ABC;
    thrust::host_vector<double> row_CAB;
    thrust::host_vector<double> row_BCA;

    const unsigned int N_;
    const unsigned int P_;
};


dx_matrix :: dx_matrix(const unsigned int P,
                          const unsigned int N,
                          double h,
                          const dg::bc bcx,
                          const dg::direction dir) :
    main_diag_data(P * P),
    upper_diag_data(P * P),
    lower_diag_data(P * P),
    bc_ul_data(P * P),
    bc_lr_data(P * P),
    bc_ur_data(P * P, 0.0), // Initialize with zero. Change below if BC are not periodic 
    bc_ll_data(P * P, 0.0), // Initialize with zero. Change below if BC are not periodic
    row_ABC(3 * P * P),
    row_CAB(3 * P * P),
    row_BCA(3 * P * P),
    N_(N), P_(P)
{
    // Create block matrices
    Operator<double> l = create::lilj(P);
    Operator<double> r = create::rirj(P);
    Operator<double> lr = create::lirj(P);
    Operator<double> rl = create::rilj(P);
    Operator<double> d = create::pidxpj(P);
    Operator<double> t = create::pipj_inv(P);
    t *= 2. / h;

    Operator<double> a = 0.5 * t * (d - d.transpose());

    Operator<double> a_bc_right(a);
    Operator<double> a_bc_left(a);

    Operator<double> b = t * (0.5 * rl);
    Operator<double> bp = t * (-0.5 * lr);

    Grid1d<double> g(0.0, 1.0, P, N);
    Operator<double> backward = g.dlt().backward();
    Operator<double> forward = g.dlt().forward();

    switch (bcx)
    {
        case dg::DIR:
            a_bc_left = a + 0.5 * t * l;
            a_bc_right = a - 0.5 * t * r;
            break;

        case dg::DIR_NEU:
            a_bc_left = a + 0.5 * t * l;
            a_bc_right = a + 0.5 * t * r;
            break;

        case dg::NEU:
            a_bc_left = a - 0.5 * t * l;
            a_bc_right = a + 0.5 * t * r;
            break;

        case dg::NEU_DIR:
            a_bc_left = a -0.5 * t * l;
            a_bc_right = a - 0.5 * t * l;
            break;

        case dg::PER:
            a_bc_left = a;
            a_bc_right = a;
            break;
    }


    a = backward * a * forward;
    a_bc_left = backward * a_bc_left * forward;
    a_bc_right = backward * a_bc_right * forward;
    b = backward * b * forward;
    bp = backward * bp * forward;

    //std::cout << "a = " << a << std::endl;
    //std::cout << "a_bc_left = " << a_bc_left << std::endl;
    //std::cout << "a_bc_right = " << a_bc_right << std::endl;
    //std::cout << "b = " << b << std::endl;
    //std::cout << "bp = " << bp << std::endl;

    // Assemble matrix blocks
    switch (dir)
    {
    case dg::centered:
            main_diag_data = a.data();
            upper_diag_data = b.data();
            lower_diag_data = bp.data();
            bc_ul_data = a_bc_left.data();
            bc_lr_data = a_bc_right.data();
            if (bcx == dg::PER)
            {
                bc_ur_data = bp.data();
                bc_ll_data = b.data();
            }
            break;

        case dg::backward:
//            // not implemented yet
//            main_diag_data.assign(P_ * P_, 10.0);
//            upper_diag_data.assign(P_ * P_, 20.0); 
//            lower_diag_data.assign(P_ * P_, -40.0);
//
//            switch (bcx)
//            {
//                case dg::dirichlet:
//                    bc_ur_data.assign(P_ * P_, -10.0);
//                    bc_ll_data.assign(P_ * P_, -20.0);
//                    break;
//
//                case dg::neumann:
//                    bc_ur_data.assign(P_ * P_, -11.0);
//                    bc_ll_data.assign(P_ * P_, -21.0);
//                    break;
//
//                case dg::periodic:
//                    bc_ur_data.assign(P_ * P_, -12.0);
//                    bc_ll_data.assign(P_ * P_, -22.0);
//                    break;
//            } // switch(bcx)
            break;

        case dg::forward:
//            //not implemented yet
//            main_diag_data.assign(P_ * P_, 100.0);
//            upper_diag_data.assign(P_ * P_, 200.0); 
//            lower_diag_data.assign(P_ * P_, -400.0);
//
//            switch (bcx)
//            {
//                case dg::dirichlet:
//                    bc_ur_data.assign(P_ * P_, -100.0);
//                    bc_ll_data.assign(P_ * P_, -200.0);
//                    break;
//
//                case dg::neumann:
//                    bc_ur_data.assign(P_ * P_, -101.0);
//                    bc_ll_data.assign(P_ * P_, -201.0);
//                    break;
//
//                case dg::periodic:
//                    bc_ur_data.assign(P_ * P_, -102.0);
//                    bc_ll_data.assign(P_ * P_, -202.0);
//                    break;
//            } // switch(bcx)
            break;
    } // switch(dir)
    // Populate all occuring rows with data, taken from the blocks
    // The block matrices are PxP and there are three blocks per row.
    // Store the row data consecutively, as it occurs in any row.
    // row_ABC[0  .. 2P-1]  = first row of (A B C)
    // row_ABC[2P .. 5P-1] = second row of ABC ...
    //
    // Populate the column data vectors
    
    unsigned int row_offset = 0;
    for(int p_row = 0; p_row < P_; p_row++)
    {
        row_offset = p_row * (3 * P_);
        for(int p_col = 0; p_col < P_; p_col++)
        {
            row_ABC[row_offset +          p_col] = bc_ul_data[p_row * P_ + p_col];
            row_ABC[row_offset +     P_ + p_col] = upper_diag_data[p_row * P_ + p_col];
            row_ABC[row_offset + 2 * P_ + p_col] = bc_ur_data[p_row * P_+ p_col];

            row_CAB[row_offset +          p_col] = lower_diag_data[p_row * P_ + p_col];
            row_CAB[row_offset +     P_ + p_col] = main_diag_data[p_row * P_ + p_col];
            row_CAB[row_offset + 2 * P_ + p_col] = upper_diag_data[p_row * P_ + p_col];

            row_BCA[row_offset +          p_col] = bc_ll_data[p_row * P_ + p_col];
            row_BCA[row_offset +     P_ + p_col] = lower_diag_data[p_row * P_ + p_col];
            row_BCA[row_offset + 2 * P_ + p_col] = bc_lr_data[p_row * P_ + p_col];
        }
    }

//    for(unsigned int n = 0; n < N * P; n++)
//    {
//        for(unsigned int m = 0; m < N * P; m++)
//        {
//            std::cout << std::setprecision(5) << (*this)(n, m) << "\t";
//        }
//        std::cout << std::endl;
//    }

}

double dx_matrix :: operator() (const dx_matrix_pos p) const
{
    return operator()(p.get_row(), p.get_col());
}


double dx_matrix :: operator() (const unsigned int n, const unsigned int m) const
{
    // Find the block the index corresponds to
    const unsigned int row_block = get_row_block(n);
    const unsigned int col_block = get_col_block(m);

    // Indices within a block
    const unsigned int bidx_row = n % P_;
    const unsigned int bidx_col = m % P_;

    //cout << "N = " << N << ", P_ = " << P_ << "\tmapping (" << n << ", " << m << ") -> (" << bidx_row << ", " << bidx_col << ")\n";

    if (is_on_main_diagonal(row_block, col_block))
        if (row_block == 0)
            return bc_ul_data[bidx_row * P_ + bidx_col];
        else if (row_block == N_ - 1)
            return bc_lr_data[bidx_row * P_ + bidx_col];
        else
            return main_diag_data[bidx_row * P_ + bidx_col];

    else if (is_on_upper_diagonal(row_block, col_block))
        return upper_diag_data[bidx_row * P_ + bidx_col];

    else if (is_on_lower_diagonal(row_block, col_block))
        return lower_diag_data[bidx_row * P_ + bidx_col];

    else if (is_upper_left_bc(row_block, col_block))
        return bc_ur_data[bidx_row * P_ + bidx_col];

    else if (is_lower_right_bc(row_block, col_block))
        return bc_ll_data[bidx_row * P_ + bidx_col];

    else
        return double(0.0);
}


// Update vector result with values in a given row
void dx_matrix :: get_row_values(unsigned int row, thrust::host_vector<double>& result) const
{
    static const int row_size = 3 * P_;
    if(row / P_ == 0)
    {
        row = row % P_;
        result.assign(row_ABC.begin() + row * row_size, row_ABC.begin() + (row + 1) * row_size);
    } 
    else if (row / P_ == (N_ - 1))
    {
        row = row % P_;
        result.assign(row_BCA.begin() + row * row_size, row_BCA.begin() + (row + 1) * row_size);
    }
    else
    {
        row = row % P_;
        result.assign(row_CAB.begin() + row * row_size, row_CAB.begin() + (row + 1) * row_size);
    }
}


// Data type represent values in a single column. Stores data and indices in its own vector
template <typename T>
class dx_matrix_row
{
    public:
        dx_matrix_row(const dx_matrix&);
        dx_matrix_row(const dx_matrix&, const unsigned int);

        void assign_row(const dx_matrix&, const unsigned int);
        unsigned int get_row() const { return(row_); };

        // Getter elements on private data
        unsigned int get_N() const {return (N_);};
        unsigned int get_P() const {return (P_);};
        // Return view on data vector
        thrust::host_vector<T>& get_data() {return (data); };
        void set_data(thrust::host_vector<T>& new_data) { get_data() = new_data; };

        T& get_data(const unsigned int n) {return (data[n]);};

    private:
        // This vector stores the indices or values (specialized in template)  of the non-zero elements
        // of the dx_matrix
        thrust::host_vector<T> data;
        unsigned int row_;
        const unsigned int N_;
        const unsigned int P_;
};

template <typename T>
dx_matrix_row<T> :: dx_matrix_row(const dx_matrix& mat)
    : data(3 * mat.get_P(), T(0)), 
      row_(0), 
      N_(mat.get_N()), 
      P_(mat.get_P())
{
}

template <typename T>
dx_matrix_row<T> :: dx_matrix_row(const dx_matrix& mat, const unsigned int row)
    : data(3 * mat.get_P(), T(0)), 
      row_(row),
      N_(mat.get_N()), 
      P_(mat.get_P())
{
    //const unsigned int row_block = row_ / P_;
    //std::vector<int> block_list(3);
    //dx_matrix_pos pos(N_, P_, row_, 0);

    // Find the index of non-zero blocks in this row
    //block_list = pos.get_block_list();
    //std::cout << "Creating row object, row=" << row_ << std::endl;
    //std::cout << "\tblock_list = ";
    //for(int i = 0; i < 3; i++)
    //    std::cout << block_list[i] << " ";
    //std::cout << std::endl;
}


///////////////////////////////////////////////////////////////////////////
//
// Class representing the non-zero values in a given matrix row
//
//////////////////////////////////////////////////////////////////////////

class dx_matrix_row_val : public dx_matrix_row<double>
{
    public:
        dx_matrix_row_val(const dx_matrix&, const unsigned int);
        void update_row(const dx_matrix&, const unsigned int);
};


dx_matrix_row_val :: dx_matrix_row_val(const dx_matrix& mat, const unsigned int row)
    : dx_matrix_row(mat, row)
{
    mat.get_row_values(row, (*this).get_data());
//    for (int i = 0; i < 3; i++)
//    {
//        for(int p = 0; p < get_P(); p++)
//        {
//            std::cout << "Index: " << i * get_P() + p << ", data=" << get_data(i * get_P() + p) << "   ";
//        }
//    }
//    std::cout << std::endl;
}


void dx_matrix_row_val :: update_row(const dx_matrix& mat, const unsigned int row)
{
    mat.get_row_values(row, (*this).get_data());
}


///////////////////////////////////////////////////////////////////////////
//
// Row representing the indices of non-zero elements in a given matrix row
//
//////////////////////////////////////////////////////////////////////////

class dx_matrix_row_idx : public dx_matrix_row<int>
{
    public: 
        dx_matrix_row_idx(const dx_matrix& mat, const unsigned int);
        void update_row(const dx_matrix&, const unsigned int);
};

dx_matrix_row_idx :: dx_matrix_row_idx(const dx_matrix& mat, const unsigned int row)
    : dx_matrix_row(mat, row)
{
    std::vector<int> block_list(3);
    dx_matrix_pos pos(get_N(), get_P(), row, 0);
    block_list = pos.get_block_list();

    thrust::host_vector<int> new_data(get_data());

    // Populate index-vector of non-zero elements
    for (int i = 0; i < 3; i++)
    {
        for(int p = 0; p < get_P(); p++)
        {
            new_data[i * get_P() + p] = block_list[i] * get_P() + p;
        }
    }
    set_data(new_data);

//    for (int i = 0; i < 3; i++)
//    {
//        for(int p = 0; p < get_P(); p++)
//        {
//            std::cout << "Index: " << i * get_P() + p << ", data=" << get_data(i * get_P() + p) << "   ";
//        }
//    }
//    std::cout << std::endl;
}


void dx_matrix_row_idx :: update_row(const dx_matrix& mat, const unsigned int row)
{
    std::vector<int> block_list(3);
    dx_matrix_pos pos(get_N(), get_P(), row, 0);
    block_list = pos.get_block_list();

    thrust::host_vector<int> new_data(get_data());

    for(int i = 0; i < 3; i++)
    {
        for(int p = 0; p < get_P(); p++)
        {
            new_data[i * get_P() + p] = block_list[i] * get_P() + p;
        }
    }

    set_data(new_data);
}

///////////////////////////////////////////////////////////////////////////
//
// Implementation of dx_matrix_iter
//
//////////////////////////////////////////////////////////////////////////

//// iterator over all non-zero elements of the matrix
//class dx_matrix_iter
//{
//    public:
//        dx_matrix_iter(const dx_matrix&, dx_matrix_pos); 
//
//        // Comparison operator
//        bool operator== (const dx_matrix_iter& rhs); 
//        bool operator!= (const dx_matrix_iter& rhs);
//        dx_matrix_iter& operator++();
//        dx_matrix_iter operator++(int);
//
//        ///string get_pos();
//
//    private:    
//        const dx_matrix& dx_;
//        dx_matrix_pos pos_;
//};
//
//
//dx_matrix_iter :: dx_matrix_iter(const dx_matrix& dx, dx_matrix_pos pos) 
//    : dx_(dx), pos_(pos)
//{
//}
//
//
////string dx_matrix_iter :: get_pos()
////{
////    stringstream out;
////    out << "( " << pos_.get_col() << ", " << pos_.get_row() << ")";
////    return out.str();
////}
//
//dx_matrix_iter dx_matrix :: begin() const
//{ 
//    return dx_matrix_iter(*this, dx_matrix_pos(N_, P_, 0, 0)); 
//}
//
//dx_matrix_iter dx_matrix :: end() const
//{
//    return dx_matrix_iter(*this, dx_matrix_pos(N_, P_, N_ * P_ - 1, N_ * P_ - 1));
//};
//
//
//bool dx_matrix_iter :: operator==(const dx_matrix_iter &rhs) 
//{
//    return (( &((*this).dx_) == &(rhs.dx_)) && ((*this).pos_ == rhs.pos_));
//}
//
//
//bool dx_matrix_iter :: operator!=(const dx_matrix_iter &rhs)
//{
//    return !(this -> operator==(rhs));
//}
//
//
//dx_matrix_iter& dx_matrix_iter :: operator++()
//{
//    ++pos_;
//    return (*this);
//}
//
//dx_matrix_iter dx_matrix_iter :: operator++(int)
//{
//    dx_matrix_iter retval(*this);
//    pos_++;
//    return (retval); 
//}


template <>
struct MatrixTraits<dx_matrix>
{
    typedef double value_type;
    typedef dx_matrixTag matrix_category;
};


template<>
struct MatrixTraits<const dx_matrix>
{
    typedef double value_type;
    typedef dx_matrixTag matrix_category;
};

} //namespace dg



