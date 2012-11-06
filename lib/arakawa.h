/*!
 * \file
 * \brief Implementation of the arakawa scheme
 * \author Matthias Wiesenberger
 * \email Matthias.Wiesenberger@uibk.ac.at
 * 
 */
#ifndef _ARAKAWA_
#define _ARAKAWA_

#include "quadmat.h"
#include "vector.h"

namespace toefl{

    enum BC{ TL_PER, TL_DIR};
    
    /*! @brief compute an interior point in the arakawa scheme
     *
     * @tparm M the type of the matrix
     * @param i0 row index
     * @param j0 column index
     * @param lhs left hand side
     * @param rhs right hand size
     * @return unnormalized value of the arakawa bracket
     */
    template< class M>
    static double interior( const size_t i0, const size_t j0, const M& lhs, const M& rhs);
    template< class M>
    static double boundary( const size_t i0, const size_t j0, const M& lhs, const M& rhs);
    template< int s0, int s1, class M>
    static double edge( const size_t i0, const size_t j0, const M& lhs, const M& rhs, const unsigned = 0);
    template< int s0, int s1, class M>
    static double corner( const size_t i0, const size_t j0, const M& lhs, const M& rhs);
    
    /*! @brief Implements the arakawa scheme for various BC
     *
     * The 2D Jacobian is defined as the Poisson bracket:
     \f[ 
     j(x,y) := \{l(x,y), r(x,y)\} = \partial_x l \partial_y r - \partial_y l \partial_x r 
     \f] 
     */
    class Arakawa
    {
      private:
        const double c;
      public:
        /*! @brief constructor
         *
         * @param h the physical grid constant
         */
        Arakawa( const double h): c(1.0/(12.0*h*h)){}
        /*! @brief compute Poisson bracket for double periodic BC
         *
         * @param lhs left hand side
         * @param rhs right hand side
         * @param jac the jacobian
         */
        template<class M>
        void per_per(const M& lhs, const M& rhs, M& jac);
        /*! @brief compute Poisson bracket for horizontal periodic and vertical Dirichlet BC
         *
         * @param lhs left hand side
         * @param rhs right hand side
         * @param jac the jacobian
         */
        template<class M>
        void per_dir(const M& lhs, const M& rhs, M& jac);
        /*! @brief compute Poisson bracket for horizontal Dirichlet and vertical periodic BC
         * 
         * @param lhs left hand side
         * @param rhs right hand side
         * @param jac the jacobian
         */
        template<class M>
        void dir_per(const M& lhs, const M& rhs, M& jac);
        /*! @brief compute Poisson bracket for double Dirichlet BC
         *
         * @param lhs left hand side
         * @param rhs right hand side
         * @param jac the jacobian
         */
        template<class M>
        void dir_dir(const M& lhs, const M& rhs, M& jac);
    };
    
    
    template< class M>
    void Arakawa::per_per( const M& lhs, 
                            const M& rhs, 
                            M& jac)
    {
        const size_t rows = jac.rows(), cols = jac.cols();
    
        for( size_t j0 = 0; j0 < cols; j0++)
            jac(0,j0)       = c*boundary( 0, j0, lhs, rhs);
        for( size_t i0 = 1; i0 < rows-1; i0++)
        {
            jac(i0,0)       = c*boundary( i0, 0, lhs, rhs);
            for( size_t j0 = 1; j0 < cols-1; j0++)
                jac(i0,j0)  = c*interior( i0, j0, lhs, rhs);
            jac(i0,cols-1)  = c*boundary( i0, cols-1, lhs, rhs);
        }
        for( size_t j0 = 0; j0 < cols; j0++)
            jac(rows-1,j0)  = c*boundary( rows-1, j0, lhs, rhs);
    }
    
    template<class M>
    void Arakawa::per_dir(  const M& lhs, 
                            const M& rhs, 
                            M& jac)
    {
        const size_t rows = jac.rows(), cols = jac.cols();
    
        for( size_t j0 = 0; j0 < cols; j0++)
            jac(0,j0)   = c*edge<1,0>( 0, j0, lhs, rhs);
        for( size_t i0 = 1; i0 < rows-1; i0++)
        {
            jac(i0,0)       = c*boundary( i0, 0, lhs, rhs);
            for( size_t j0 = 1; j0 < cols-1; j0++)
                jac(i0,j0)  = c*interior( i0, j0, lhs, rhs);
            jac(i0,cols-1)  = c*boundary( i0, cols-1, lhs, rhs);
        }
        for( size_t j0 = 0; j0 < cols; j0++)
            jac(rows -1,j0) = c*edge<-1,0>( rows-1, j0, lhs, rhs);
    }
    
    template<class M>
    void Arakawa::dir_per( const M& lhs, 
                            const M& rhs, 
                            M& jac)
    {
        const size_t rows = jac.rows(), cols = jac.cols();
        jac(0,0)        = c*edge<0, 1>( 0,0,lhs, rhs);
        for( size_t j0 = 1; j0 < cols-1; j0++)
            jac(0,j0)   = c*boundary( 0, j0, lhs, rhs);
        jac(0,cols-1)   = c*edge<0,-1>( 0, cols-1, lhs, rhs);
        for( size_t i0 = 1; i0 < rows-1; i0++)
        {
            jac(i0,0)       = c*edge<0, 1>( i0, 0, lhs, rhs);
            for( size_t j0 = 1; j0 < cols-1; j0++)
                jac(i0,j0)  = c*interior( i0, j0, lhs, rhs);
            jac(i0,cols-1)  = c*edge<0,-1>( i0, cols-1, lhs, rhs);
        }
        jac(rows-1,0)       = c*edge<0,1>( rows-1,0,lhs, rhs);
        for( size_t j0 = 1; j0 < cols-1; j0++)
            jac(rows -1,j0) = c*boundary( rows-1, j0, lhs, rhs);
        jac(rows-1,cols-1)  = c*edge<0,-1>( rows-1, cols-1, lhs, rhs);
    }
    
    template< class M>
    void Arakawa::dir_dir(  const M& lhs, 
                            const M& rhs, 
                            M& jac)
    {
        const size_t rows = jac.rows(), cols = jac.cols();
    
        jac(0,0)        = c*corner<1, 1>( 0, 0, lhs, rhs);
        for( size_t j0 = 1; j0 < cols-1; j0++)
            jac(0,j0)   = c*  edge<1, 0>( 0, j0, lhs, rhs);
        jac(0,cols-1)   = c*corner<1,-1>( 0, cols-1, lhs, rhs);
        for( size_t i0 = 1; i0 < rows-1; i0++)
        {
            jac(i0,0)      = c*  edge< 0, 1>( i0, 0, lhs, rhs);
            for( size_t j0 = 1; j0 < cols-1; j0++)
                jac(i0,j0) = c*interior( i0, j0, lhs, rhs);
            jac(i0,cols-1) = c*  edge< 0,-1>( i0, cols-1, lhs, rhs);
        }
        jac(rows-1,0)        = c*corner<-1,+1>( rows-1, 0, lhs, rhs);
        for( size_t j0 = 1; j0 < cols-1; j0++)
            jac(rows-1,j0)   = c*  edge<-1, 0>( rows-1, j0, lhs, rhs);
        jac(rows-1,cols-1)   = c*corner<-1,-1>( rows-1, cols-1, lhs, rhs);
    }
    
    /******************Access pattern of interior************************
     * xo.   
     * o .    andere Ecken analog (4 mal 2 Mult)
     * ...
     *
     * oo.
     * x .    andere Teile analog ( 4 mal 4 Mult)
     * oo.
     */
    /*! @brief computes an interior point in the Arakawa scheme
     *
     *  @tparam M M class that has to provide m(i, j) access, a rows() and a cols() method.
     *      (type is normally inferred by the compiler)
     *  @param i0 row index of the interior point
     *  @param j0 col index of the interior point
     *  @param lhs left hand side 
     *  @param rhs right hand side 
     *  @return the unnormalized value of the Arakawa bracket
     */
    template< class M>
    double interior( const size_t i0, const size_t j0, const M& lhs, const M& rhs) 
    {
        double jacob;
        const size_t ip = i0 + 1;
        const size_t jp = j0 + 1;
        const size_t im = i0 - 1;
        const size_t jm = j0 - 1;
        jacob  = rhs(i0,jm) * ( lhs(ip,j0) -lhs(im,j0) -lhs(im,jm) +lhs(ip,jm) );
        jacob += rhs(i0,jp) * (-lhs(ip,j0) +lhs(im,j0) -lhs(ip,jp) +lhs(im,jp) );
        jacob += rhs(ip,j0) * ( lhs(i0,jp) -lhs(i0,jm) +lhs(ip,jp) -lhs(ip,jm) );
        jacob += rhs(im,j0) * (-lhs(i0,jp) +lhs(i0,jm) +lhs(im,jm) -lhs(im,jp) );
        jacob += rhs(ip,jm) * ( lhs(ip,j0) -lhs(i0,jm) );
        jacob += rhs(ip,jp) * ( lhs(i0,jp) -lhs(ip,j0) );
        jacob += rhs(im,jm) * ( lhs(i0,jm) -lhs(im,j0) );
        jacob += rhs(im,jp) * ( lhs(im,j0) -lhs(i0,jp) );
        return jacob;
    }
    
    /*! @brief calculates a boundary point in the Arakawa scheme
     *
     *  It assumes periodic BC on the edges!
     *  @tparam M M class that has to provide m(i, j) access, a rows() and a cols() method.
     *      (type is normally inferred by the compiler)
     *  @param i0 row index of the edge point
     *  @param j0 col index of the edge point
     *  @param lhs left hand side M
     *  @param rhs right hand side M
     *  @return the unnormalized value of the Arakawa bracket
     */
    template< class M>
    double boundary( const size_t i0, const size_t j0, const M& lhs, const M& rhs) 
    {
        static QuadMat<double, 3> l, r;
        const size_t rows = lhs.rows(), cols = lhs.cols();
        const size_t ip = (i0==(rows-1)) ? 0 : i0+1;
        const size_t im = (i0==0) ? (rows-1) : i0-1;
        const size_t jp = (j0==(cols-1)) ? 0 : j0+1;
        const size_t jm = (j0==0) ? (cols-1) : j0-1;

        //assignment
        for( size_t i = 0; i < 3; i++)
            for( size_t j = 0; j < 3; j++)
            {
                l(i,j) = lhs( (i == 1)? i0 : ((i==0)?im:ip), (j == 1)? j0 : ((j==0)?jm:jp));
                r(i,j) = rhs( (i == 1)? i0 : ((i==0)?im:ip), (j == 1)? j0 : ((j==0)?jm:jp));
            }

        return interior( 1, 1, l, r);
    }
    
    
    /*! @brief computes the real modulus of integer number
     *
     * i.e. (-2)mod 10 is 8
     * @param i integer number
     * @param m the module
     * @return i mod m
     */
    const size_t mod( const int i, const size_t m)
    {
        return i>=0 ? i%m : mod( i + m, m);
    }

    void mod( Vector<int, 2>& v, const size_t rows, const size_t cols)
    {
        v[0] = mod(v[0], rows);
        v[1] = mod(v[1], cols);
    }

    // bessere idee: lege statische 3x3 Matrix an, kopiere Werte rein und übergib diese an interior
    // da dies nur die Randwerte betrifft sollte der overhaed nicht groß sein
    /*! @brief calculates and edge point in the Arakawa scheme
     *
     * The paramters s0 and s1 indicate the interior of the matrix.
     * Possible values are (0,1), (1,0), (0,-1) and (-1, 0).
     * This routine can be used to calculate a corner point in the
     * case of mixed periodic and dirichlet BC. 
     *
     *  @tparam s0 row index of the vector that points inwards ( 1 or 0 or -1)
     *  @tparam s1 col index of the vector that points inwards ( 1 or 0 or -1)
     *  @tparam M M class that has to provide m(i, j) access, a rows() and a cols() method.
     *      (type is normally inferred by the compiler)
     *  @param i0 row index of the edge point
     *  @param j0 col index of the edge point
     *  @param lhs left hand side M
     *  @param rhs right hand side M
     *  @param bc0 boundary condition for the ghost cells 
     *  @return the unnormalized value of the Arakawa bracket
     */
    template <int s0, int s1, class M> //indicates where the interior is
    double edge( const size_t i0, const size_t j0, const M& lhs, const M& rhs, const unsigned bc0 = 0) 
    {
        static QuadMat<double, 3> l, r;
        const size_t rows = lhs.rows(), cols = lhs.cols();
        static QuadMat< Vector<int,2>, 3> indexMap;
        static Vector< Vector<int,2>, 3> ghostIdx;
        static Vector< Vector<int,2>, 6> materialIdx;
        l.zero(), r.zero();
        Vector<int, 2> idx, s, ex,ey;
        s[0] = s0, s[1] = s1;
        idx[0] = i0, idx[1] = j0;

        //ghostIdx stores the indices of l and r that are ghost cells
        ghostIdx[1][0] = 1-s0;
        ghostIdx[1][1] = 1-s1;
        ghostIdx[0] = ghostIdx[1] + s.perp();
        ghostIdx[2] = ghostIdx[1] - s.perp();
        //std::cout << ghostIdx << std::endl;
        //matrialIdx stores the indices of l and r that are not ghost cells
        materialIdx[0] = ghostIdx[0] + s;
        materialIdx[1] = ghostIdx[1] + s;
        materialIdx[2] = ghostIdx[2] + s;
        materialIdx[3] = ghostIdx[0] + 2*s;
        materialIdx[4] = ghostIdx[1] + 2*s;
        materialIdx[5] = ghostIdx[2] + 2*s;
        //std::cout << materialIdx << std::endl;
        ex[0] = 0, ex[1] = 1;
        ey = ex.perp();
        //indexMap maps the indices of l and r to indices of lhs and rhs
        indexMap(1,1) = idx;
        indexMap(1,2) = idx + ex;
        indexMap(1,0) = idx - ex;
        indexMap(0,1) = idx + ey;
        indexMap(0,2) = indexMap(0,1) + ex;
        indexMap(0,0) = indexMap(0,1) - ex;
        indexMap(2,1) = idx - ey;
        indexMap(2,0) = indexMap(2,1) - ex;
        indexMap(2,2) = indexMap(2,1) + ex;
        //std::cout << indexMap << std::endl;
        
        //absolute position
        for( size_t i = 0; i < 3; i++)
            for( size_t j = 0; j < 3; j++)
                mod(indexMap(i,j), rows, cols);
        Vector< int, 2> temp;
        //assign material values
        for( size_t i = 0; i < 6; i++)
        {
            temp = materialIdx[i];
            l(temp[0], temp[1]) = lhs(indexMap(temp[0], temp[1])[0], indexMap(temp[0], temp[1])[1]);
            r(temp[0], temp[1]) = rhs(indexMap(temp[0], temp[1])[0], indexMap(temp[0], temp[1])[1]);
        }
        //assign ghost values
        switch(bc0)
        {
            case(-1):
                for( size_t i = 0; i < 3; i++)
                {
                    l(ghostIdx[i][0],ghostIdx[i][1]) = -l((ghostIdx[i]+s)[0], (ghostIdx[i]+s)[1]);
                    r(ghostIdx[i][0],ghostIdx[i][1]) = -r((ghostIdx[i]+s)[0], (ghostIdx[i]+s)[1]);
                }
            case(0):
                for( size_t i = 0; i < 3; i++)
                {
                    l(ghostIdx[i][0], ghostIdx[i][1]) = 0;
                    r(ghostIdx[i][0], ghostIdx[i][1]) = 0;
                }
                //std::cout << l << std::endl;
                break;
            case(1):
                for( size_t i = 0; i < 3; i++)
                {
                    l(ghostIdx[i][0],ghostIdx[i][1]) = +l((ghostIdx[i]+s)[0], (ghostIdx[i]+s)[1]);
                    r(ghostIdx[i][0],ghostIdx[i][1]) = +r((ghostIdx[i]+s)[0], (ghostIdx[i]+s)[1]);
                }
                break;
            case(2):
                for( size_t i = 0; i < 3; i++)
                {
                    l(ghostIdx[i][0],ghostIdx[i][1]) = +l((ghostIdx[i]+2*s)[0], (ghostIdx[i]+2*s)[1]);
                    r(ghostIdx[i][0],ghostIdx[i][1]) = +r((ghostIdx[i]+2*s)[0], (ghostIdx[i]+2*s)[1]);
                }
                break;
            default:
                throw Message( "Unknown boundary condition", ping);
        }
        
    
        /*
        jacob  = rhs_0m * ( lhs_10  +lhs_1m );
        jacob += rhs_01 * ( -lhs_10 -lhs_11 );
        jacob += rhs_10 * ( lhs_01 -lhs_0m +lhs_11 -lhs_1m );
        jacob += rhs_1m * ( lhs_10 -lhs_0m );
        jacob += rhs_11 * ( lhs_01 -lhs_10 );
        */
    
        //compute arakawa bracket
        return interior(1,1,l,r);
    }
    
    
    /*! @brief calculates a corner in the Arakawa scheme for homogeneous Dirichlet BC
     *
     * The template parameters s0 and s1 are used to indicate the corner.
     *
     *  @tparam s0 row index of the vector that points inwards ( 1 or -1)
     *  @tparam s1 col index of the vector that points inwards ( 1 or -1)
     *  @tparam M M class that has to provide m(i, j) access (normally inferred by the compiler)
     *  @param i0 row index of the corner (technically not necessary but kept for consistency)
     *  @param j0 col index of the corner 
     *  @param lhs left hand side M
     *  @param rhs right hand side M
     *  @return the unnormalized value of the Arakawa bracket
     */
    template <int s0, int s1, class M> //indicates the interior
    double corner( const size_t i0, const size_t j0, const M& lhs, const M& rhs) 
    {
        double jacob;
        const double& rhs_11 = rhs(i0 + s0,j0 + s1), &lhs_11 = lhs(i0 + s0,j0 + s1);
        const double& rhs_01 = rhs(i0 + (s0 - s1)/2, j0 + (s0 + s1)/2), &lhs_01 = lhs(i0 + (s0 - s1)/2, j0 + (s0 + s1)/2);
        const double& rhs_10 = rhs(i0 + (s0 + s1)/2, j0 + (s1 - s0)/2), &lhs_10 = lhs(i0 + (s0 + s1)/2, j0 + (s1 - s0)/2);
        jacob = rhs_01 * ( -lhs_10 -lhs_11 );
        jacob += rhs_10 * ( lhs_01 +lhs_11 );
        jacob += rhs_11 * ( lhs_01 -lhs_10 );
       /*  000  000  
        *  0 x  x 0  etc.
        *  0xx  xx0
        */
        return jacob;
    }


}
#endif// _ARAKAWA_





