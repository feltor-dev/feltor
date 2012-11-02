/*!
 * @file
 * @brief Implementation of the arakawa scheme
 * @author Matthias Wiesenberger
 * @email Matthias.Wiesenberger@uibk.ac.at
 * 
 */
#ifndef _ARAKAWA_
#define _ARAKAWA_

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
    static double edge( const size_t i0, const size_t j0, const M& lhs, const M& rhs);
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
        double jacob;
        const size_t rows = lhs.rows(), cols = lhs.cols();
        const size_t ip = (i0==(rows-1)) ? 0 : i0+1;
        const size_t im = (i0==0) ? (rows-1) : i0-1;
        const size_t jp = (j0==(cols-1)) ? 0 : j0+1;
        const size_t jm = (j0==0) ? (cols-1) : j0-1;
     //  xxx
     //  x x (access pattern)
     //  xxx
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
     *  @return the unnormalized value of the Arakawa bracket
     */
    template <int s0, int s1, class M> //indicates where the interior is
    double edge( const size_t i0, const size_t j0, const M& lhs, const M& rhs) 
    {
        const size_t rows = lhs.rows(), cols = lhs.cols();
        size_t s[2], s_01[2], s_10[2], s_0m[2], s_11[2], s_1m[2];
        double jacob;
    
        s[0] = i0, s[1] = j0;
        //Vektoren in x und y Richtung
        s_10[0] = s0; s_01[0] = -s1;
        s_10[1] = s1; s_01[1] = s0;
        s_0m[0] = -s_01[0]; 
        s_0m[1] = -s_01[1]; 
        s_11[0] = s_10[0] + s_01[0];
        s_11[1] = s_10[1] + s_01[1];
    
        s_1m[0] = s_10[0] + s_0m[0];
        s_1m[1] = s_10[1] + s_0m[1];
    
        s_10[0] += s[0], s_01[0] += s[0], s_0m[0] += s[0], s_11[0]+=s[0], s_1m[0] += s[0];
        s_10[1] += s[1], s_01[1] += s[1], s_0m[1] += s[1], s_11[1]+=s[1], s_1m[1] += s[1];
        
     //  0xx  000  
     //  0 x  x x  etc.
     //  0xx  xxx
        const double& rhs_10 = rhs( mod(s_10[0], rows), mod( s_10[1], cols)), &lhs_10 = lhs( mod(s_10[0], rows), mod( s_10[1], cols));
        const double& rhs_0m = rhs( mod(s_0m[0], rows), mod( s_0m[1], cols)), &lhs_0m = lhs( mod(s_0m[0], rows), mod( s_0m[1], cols));
        const double& rhs_01 = rhs( mod(s_01[0], rows), mod( s_01[1], cols)), &lhs_01 = lhs( mod(s_01[0], rows), mod( s_01[1], cols));
        const double& rhs_11 = rhs( mod(s_11[0], rows), mod( s_11[1], cols)), &lhs_11 = lhs( mod(s_11[0], rows), mod( s_11[1], cols));
        const double& rhs_1m = rhs( mod(s_1m[0], rows), mod( s_1m[1], cols)), &lhs_1m = lhs( mod(s_1m[0], rows), mod( s_1m[1], cols));
    
        jacob  = rhs_0m * ( lhs_10  +lhs_1m );
        jacob += rhs_01 * ( -lhs_10 -lhs_11 );
        jacob += rhs_10 * ( lhs_01 -lhs_0m +lhs_11 -lhs_1m );
        jacob += rhs_1m * ( lhs_10 -lhs_0m );
        jacob += rhs_11 * ( lhs_01 -lhs_10 );
    
        return jacob;
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





