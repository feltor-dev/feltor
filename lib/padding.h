#ifndef _PADDING_
#define _PADDING_

namespace toefl{

    enum Padding{ TL_NONE, TL_DFT_1D, TL_DFT_2D, TL_DRT_DFT, TL_DFT_DFT};

    /*! @brief template traits class for the efficient implementation of
     * the access operators in the matrix class.
     *
     * These functions are also used in copy and construction operators.
     */
    template <enum Padding P> //i.e. TL_NONE
    struct TotalNumberOf
    {
        static inline size_t cols( const size_t m){return m;}
        static inline size_t elements( const size_t n, const size_t m){return n*m;}
    };
    
    template <>
    struct TotalNumberOf<TL_DFT_1D>
    {
        static inline size_t cols( const size_t m){ return m - m%2 + 2;}
        static inline size_t elements( const size_t n, const size_t m){return n*(m - m%2 + 2);}
    };
    
    template <>
    struct TotalNumberOf<TL_DFT_2D>
    {
        static inline size_t cols( const size_t m){ return m;}
        static inline size_t elements( const size_t n, const size_t m){return n*(m - m%2 + 2);}
    };
    template <>
    struct TotalNumberOf<TL_DRT_DFT>
    {
        static inline size_t cols( const size_t m){ return m;}
        static inline size_t elements( const size_t n, const size_t m){return m*(n - n%2 + 2);}
    };
    template <>
    struct TotalNumberOf<TL_DFT_DFT>
    {
        static inline size_t cols( const size_t m){ return m - m%2 + 2;}
        static inline size_t elements( const size_t n, const size_t m){return n*(m - m%2 + 2);}
    };



}



#endif //_PADDING_
