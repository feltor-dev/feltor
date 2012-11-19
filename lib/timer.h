#ifndef _TL_TIMER_
#define _TL_TIMER_

#include <sys/time.h>


namespace toefl{
    /*! @brief Very simple tool for performance measuring*/
    class Timer
    {
        timeval start;
        timeval stop;
        public:
        /*! @brief Start timer using gettimeofday */
        void tic(){ gettimeofday( &start, NULL);}
        /*! @brief Stop timer using gettimeofday */
        void toc(){ gettimeofday( &stop, NULL);}
        /*! \brief Return time elapsed between tic and toc
         *
         * \return Time in seconds between calls of tic and toc*/
        double diff(){ return (stop.tv_sec - start.tv_sec) + 1e-6*(stop.tv_usec - start.tv_usec);}
    };
}

#endif //_TL_TIMER_
