#ifndef _TL_TIMER_
#define _TL_TIMER_

#include <sys/time.h>


class Timer
{
    timeval start;
    timeval stop;
    public:
    void tic(){ gettimeofday( &start, NULL);}
    void toc(){ gettimeofday( &stop, NULL);}
    double diff(){ return stop.tv_sec - start.tv_sec + 1e-6*(stop.tv_usec - start.tv_usec);}
};

#endif //_TL_TIMER_
