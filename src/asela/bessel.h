// ----------------------------------------------------------------------
double bessj1(double x) 
//Returns the Bessel function J1(x) for any real x. 
{   double ax,z; 
    double xx,y,ans,ans1,ans2; 
    if ((ax=fabs(x)) < 8.0) 
    { 
	y=x*x; 
	ans1 = x*(72362614232.0 + y*(-7895059235.0 + 
		   y*(242396853.1 + y*(-2972611.439 + 
		     y*(15704.48260+y*(-30.16036606))))));  
	ans2 = 144725228442.0 + y*(2300535178.0 + 
		   y*(18583304.74 + y*(99447.43394 + 
		    y*(376.9991397+y*1.0)))); 
	ans=ans1/ans2; 
    } 
    else 
    { 
	z=8.0/ax; 
	y=z*z; 
	xx=ax-2.356194491; 
	ans1 = 1.0+y*(0.183105e-2+y*(-0.3516396496e-4 +
		   y*(0.2457520174e-5+y*(-0.240337019e-6)))); 
	ans2 = 0.04687499995+y*(-0.2002690873e-3 +
                   y*(0.8449199096e-5+y*(-0.88228987e-6 +y*0.105787412e-6))); 
	ans = sqrt(0.636619772/ax)*(cos(xx)*ans1-z*sin(xx)*ans2); 
	if (x < 0.0) ans = -ans; 
    } 
    return ans; 
}   

// ----------------------------------------------------------------------

double bessi1(double x) 
//Returns the modified Bessel function I1(x) for any real x. 
{   double ax,ans; 
    double y; 
    if ((ax=fabs(x)) < 3.75) 
    {
	y=x/3.75; 
	y*=y; 
	ans = ax*(0.5+y*(0.87890594+y*(0.51498869+y*(0.15084934 +
               y*(0.2658733e-1+y*(0.301532e-2+y*0.32411e-3)))))); 
    } 
    else 
    { 
	y=3.75/ax; 
	ans = 0.2282967e-1+y*(-0.2895312e-1+y*(0.1787654e-1 -
              y*0.420059e-2)); ans=0.39894228+y*(-0.3988024e-1+
              y*(-0.362018e-2 +y*(0.163801e-2+y*(-0.1031555e-1+y*ans)))); 
	ans *= (exp(ax)/sqrt(ax)); 
    } 
    return x < 0.0 ? -ans : ans; 
}

// ----------------------------------------------------------------------
double bessk1(double x)
// Returns the modified Bessel function K1(x) for positive real x.
{ 
    double bessi1(double x); 
    double y,ans;
    if (x <= 2.0) 
    {
	y=x*x/4.0; 
	ans = (log(x/2.0)*bessi1(x))+(1.0/x)*(1.0+y*(0.15443144 +
               y*(-0.67278579+y*(-0.18156897+y*(-0.1919402e-1 +
               y*(-0.110404e-2+y*(-0.4686e-4))))))); 
    } 
    else 
    { 
	y=2.0/x; 
	ans = (exp(-x)/sqrt(x))*(1.25331414+y*(0.23498619 +
              y*(-0.3655620e-1+y*(0.1504268e-1+y*(-0.780353e-2 +
              y*(0.325614e-2+y*(-0.68245e-3))))))); 
    } 
    return ans; 
}

