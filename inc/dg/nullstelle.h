/*! @file 
 * @brief vereint verschiedene Methoden zur Nullstellensuche
 * @author Matthias Wiesenberger
 * @verbatim
 * Nullstellensuche 12.4.2010
  @endverbatim 
 */
#ifndef _NULLSTELLE_
#define _NULLSTELLE_

#include <exception>
#include <math.h>
namespace dg{

/**
 * @brief Class you might want to throw in case of a non convergence
 *
 * @ingroup utilities
 */
struct Ooops : public std::exception
{

    /**
     * @brief Construct from error string
     *
     * @param c error string
     */
    Ooops( const char * c): c_( c) {}
    /**
     * @brief What string
     *
     * @return error string 
     */
    char const* what() const throw(){ return c_;}
  private:
    const char* c_;
};
/*! @brief Fehlerklasse die Grenzen für 1D Nullstellensuche sichert
 */
class KeineNST_1D: public std::exception
{
  private:
    double x1, x2;
  public:
    /*! @brief Konstruktor
     *
     * \param x_min linke Grenze
     * \param x_max rechte Grenze
     * \param file Fehler
     * \param line Fehler
     */
    KeineNST_1D(double x_min, double x_max): x1(x_min), x2(x_max){}
    /*! @brief Anzeigemethodea auf std::cerr
     *
     * meldet %Fehler und gibt linke und rechte Grenze aus
     */
    void anzeigen() const
    {
      std::cerr << "Zwischen " <<x1 << " und "<<x2<<" liegt keine NST!\n";
    }
    char const* what() const throw(){ return "Failed to find root!";}
};

/*! @defgroup Nullstellensuchroutinen Nullstellensuchroutinen
 * @{
 */
/*! @brief Bestimmung der Nullstelle einer 1-D Funktion bzw. Funktors in vorgegebenen Grenzen
 *
 * Es wird davon ausgegangen, dass ein Vorzeichenwechsel bei der NST vorliegt
 * Funktion hüpft immer näher an die NST heran, indem es am VZ prüft ob es sich links oder rechts der NST befindet
 * \param funktion Funktion oder Funktor
 * \param x_min vorzugebene linkseitige Grenze, enthält nach Ausführung die neue linke Grenze
 * \param x_max vorzugebene rechtsseitige Grenze, enthält nach Ausführung die neue rechte Grenze
 * \param aufloesung	Fehlertoleranz für die NST 
 * \return Zahl der benötigten Schritte bis zum Erreichen der Auflösung
 * \throw KeineNST_1D falls zwischen den angegebenen Grenzen keine Nullstelle liegt
 * \throw Fehler falls mit 50 Schritten die Auflösung nicht erreicht werden kann
 *
 * \code nullstelle_1D(funk, x_min, x_max, aufloesung); \endcode
 * \note Falls die Nullstelle zufälligerweise exakt gefunden wird, so ist x_min = x_max!
 */
template <typename UnaryOp>           
int bisection1d (UnaryOp& funktion, double& x_min, double& x_max, const double aufloesung) 
{
    double  mitte;
    double wert_oben, wert_mitte, wert_unten;
    wert_oben=funktion(x_max);
    wert_unten=funktion(x_min);

    if(wert_oben*wert_unten>=0) 
        throw KeineNST_1D(x_min, x_max);
    
    int j_max = 50;
    for(int j=0; j<j_max; j++)
    {
        wert_mitte = funktion( mitte=(x_min+x_max)/2.0 );
        if(wert_mitte==0) 			        {x_min=x_max=mitte; return j+3;}
        else if(wert_mitte*wert_oben>0) 	x_max = mitte;
        else 				                x_min = mitte;
        if((x_max-x_min)<aufloesung)        return j+3; 
    }
    throw Ooops("Zu viele Schritte bei Nullstellensuche! evtl. j_max aendern");
}

      
//@}
}//namespace dg
#endif //_NULLSTELLE_  
  
  
  
