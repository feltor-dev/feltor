#error Documentation only
/*! @namespace toefl
 * @brief This is the namespace for all functions and 
 * classes defined and used by the toefl solvers.
 *
 *
 * @mainpage
 * This is the toefl library.
 * @par A word on indexing
 *
 * In most image container formats (e.g. PGM) and in windowing systems (e.g. GLFW) 
 * the first element of a vector (double* field; field[0]) corresponds to the upper left corner of an image. 
 * Subsequent elements progress left-to-right in the highest row of and then in successively lower rows of the image. 
 * The final element corresponds to the lower right corner of the image. (cf. http://de.wikipedia.org/wiki/Portable_Graymap)
 * This is the way you read text. 
 * In C-Convention the 2nd index of a 2-dimensional field is the continuous one.
 * That means that the X-INDEX IS THE SECOND ONE! 
 * For cases that defer to this we use the suffix _T (for transposed) in variable names, 
 * i.e. the vector double* field_T would store the y-direction continuously and the x-index would be the first one.
 *
 * Note that there are notable exceptions to this rule e.g. the OpenGl texture convention places the origin in the 
 * bottom left corner, but also interprets x-values continuously.  (mathematical coordinate system)
 *
 */
