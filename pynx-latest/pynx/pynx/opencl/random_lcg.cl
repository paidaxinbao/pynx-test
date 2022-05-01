/* -*- coding: utf-8 -*-
*
* PyNX - Python tools for Nano-structures Crystallography
*   (c) 2019-present : ESRF-European Synchrotron Radiation Facility
*       authors:
*         Vincent Favre-Nicolin, favre@esrf.fr
*/

/// Pseudo-random number linear congruential generator, returning a random number in [0 ; 2**31[
inline unsigned long random_lcg(const unsigned long x)
{
  return (1103515245ul * x + 12345ul ) % 2147483648ul;
}
