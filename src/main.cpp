#include <iostream>
#include <Dense>
#include "MDN.h"
#include <vector>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
using namespace std;
using namespace Eigen;


int main()
{
/* To Modify:
  1. The absolute direcotry of the parameter and include.
  2. The Input(5,4), 5 is the timestep.
*/

MatrixXd Input_History = MatrixXd::Ones(5,4);
MatrixXd Input_Speed = MatrixXd::Ones(1,2);
double likelihood = 1;

Input_Speed << 7,7;
Input_History<<10.079, -15.062,  24.679, -15.058,
               10.718, -15.092,  25.268, -15.15, 
               11.363, -15.115,  25.863, -15.244,
               12.015, -15.128,  26.462, -15.342,
               12.672, -15.132,  27.067, -15.444;
  
MDN mdn;
mdn.get_likelihood(Input_Speed, Input_History, likelihood);
cout << "likelihood: " << likelihood ;

return 0;
}
