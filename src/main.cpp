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

void load_txt(const std::string& path, Eigen::MatrixXd& Parameter)
{
	ifstream in;
	std::string line;
	int row = Parameter.rows(), col = Parameter.cols();
	Eigen::MatrixXd m(row, col);

	in.open(path.c_str());	
	int i = 0, j = 0;
	while (std::getline(in, line)) {
    for (int k = 0; k < line.size(); k++)// 如果输入中有‘，’，将其换成空格
         if (line[k] == ',')
             line[k] =' ';
		std::stringstream ss(line);
		double d;
		while (ss >> d) {
			if (j == col) {
				++i;
				j = 0;
			}
			m(i, j) = d;
			++j;
		}
	}
  Parameter = m;
};



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


Input_History<<-40.543, -16.067, 28.605, 23.207,
              -40.079, -16.039,  28.433,  23.018,
              -39.615, -16.01,   28.26,  22.83, 
              -39.15,  -15.981,  28.087,  22.642,
              -38.685, -15.951,  27.911,  22.455;
Input_Speed <<4.66390062, 2.57588703;


/*
MDN mdn;
mdn.get_likelihood(Input_Speed, Input_History, likelihood);
cout << "likelihood: " << likelihood ;
*/




string path_input = "/home/letian/Benchmark(copy)/MDN/input_label_raw.txt";
MatrixXd input = MatrixXd::Zero(1016*5,4) ;
load_txt(path_input, input);
cout<<"input.cols:"<<input.cols()<<endl;
cout<<"input_rows:"<<input.rows()<<endl;
cout<<input.block(0,0,5,4)* 10e-4<<endl;

string path_output = "/home/letian/Benchmark(copy)/MDN/output_label_raw.txt";
MatrixXd output = MatrixXd::Zero(1016,2) ;
load_txt(path_output, output);
cout<<"output.cols:"<<output.cols()<<endl;
cout<<"output_rows:"<<output.rows()<<endl;
cout<<output.block(0,0,1,2) * 10e-13<<endl;

std::vector<double> LIKELIHOOD;
for (int k = 0; k < 1015; k++)// 如果输入中有‘，’，将其换成空格
{
  
  //cout<<k;
  Input_History = input.block(k*5, 0, 5, 4) / 10e+2;  
  Input_Speed = output.block(k, 0, 1, 2) / 10e+11;
  likelihood = 1;
  MDN mdn;
  mdn.get_likelihood(Input_Speed, Input_History, likelihood);
  //cout << "likelihood: " << likelihood ;
  LIKELIHOOD.push_back(likelihood);
  cout<<LIKELIHOOD[k]<<endl;
};


return 0;
}
