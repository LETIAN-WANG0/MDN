#include <iostream>
#include </home/letian/Benchmark/MDN/Eigen/Dense>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
using namespace std;
using namespace Eigen;
#include <math.h>
#include "MDN.h"

MDN::MDN()
{
    WG = MatrixXd::Ones(12,16);
    bG = MatrixXd::Ones(16,1);
    Wc = MatrixXd::Ones(12,8);
    bc = MatrixXd::Ones(8,1);
    W0 = MatrixXd::Ones(8,16);
    W1 = MatrixXd::Ones(16,16);
    W2 = MatrixXd::Ones(16,6);
    b0 = MatrixXd::Ones(16,1);
    b1 = MatrixXd::Ones(16,1);
    b2 = MatrixXd::Ones(6,1);

    path_W0 = "/home/letian/Benchmark/MDN/Parameter/weights_h0.txt";
    path_b0 = "/home/letian/Benchmark/MDN/Parameter/bias_h0.txt";
    path_W1 = "/home/letian/Benchmark/MDN/Parameter/weights_h1.txt";
    path_b1 = "/home/letian/Benchmark/MDN/Parameter/bias_h1.txt";
    path_W2 = "/home/letian/Benchmark/MDN/Parameter/weights_h2.txt";
    path_b2 = "/home/letian/Benchmark/MDN/Parameter/bias_h2.txt";
    path_Wc = "/home/letian/Benchmark/MDN/Parameter/weights_rnn_candidate.txt";
    path_bc = "/home/letian/Benchmark/MDN/Parameter/bias_rnn_candidate.txt";
    path_WG = "/home/letian/Benchmark/MDN/Parameter/weights_rnn_gate.txt";
    path_bG = "/home/letian/Benchmark/MDN/Parameter/bias_rnn_gate.txt";
}

void MDN::sigmoid(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache)
{
	A = 1 / (1 + (-Z).array().exp());
	assert(A.size() == Z.size());
	cache.push_back(Z);
}

void MDN::relu(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache)
{
	A = Z.array().max(0);
	assert(A.rows() == Z.rows());
	assert(A.cols() == Z.cols());
	cache.push_back(Z);
}

void MDN::tanh(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache)
{
	A = ((Z).array().exp() - (-Z).array().exp()) / ((Z).array().exp() + (-Z).array().exp());
	assert(A.size() == Z.size());
	cache.push_back(Z);
}

void MDN::Soft_max(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A, std::vector<Eigen::MatrixXd>& cache)
{
	A = Z.array().exp() / Z.array().exp().sum();
	cache.push_back(Z);
}

void MDN::linear_forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& W, const Eigen::VectorXd& b, Eigen::MatrixXd& Z, std::vector<Eigen::MatrixXd>& cache)
{
	Z = ((A * W).transpose() + b).transpose();
	assert(Z.rows() == A.rows());
	assert(Z.cols() == W.cols());
	cache.push_back(A);
	cache.push_back(W);
	cache.push_back(b); // VectorXd隐式转换为MatrixXd
}

void MDN::linear_activation_forward(const Eigen::MatrixXd& A_prev, const Eigen::MatrixXd& W, const Eigen::VectorXd& b, const std::string activation, Eigen::MatrixXd& A, std::vector<std::vector<Eigen::MatrixXd> >& cache)
{
	Eigen::MatrixXd Z;
	std::vector<Eigen::MatrixXd> linear_cache;
	std::vector<Eigen::MatrixXd> activation_cache;
	if (activation == "sigmoid") {
		linear_forward(A_prev, W, b, Z, linear_cache);
		sigmoid(Z, A, activation_cache);
	}
	else if (activation == "relu") {
		linear_forward(A_prev, W, b, Z, linear_cache);
		relu(Z, A, activation_cache);
	}
	else if (activation == "tanh") {
		linear_forward(A_prev, W, b, Z, linear_cache);
		tanh(Z, A, activation_cache);
	}

	assert(A.rows() == A_prev.rows());
	assert(A.cols() == W.cols());
	cache.push_back(linear_cache);
	cache.push_back(activation_cache);
}


void MDN::GRU_Cell(const Eigen::MatrixXd& c_prev, Eigen::MatrixXd& x, Eigen::MatrixXd& c, Eigen::MatrixXd& a, 
   const Eigen::MatrixXd& WG, const Eigen::VectorXd& bG,  const Eigen::MatrixXd& Wc,const Eigen::MatrixXd& bc)
{
	Eigen::MatrixXd conc(1, c_prev.cols()+x.cols());
	Eigen::MatrixXd conc_for_candidate(1, c_prev.cols()+x.cols());
	std::vector<std::vector<Eigen::MatrixXd> > cache_u;
	std::vector<std::vector<Eigen::MatrixXd> > cache_r;
	Eigen::MatrixXd ones = MatrixXd::Ones(1, c_prev.cols());
	Eigen::MatrixXd Gate;
	Eigen::MatrixXd Gate_r;
	Eigen::MatrixXd Gate_u;
	Eigen::MatrixXd c_candidate;
	int c_length = c_prev.cols();

	// concatenate
	conc<<x, c_prev;
	// Gates
	linear_activation_forward(conc, WG, bG, "sigmoid", Gate, cache_u);
	Gate_r = Gate.block(0,0,1,c_length);
	Gate_u = Gate.block(0,c_length,1,c_length);
	// conc_for_candidate, transform to array for element-wise multiplying, and transform back to matrix
	conc_for_candidate << x, (Gate_r.array() * c_prev.array()).matrix();
	// candidate	
	linear_activation_forward(conc_for_candidate, Wc, bc, "tanh", c_candidate, cache_r);
	// State and Output
	c = (Gate_u.array() * c_prev.array()).matrix() + ((ones.array() - Gate_u.array()) * c_candidate.array() ).matrix();
	a = c;
}

// x is the input(5,4), with 5 timestep
void MDN::GRU_Foward(Eigen::MatrixXd& x, Eigen::MatrixXd& c, Eigen::MatrixXd& a, 
const Eigen::MatrixXd& WG, const Eigen::MatrixXd& bG, const Eigen::MatrixXd& Wc,const Eigen::MatrixXd& bc)
{
	// Define the variables
	MatrixXd c1(1,8);
    MatrixXd a1(1,8);
	MatrixXd c2(1,8);
    MatrixXd a2(1,8);
	MatrixXd c3(1,8);
    MatrixXd a3(1,8);
	MatrixXd c4(1,8);
    MatrixXd a4(1,8);
	// Initialize the c0 as zero
	
	MatrixXd c0 = MatrixXd::Zero(1,8);
	
	// x.block(a,b,x,y)extract inputs, start from(a,b), size is (x,y)
	MatrixXd x0 = x.block(0,0,1,4);
	MatrixXd x1 = x.block(1,0,1,4);
	MatrixXd x2 = x.block(2,0,1,4);
	MatrixXd x3 = x.block(3,0,1,4);
	MatrixXd x4 = x.block(4,0,1,4);

	// GRU_Forward for 5 timesteps
	GRU_Cell(c0, x0, c1, a1, WG, bG, Wc, bc );
	GRU_Cell(c1, x1, c2, a2, WG, bG, Wc, bc );
	GRU_Cell(c2, x2, c3, a3, WG, bG, Wc, bc );
	GRU_Cell(c3, x3, c4, a4, WG, bG, Wc, bc );
	GRU_Cell(c4, x4, c, a, WG, bG, Wc, bc );
}

void MDN::FC_3(Eigen::MatrixXd& Input, Eigen::MatrixXd& Output, const Eigen::MatrixXd& W0, const Eigen::MatrixXd& W1, const Eigen::MatrixXd& W2,
	const Eigen::VectorXd& b0,  const Eigen::VectorXd& b1,  const Eigen::VectorXd& b2)
{
	Eigen::MatrixXd Output0;
	Eigen::MatrixXd Output1;
	Eigen::MatrixXd Output2;
	std::vector<std::vector<Eigen::MatrixXd> > cache0;
	std::vector<std::vector<Eigen::MatrixXd> > cache1;
	std::vector<Eigen::MatrixXd> cache2;

	linear_activation_forward(Input, W0, b0, "tanh", Output0, cache0);
	linear_forward(Output0, W2, b2, Output2, cache2);
	Output = Output2;
}

void MDN::Post_process(Eigen::MatrixXd& Input, Eigen::MatrixXd& Output)
{
	MatrixXd output(1,6);
	MatrixXd weight = Input.block(0,0,1,1);
	MatrixXd mu = 	  Input.block(0,1,1,2);
	MatrixXd sigma =  Input.block(0,3,1,2);
	MatrixXd cor =    Input.block(0,5,1,1);
	vector<Eigen::MatrixXd> cache1;
	vector<Eigen::MatrixXd> cache2;

	Soft_max(weight, weight, cache1);
	sigma = sigma.array().exp();
	tanh(cor, cor, cache2);
	output << weight,
			  mu,
			  sigma,
			  cor;
	Output = output;
}

// The MDN consists of a GRU networks of 5 timesteps and a FC network of 3 layers, postprocess the output for the final output
void MDN::MDN_Model(Eigen::MatrixXd& Input, Eigen::MatrixXd& Output,
 const Eigen::MatrixXd& WG, const Eigen::VectorXd& bG, const Eigen::MatrixXd& Wc,const Eigen::MatrixXd& bc,
 const Eigen::MatrixXd& W0, const Eigen::MatrixXd& W1, const Eigen::MatrixXd& W2, const Eigen::VectorXd& b0, const Eigen::VectorXd& b1, const Eigen::VectorXd& b2)
{
	Eigen::MatrixXd GRU_c;
    Eigen::MatrixXd GRU_a;
	Eigen::MatrixXd FC_Output;
	GRU_Foward(Input, GRU_c,  GRU_a, WG, bG, Wc, bc);
	FC_3(GRU_a, FC_Output, W0, W1, W2, b0, b1, b2);
	Post_process(FC_Output, Output);
}


void MDN::load_txt(const std::string& path, Eigen::MatrixXd& Parameter)
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

}


void MDN::load_Parameter(
 const std::string& path_WG, Eigen::MatrixXd& WG, const std::string& path_bG, Eigen::MatrixXd& bG,
 const std::string& path_Wc, Eigen::MatrixXd& Wc, const std::string& path_bc, Eigen::MatrixXd& bc,
 const std::string& path_W0, Eigen::MatrixXd& W0, const std::string& path_b0, Eigen::MatrixXd& b0,
 const std::string& path_W1, Eigen::MatrixXd& W1, const std::string& path_b1, Eigen::MatrixXd& b1,
 const std::string& path_W2, Eigen::MatrixXd& W2, const std::string& path_b2, Eigen::MatrixXd& b2
)
{ 
	load_txt(path_WG, WG);
	load_txt(path_bG, bG);
	load_txt(path_Wc, Wc);
	load_txt(path_bc, bc);
	load_txt(path_W0, W0);
	load_txt(path_b0, b0);
	load_txt(path_W1, W1);
	load_txt(path_b1, b1);
	load_txt(path_W2, W2);
	load_txt(path_b2, b2);

	//参数读取时为了尽可能保留位数，乘以了10e13,现在需要除回去
  	WG = WG * 10e-13;
  	Wc = Wc * 10e-13;
  	W0 = W0 * 10e-13;
  	W1 = W1 * 10e-13;
  	W2 = W2 * 10e-13;

	bG = bG * 10e-13;
	bc = bc * 10e-13;
	b0 = b0 * 10e-13;
	b1 = b1 * 10e-13;
	b2 = b2 * 10e-13;

}
 
void MDN::pdf_mix2D_Gaussian(Eigen::MatrixXd& Input_Speed, double& likelihood, Eigen::MatrixXd& Parameter){
	Eigen::MatrixXd one = MatrixXd::Ones(1,1);
	
	Eigen::MatrixXd alpha = Parameter.block(0,0,1,1);
	Eigen::MatrixXd mu = Parameter.block(0,1,1,2);
	Eigen::MatrixXd sigma = Parameter.block(0,3,1,2);
	Eigen::MatrixXd sigma1 = Parameter.block(0,3,1,1);
	Eigen::MatrixXd sigma2 = Parameter.block(0,4,1,1);
	Eigen::MatrixXd cor = Parameter.block(0,5,1,1);
	double k;
	
	Eigen::MatrixXd p;
	Eigen::MatrixXd p1;
	Eigen::MatrixXd p2;
	Eigen::MatrixXd p3;
	Eigen::MatrixXd z;
	Eigen::MatrixXd result;
	Eigen::MatrixXd final_result;
	Eigen::MatrixXd return_value;

	p = (Input_Speed - mu).array() * sigma.cwiseInverse().array(); 
	p1 = p.block(0,0,1,1);
	p2 = p.block(0,1,1,1);
	p3 = one - cor * cor;
	z = p1.cwiseAbs2() - 2 * (cor * p1 * p2) + p2.cwiseAbs2();
	result = (-(z / 2 * p3.cwiseInverse()));
	result = result.array().exp();
	k = 1 / sqrt(2 * M_PI);
	final_result = result.array() / (p3.array().sqrt() * sigma1.array() * sigma2.array()) * k;
	return_value = final_result * alpha;
	likelihood = return_value.sum();
}

void MDN::get_likelihood(Eigen::MatrixXd& Input_Speed, Eigen::MatrixXd& Input_History, double& likelihood)
{   
    MatrixXd Output_F = MatrixXd::Ones(1,8);
    MatrixXd Output = MatrixXd::Ones(1,8);
    MatrixXd c(1,8);
    MatrixXd a(1,8);
    MDN::load_Parameter(path_WG, WG, path_bG, bG, path_Wc, Wc, path_bc, bc, path_W0, W0, path_b0, b0, path_W1, W1, path_b1, b1, path_W2, W2, path_b2, b2);
    MDN::GRU_Foward(Input_History, c, a, WG, bG, Wc, bc);
    MDN::FC_3(c, Output, W0, W1, W2, b0, b1, b2);
    MDN::Post_process(Output, Output_F);
    MDN::pdf_mix2D_Gaussian(Input_Speed, likelihood, Output_F);
}

