#include <iostream>
#include <Dense>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include "MDN.h"
using namespace std;
using namespace Eigen;
int input_length = 4;		// Length of input(x1, y1, x2, y2)
int GRU_Neuron_Num = 8;			// GRU Neuron Number()
int Hidden_dim = 16;			// DNN Hidden layer dimension()
MDN::MDN()
{
    WG = MatrixXd::Ones(input_length + GRU_Neuron_Num, GRU_Neuron_Num*2);
    bG = MatrixXd::Ones(GRU_Neuron_Num*2,1);
    Wc = MatrixXd::Ones(input_length + GRU_Neuron_Num, GRU_Neuron_Num);
    bc = MatrixXd::Ones(GRU_Neuron_Num, 1);
    W0 = MatrixXd::Ones(GRU_Neuron_Num, Hidden_dim);
    W2 = MatrixXd::Ones(Hidden_dim,6);
    b0 = MatrixXd::Ones(Hidden_dim,1);
    b2 = MatrixXd::Ones(6,1);
	        //    /home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/bias_h0.txt
    path_W0 = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/weights_h0.txt";
    path_b0 = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/bias_h0.txt";
    path_W2 = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/weights_h2.txt";
    path_b2 = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/bias_h2.txt";
    path_Wc = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/weights_rnn_candidate.txt";
    path_bc = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/bias_rnn_candidate.txt";
    path_WG = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/weights_rnn_gate.txt";
    path_bG = "/home/letian/MDN/Benchmark_copy/newly_trained_MDN/MDN/Parameter/bias_rnn_gate.txt";
	
}

void MDN::sigmoid(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A)
{
	A = 1 / (1 + (-Z).array().exp());
	assert(A.size() == Z.size());
}

void MDN::relu(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A)
{
	A = Z.array().max(0);
	assert(A.rows() == Z.rows());
	assert(A.cols() == Z.cols());
}

void MDN::tanh(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A)
{
	A = ((Z).array().exp() - (-Z).array().exp()) / ((Z).array().exp() + (-Z).array().exp());
	assert(A.size() == Z.size());
}

void MDN::Soft_max(const Eigen::MatrixXd& Z, Eigen::MatrixXd& A)
{
	A = Z.array().exp() / Z.array().exp().sum();
}

void MDN::linear_forward(const Eigen::MatrixXd& A, const Eigen::MatrixXd& W, const Eigen::VectorXd& b, Eigen::MatrixXd& Z)
{
	Z = ((A * W).transpose() + b).transpose();
	assert(Z.rows() == A.rows());
	assert(Z.cols() == W.cols());
}

void MDN::linear_activation_forward(const Eigen::MatrixXd& A_prev, const Eigen::MatrixXd& W, const Eigen::VectorXd& b, const std::string activation, Eigen::MatrixXd& A)
{
	Eigen::MatrixXd Z;
	if (activation == "sigmoid") {
		linear_forward(A_prev, W, b, Z);
		sigmoid(Z, A);
	}
	else if (activation == "relu") {
		linear_forward(A_prev, W, b, Z);
		relu(Z, A);
	}
	else if (activation == "tanh") {
		linear_forward(A_prev, W, b, Z);
		tanh(Z, A);
	}

	assert(A.rows() == A_prev.rows());
	assert(A.cols() == W.cols());
}


void MDN::GRU_Cell(const Eigen::MatrixXd& c_prev, Eigen::MatrixXd& x_input, Eigen::MatrixXd& c_next, 
   const Eigen::MatrixXd& WG, const Eigen::VectorXd& bG,  const Eigen::MatrixXd& Wc,const Eigen::MatrixXd& bc)
{
	Eigen::MatrixXd conc(1, c_prev.cols()+x_input.cols());
	Eigen::MatrixXd conc_for_candidate(1, c_prev.cols()+x_input.cols());
	Eigen::MatrixXd ones = MatrixXd::Ones(1, c_prev.cols());
	Eigen::MatrixXd Gate;
	Eigen::MatrixXd Gate_r;
	Eigen::MatrixXd Gate_u;
	Eigen::MatrixXd c_candidate;
	int c_length = c_prev.cols();

	// concatenate
	conc<<x_input, c_prev;
	// Gates
	linear_activation_forward(conc, WG, bG, "sigmoid", Gate);
	Gate_r = Gate.block(0,0,1,c_length);
	Gate_u = Gate.block(0,c_length,1,c_length);
	// conc_for_candidate, transform to array for element-wise multiplying, and transform back to matrix
	conc_for_candidate << x_input, (Gate_r.array() * c_prev.array()).matrix();
	// candidate	
	linear_activation_forward(conc_for_candidate, Wc, bc, "tanh", c_candidate);
	// State and Output
	c_next = (Gate_u.array() * c_prev.array()).matrix() + ((ones.array() - Gate_u.array()) * c_candidate.array() ).matrix();
}

// x_input is the input(5,4), with 5 timestep
void MDN::GRU_Foward(Eigen::MatrixXd& input, Eigen::MatrixXd& output,  
const Eigen::MatrixXd& WG, const Eigen::MatrixXd& bG, const Eigen::MatrixXd& Wc,const Eigen::MatrixXd& bc)
{
	// Define the variables
	MatrixXd c1(1,GRU_Neuron_Num);
    MatrixXd a1(1,GRU_Neuron_Num);
	MatrixXd c2(1,GRU_Neuron_Num);
    MatrixXd a2(1,GRU_Neuron_Num);
	MatrixXd c3(1,GRU_Neuron_Num);
    MatrixXd a3(1,GRU_Neuron_Num);
	MatrixXd c4(1,GRU_Neuron_Num);
    MatrixXd a4(1,GRU_Neuron_Num);
	// Initialize the c0 as zero
	
	MatrixXd c0 = MatrixXd::Zero(1,GRU_Neuron_Num);
	
	// x.block(a,b,x,y)extract inputs, start from(a,b), size is (x,y)
	MatrixXd x0 = input.block(0,0,1,4);
	MatrixXd x1 = input.block(1,0,1,4);
	MatrixXd x2 = input.block(2,0,1,4);
	MatrixXd x3 = input.block(3,0,1,4);
	MatrixXd x4 = input.block(4,0,1,4);

	// GRU_Forward for 5 timesteps
	GRU_Cell(c0, x0, c1, WG, bG, Wc, bc );
	GRU_Cell(c1, x1, c2, WG, bG, Wc, bc );
	GRU_Cell(c2, x2, c3, WG, bG, Wc, bc );
	GRU_Cell(c3, x3, c4, WG, bG, Wc, bc );
	GRU_Cell(c4, x4, output, WG, bG, Wc, bc );
}

void MDN::FC_3(Eigen::MatrixXd& Input, Eigen::MatrixXd& Output, const Eigen::MatrixXd& W0, const Eigen::MatrixXd& W2,
	const Eigen::VectorXd& b0,  const Eigen::VectorXd& b2)
{
	Eigen::MatrixXd Output0;
	Eigen::MatrixXd Output1;
	Eigen::MatrixXd Output2;

	linear_activation_forward(Input, W0, b0, "tanh", Output0);
	linear_forward(Output0, W2, b2, Output2);
	Output = Output2;
}

void MDN::Post_process(Eigen::MatrixXd& Input, Eigen::MatrixXd& Output)
{
	MatrixXd output(1,6);
	MatrixXd weight = Input.block(0,0,1,1);
	MatrixXd mu = 	  Input.block(0,1,1,2);
	MatrixXd sigma =  Input.block(0,3,1,2);
	MatrixXd cor =    Input.block(0,5,1,1);

	Soft_max(weight, weight);
	sigma = sigma.array().exp();
	tanh(cor, cor);
	output << weight,
			  mu,
			  sigma,
			  cor;
	Output = output;
}

// The MDN consists of a GRU networks of 5 timesteps and a FC network of 3 layers, postprocess the output for the final output
void MDN::MDN_Model(Eigen::MatrixXd& Input, Eigen::MatrixXd& Output,
 const Eigen::MatrixXd& WG, const Eigen::VectorXd& bG, const Eigen::MatrixXd& Wc, const Eigen::MatrixXd& bc,
 const Eigen::MatrixXd& W0, const Eigen::MatrixXd& W2, const Eigen::VectorXd& b0, const Eigen::VectorXd& b2)
{
	Eigen::MatrixXd GRU_output;
	Eigen::MatrixXd FC_Output;
	GRU_Foward(Input, GRU_output, WG, bG, Wc, bc);
	FC_3(GRU_output, FC_Output, W0, W2, b0, b2);
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
 const std::string& path_W2, Eigen::MatrixXd& W2, const std::string& path_b2, Eigen::MatrixXd& b2
)
{ 
	load_txt(path_WG, WG);
	load_txt(path_bG, bG);
	load_txt(path_Wc, Wc);
	load_txt(path_bc, bc);
	load_txt(path_W0, W0);
	load_txt(path_b0, b0);
	load_txt(path_W2, W2);
	load_txt(path_b2, b2);

	//参数读取时为了尽可能保留位数，乘以了10e13,现在需要除回去
  	WG = WG * 10e-13;
  	Wc = Wc * 10e-13;
  	W0 = W0 * 10e-13;
  	W2 = W2 * 10e-13;

	bG = bG * 10e-13;
	bc = bc * 10e-13;
	b0 = b0 * 10e-13;
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
	k = 1 / (2 * M_PI);
	final_result = result.array() / (p3.array().sqrt() * sigma1.array() * sigma2.array()) * k;
	return_value = final_result * alpha;
	likelihood = return_value.sum();
}

void MDN::get_likelihood(Eigen::MatrixXd& Input_Speed, Eigen::MatrixXd& Input_History, double& likelihood)
{   
    MatrixXd Output_F = MatrixXd::Ones(1,6);
    MatrixXd Output = MatrixXd::Ones(1,6);
    MatrixXd c(1,GRU_Neuron_Num);
    MatrixXd a(1,GRU_Neuron_Num);
    MDN::load_Parameter(path_WG, WG, path_bG, bG, path_Wc, Wc, path_bc, bc, path_W0, W0, path_b0, b0, path_W2, W2, path_b2, b2);
    MDN::GRU_Foward(Input_History, c, WG, bG, Wc, bc);
    MDN::FC_3(c, Output, W0, W2, b0, b2);
    MDN::Post_process(Output, Output_F);
    MDN::pdf_mix2D_Gaussian(Input_Speed, likelihood, Output_F);
}
