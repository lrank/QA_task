#include "cnn/nodes.h"
#include "cnn/cnn.h"
#include "cnn/training.h"
#include "cnn/gpu-ops.h"
#include "cnn/expr.h"
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>

#include <iostream>
#include <fstream>

using namespace std;
using namespace cnn;
using namespace cnn::expr;

int main(int argc, char** argv) {
	cnn::Initialize(argc, argv);

/*	if (argc == 2) {
		ifstream in("");
		boost::archive::text_iarchive ia(in);
		ia >> m;
	}*/

	// parameters
	ifstream fin("C:\\Data\\train.txt");
	unsigned INPUT_SIZE = 22 * 22;
	unsigned DATA_SIZE = 4076;
	unsigned OUTPUT_SIZE = 2;
	fin >> INPUT_SIZE >> DATA_SIZE;
	const unsigned HIDDEN_SIZE = 128;
	const unsigned ITERATIONS = 200;

	Model m;
	SimpleSGDTrainer sgd(&m);

	ComputationGraph cg;

	Expression W = parameter(cg, m.add_parameters({ HIDDEN_SIZE, INPUT_SIZE }));
	Expression b = parameter(cg, m.add_parameters({ HIDDEN_SIZE }));
	Expression V = parameter(cg, m.add_parameters({ OUTPUT_SIZE, HIDDEN_SIZE }));
	Expression a = parameter(cg, m.add_parameters({ OUTPUT_SIZE }));

	vector<cnn::real> x_values;// (INPUT_SIZE * DATA_SIZE);
	x_values.clear();
	vector<cnn::real> y_values;// (OUTPUT_SIZE * DATA_SIZE);
	y_values.clear();
	for (int i = 0; i < DATA_SIZE; ++i)
	{
		int label;
		fin >> label;
		if (label == 0)
		{
			y_values.push_back(cnn::real(0));
			y_values.push_back(cnn::real(1));
		}
		else
		{
			y_values.push_back(cnn::real(1));
			y_values.push_back(cnn::real(0));
		}

		for (int j = 0; j < INPUT_SIZE; ++j)
		{
			fin >> label;
			x_values.push_back(cnn::real(label));
		}
	}

	cerr << x_values.size() << '\n' << y_values.size() << '\n';
	Dim x_dim({ INPUT_SIZE }, DATA_SIZE), y_dim({ OUTPUT_SIZE }, DATA_SIZE);
	cerr << "x_dim=" << x_dim << ", y_dim=" << y_dim << endl;
	// set x_values to change the inputs to the network
	Expression x = input(cg, x_dim, &x_values);
	// set y_values expressing the output
	Expression y = input(cg, y_dim, &y_values);

	Expression h = logistic(W*x + b);
	Expression y_pred = softmax(V*h + a);
	Expression loss = binary_log_loss(y_pred, y);
	//Expression loss = squared_distance(y_pred, y);
	Expression sum_loss = sum_batches(loss);

	cg.PrintGraphviz();

	// train the parameters
	for (unsigned iter = 0; 1; ++iter) {
		float my_loss = as_scalar(cg.forward());
		cg.backward();
		sgd.update(1e-3);
		sgd.update_epoch();
		cerr << "ITERATIONS = " << iter << endl;
		cerr << "E = " << my_loss << endl; //E=18.62, iter = 2500
	}
	//boost::archive::text_oarchive oa(cout);
	//oa << m;
}

