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

	Parameters* P_W = m.add_parameters({ HIDDEN_SIZE, INPUT_SIZE });
	Parameters* P_b = m.add_parameters({ HIDDEN_SIZE });
	Parameters* P_V = m.add_parameters({ OUTPUT_SIZE, HIDDEN_SIZE });
	Parameters* P_a = m.add_parameters({ OUTPUT_SIZE });

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
			double in;
			fin >> in;
			x_values.push_back(cnn::real(in));
		}
	}

	cerr << x_values.size() << '\n' << y_values.size() << '\n';
	Dim x_dim({ INPUT_SIZE }, DATA_SIZE), y_dim({ OUTPUT_SIZE }, DATA_SIZE);
	cerr << "x_dim=" << x_dim << ", y_dim=" << y_dim << endl;

	// train the parameters
	for (unsigned iter = 0; 1; ++iter) {
		{
			ComputationGraph cg;

			Expression W = parameter(cg, P_W);
			Expression b = parameter(cg, P_b);
			Expression V = parameter(cg, P_V);
			Expression a = parameter(cg, P_a);

			// set x_values to change the inputs to the network
			Expression x = input(cg, x_dim, &x_values);
			// set y_values expressing the output
			Expression y = input(cg, y_dim, &y_values);

			Expression h = logistic(W*x + b);
			Expression y_pred = softmax(V*h + a);
			Expression loss = binary_log_loss(y_pred, y);
			Expression sum_loss = sum_batches(loss);

			//cg.PrintGraphviz();

			float my_loss = as_scalar(cg.forward());
			cg.backward();
			sgd.update(1);
			sgd.update_epoch();
			cerr << "ITERATIONS = " << iter << '\t';
			cerr << "E = " << my_loss << '\t'; //P = 1, iter = 6000, l_rate = 1
		}

		double l = 0;
		for (int i = 0; i < DATA_SIZE; ++i)
		{
			ComputationGraph cgr;
			Expression Wr = parameter(cgr, P_W);
			Expression br = parameter(cgr, P_b);
			Expression Vr = parameter(cgr, P_V);
			Expression ar = parameter(cgr, P_a);

			vector<cnn::real> x(INPUT_SIZE);
			for (int j = 0; j < INPUT_SIZE; ++j)
			{
				x[j] = x_values[j + i * INPUT_SIZE];
			}
			Expression xr = input(cgr, { INPUT_SIZE }, &x);
			vector<cnn::real> y(2);
			y[0] = 0; y[1] = 1;
			Expression yr = input(cgr, { OUTPUT_SIZE }, &y);
			Expression hr = logistic(Wr*xr + br);
			Expression y_predr = softmax(Vr*hr + ar);
			Expression lossr = dot_product(y_predr, yr);

			double t = as_scalar(cgr.forward()) > 0.5? 0 : 1;
			l += (t == y_values[i * 2]) ? 1 : 0;
		}
		cerr << "P = " << l << '\n';
		
	}
	//boost::archive::text_oarchive oa(cout);
	//oa << m;
	system("pause");
	return 0;
}

