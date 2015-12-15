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
	ifstream fin("C:\\Data\\msr_train.txt");
	ifstream fin_mt("C:\\Data\\mtscore\\All_train_score.txt");

	unsigned INPUT_SIZE_MT = 7;
	unsigned INPUT_SIZE = 22 * 22;
	unsigned DATA_SIZE = 4076;
	unsigned OUTPUT_SIZE = 2;
	
	const unsigned HIDDEN1_SIZE = 128;
	const unsigned HIDDEN2_SIZE = 64;
	const unsigned ITERATIONS = 50000;
	fin >> INPUT_SIZE >> DATA_SIZE;

	Model m;
	//SimpleSGDTrainer sgd(&m);
	MomentumSGDTrainer sgd(&m);

	Parameters* P_W1 = m.add_parameters({ HIDDEN1_SIZE, INPUT_SIZE + INPUT_SIZE_MT });
	Parameters* P_b1 = m.add_parameters({ HIDDEN1_SIZE });
	Parameters* P_W2 = m.add_parameters({ HIDDEN2_SIZE, HIDDEN1_SIZE });
	Parameters* P_b2 = m.add_parameters({ HIDDEN2_SIZE });
	Parameters* P_V = m.add_parameters({ OUTPUT_SIZE, HIDDEN2_SIZE });
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

		for (int j = 0; j < INPUT_SIZE_MT; ++j)
		{
			double x;
			fin_mt >> x;
			x_values.push_back(cnn::real(x));
		}
		for (int j = 0; j < INPUT_SIZE; ++j)
		{
			double x;
			fin >> x;
			x_values.push_back(cnn::real(x));
		}
	}
	fin.close();
	fin_mt.close();

	cerr << x_values.size() << '\n' << y_values.size() << '\n';
	Dim x_dim({ INPUT_SIZE + INPUT_SIZE_MT }, DATA_SIZE), y_dim({ OUTPUT_SIZE }, DATA_SIZE);
	cerr << "x_dim=" << x_dim << ", y_dim=" << y_dim << endl;


	//Load dev data
	//ifstream f_test("C:\\Data\\msr_train.txt");
	ifstream f_test("C:\\Data\\msr_test.txt");
	ifstream f_test_mt("C:\\Data\\mtscore\\All_test_score.txt");

	vector<cnn::real> x_test_values;// (INPUT_SIZE * DATA_SIZE);
	x_test_values.clear();
	vector<cnn::real> y_test_values;// (OUTPUT_SIZE * DATA_SIZE);
	y_test_values.clear();
	unsigned TEST_SIZE;
	f_test >> INPUT_SIZE >> TEST_SIZE;
	for (int i = 0; i < TEST_SIZE; ++i)
	{
		int label;
		f_test >> label;
		if (label == 0)
		{
			y_test_values.push_back(cnn::real(0));
			y_test_values.push_back(cnn::real(1));
		}
		else
		{
			y_test_values.push_back(cnn::real(1));
			y_test_values.push_back(cnn::real(0));
		}
		for (int j = 0; j < INPUT_SIZE_MT; ++j)
		{
			double x;
			f_test_mt >> x;
			x_test_values.push_back(cnn::real(x));
		}
		for (int j = 0; j < INPUT_SIZE; ++j)
		{
			double x;
			f_test >> x;
			x_test_values.push_back(cnn::real(x));
		}
	}
	f_test_mt.close();
	f_test.close();


	double max = 0;
	int ki = 0;
	for (unsigned iter = 0; iter < ITERATIONS; ++iter) {
	//for (unsigned iter = 0; true; ++iter) {
		// train the parameters
		{
			ComputationGraph cg;

			Expression W1 = parameter(cg, P_W1);
			Expression b1 = parameter(cg, P_b1);
			Expression W2 = parameter(cg, P_W2);
			Expression b2 = parameter(cg, P_b2);
			Expression V = parameter(cg, P_V);
			Expression a = parameter(cg, P_a);

			// set x_values to change the inputs to the network
			Expression x = input(cg, x_dim, &x_values);
			// set y_values expressing the output
			Expression y = input(cg, y_dim, &y_values);

			Expression h1 = rectify(W1*x + b1);
			Expression h2 = rectify(W2*h1 + b2);
			Expression y_pred = softmax(V*h2 + a);
			Expression loss = binary_log_loss(y_pred, y);
			Expression sum_loss = sum_batches(loss);

			//cg.PrintGraphviz();

			float my_loss = as_scalar(cg.forward());
			cg.backward();
			sgd.update(1e-2);
			sgd.update_epoch();
			cerr << "ITERATIONS = " << iter << '\t';
			cerr << "E = " << my_loss << '\t'; //P = 1, iter = 6000, l_rate = 1
		}

		//DEV SCORE
		double l = 0;
		for (int i = 0; i < TEST_SIZE; ++i)
		{
			ComputationGraph cgr;
			Expression Wr1 = parameter(cgr, P_W1);
			Expression br1 = parameter(cgr, P_b1);
			Expression Wr2 = parameter(cgr, P_W2);
			Expression br2 = parameter(cgr, P_b2);
			Expression Vr = parameter(cgr, P_V);
			Expression ar = parameter(cgr, P_a);

			vector<cnn::real> x(INPUT_SIZE + INPUT_SIZE_MT);
			for (int j = 0; j < INPUT_SIZE + INPUT_SIZE_MT; ++j)
			{
				x[j] = x_test_values[j + i * (INPUT_SIZE + INPUT_SIZE_MT)];
			}
			Expression xr = input(cgr, { INPUT_SIZE + INPUT_SIZE_MT }, &x);
			vector<cnn::real> y(2);
			y[0] = 0; y[1] = 1;
			Expression yr = input(cgr, { OUTPUT_SIZE }, &y);
			Expression hr1 = rectify(Wr1*xr + br1);
			Expression hr2 = rectify(Wr2*hr1 + br2);
			Expression y_predr = softmax(Vr*hr2 + ar);
			Expression lossr = dot_product(y_predr, yr);

			double t = as_scalar(cgr.forward()) > 0.5? 0 : 1;
			l += (t == y_test_values[i * 2]) ? 1 : 0;
			//cerr << ((t == y_test_values[i * 2]) ? 1 : 0) << ' ';
		}
		l /= TEST_SIZE;
		cerr << "P = " << l << '\t';
		
		if (l > max)
		{
			max = l;
			ki = iter;
		}
		cerr << "max acc = " << max << "\tat\t" << ki << '\n';
	}
	//boost::archive::text_oarchive oa(cout);
	//oa << m;
	system("pause");
	return 0;
}

