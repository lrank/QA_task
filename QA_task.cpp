// QA_task.cpp : Defines the entry point for the console application.
//
#include <string>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <math>
#include <boost/tokenizer.hpp>

typedef int LABEL;
const int emb_dim = 50;

void LoadDict(
	std::map<std::string, std::array<double, emb_dim>>& dict
	)
{
	fprintf(stderr, "Reading word embedding...\n");
	
	dict.clear();

	const char File_Embedding[] = "c:/data/wrod_embedding_size_50.txt";
	FILE* fin = fopen(File_Embedding, "r");

	char word[] = "";
	while (fscanf(fin, "%s", word) != EOF)
	{

		std::array<double, emb_dim> v;
		v.clear();
		for (int i = 0; i < emb_dim; ++i)
		{
			double v_i;
			fscanf(fin, "%lf", v_i);
			v.push_back(v_i);
		}

		std::string new_word(word);
		dict[word] = v;
	}

	fclose(fin);
}

double ComputeAngel(
	const std::array<double, emb_dim>& v1,
	const std::array<double, emb_dim>& v2
	)
{
	double dot = 0, l1 = 0, l2 = 0;
	for (int i = 0; i < emb_dim; ++i)
	{
		dot += v1[i] * v2[i];
		l1 += v1[i] * v1[i];
		l2 += v2[i] * v2[i];
	}

	return dot / sqrt(l1 * l2);
}

void LookUp(
	const std::map<std::string, std::array<double, emb_dim>>& dict,
	const std::string& word,
	std::array<double, emb_dim>& v
	)
{
	if (dict.find(word, v))
		return;

	for (int i = 0; i < emb_dim; ++i)
		v[i] = random();
}

void LoadSentence(
	const std::map<std::string, std::array<double, emb_dim>>& dict,
	std::vector<double>& thetas
	)
{
	fprintf(stderr, "Load Sentences...\n");

	const char Train_data[] = "c:/data/msr_paraphrase_train.txt";

	std::ifstream fin(Train_data);

	while (!fin.eof)
	{
		LABEL label;
		int s_id1, s_id2;
		fin >> label >> s_id1 >> s_id2;

		std::string line;
		fin.
		fin.getline(line);

		std::vector<std::string> sentence = line.split('\t');
		assert(sentence.size() == 2);

		std::array<double, emb_dim> v1, v2, v;
		v1.clear();
		tokenizer<> tok1(sentence[0]);
		for (tokenizer<>::iterator it = tok1.begin(); it != tok1.end(); ++it)
		{
			LookUp(dict, *it, v);
			for (int i = 0; i < emb_dim; ++i)
			{
				v1[i] += v[i];
			}
		}

		v2.clear();
		tokenizer<> tok2(sentence[1]);
		for (tokenizer<>::iterator it = tok2.begin(); it != tok2.end(); ++it)
		{
			LookUp(dict, *it, v);
			for (int i = 0; i < emb_dim; ++i)
			{
				v2[i] += v[i];
			}
		}
		
		thetas.push_back(ComputeAngel(v1, v2));
	}
}

int main()
{
	std::map<std::string, std::vector<double>> dict;

	LoadDict(dict);

	std::vector<double> thetas;
	thetas.clear();
	LoadSentence(dict, thetas);

	SVM();

    return 0;
}
