#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <array>
#include <map>
#include <cmath>
#include <boost/tokenizer.hpp>
#include <random>

#define _DEBUG 0;
#define TRAIN 0;

typedef int LABEL;
const int emb_dim = 200;
std::array<double, emb_dim> UNKNOWN;

void LoadDict(
	std::map<std::string, std::array<double, emb_dim>>& dict
	)
{
	fprintf(stderr, "Reading word embedding...\n");

	dict.clear();

#if _DEBUG
	const char File_Embedding[] = "c:\\data\\word_embedding_simple.txt"; // for debug
#else
	const char File_Embedding[] = "c:\\data\\word_embedding_size_200.txt";
#endif
	std::ifstream fin(File_Embedding);
	
	std::string new_word;
#if _DEBUG
	int it = 0;
#endif

	//Read *UNKNOWN*
	fin >> new_word;
	UNKNOWN.fill(0);
	for (int i = 0; i < emb_dim; ++i)
	{
		fin >> UNKNOWN[i];
	}

	while (!fin.eof())
	{
#if _DEBUG
		if (it % 1000 == 0)
			fprintf(stderr, "%d\n", it);
		++it;
#endif

		fin >> new_word;

		std::array<double, emb_dim> v;
		v.fill(0);
		for (int i = 0; i < emb_dim; ++i)
		{
			fin >> v[i];
		}

		dict[new_word] = v;
	}
	fin.close();
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

const std::uniform_real_distribution<> dist(0, 1);
std::random_device rd;
std::mt19937 gen(rd());

bool LookUp(
	const std::map<std::string, std::array<double, emb_dim>>& dict,
	const std::string& word,
	std::array<double, emb_dim>& v
	)
{
	auto search = dict.find(word);
	
	if (search == dict.end())
	{

		//for (int i = 0; i < emb_dim; ++i)
		//	v[i] = dist(gen);

		v = UNKNOWN;
		return false;
	}
	else
	{
		v = search->second;
		return true;
	}
}

void LoadSentence(
	const std::map<std::string, std::array<double, emb_dim>>& dict,
	std::vector<double>& thetas,
	std::vector<LABEL>& target
	)
{
	fprintf(stderr, "Load Sentences...\n");

#if TRAIN
	const char Train_data[] = "C:\\Data\\msr_train.txt";
#else
	const char Train_data[] = "C:\\Data\\msr_test.txt";
#endif

	std::ifstream fin(Train_data);

	int total_sentence = 0;

	while (!fin.eof())
	{
		++total_sentence;
		LABEL label;
		int s_id1, s_id2;
		fin >> label >> s_id1 >> s_id2;
		target.push_back(label);

#if _DEBUG
		fprintf(stderr, "%d %d %d\n", label, s_id1, s_id2);
#endif

		std::array<double, emb_dim> v1, v2, v;
		v1.fill(0);

		char ch;
		fin.get(ch);

		std::string sentence;
		getline(fin, sentence);
		
		//boost::char_separator<char> sep{ " \t.,\'?"};
		boost::char_separator<char> sep{ " \t" };
		boost::tokenizer<boost::char_separator<char>> tok1(sentence, sep);
		int len = 0;
		int c_unknown = 0;
		for (const auto& it : tok1)
		{
#if _DEBUG
			fprintf(stderr, "%s\n", it);
#endif
			if (!LookUp(dict, it, v))
				++c_unknown;
			for (int i = 0; i < emb_dim; ++i)
			{
				v1[i] += v[i];
			}
			++len;
		}
		for (int i = 0; i < emb_dim; ++i)
		{
			v1[i] /= len;
		}
		fprintf(stderr, "Sentence %d S1: UNKNOWN / LENGTH = %d / %d\n", total_sentence, c_unknown, len);

		v2.fill(0);
		getline(fin, sentence);
		boost::tokenizer<boost::char_separator<char>> tok2(sentence, sep);
		len = 0;
		c_unknown = 0;
		for (const auto& it : tok2)
		{
			if (!LookUp(dict, it, v))
				++c_unknown;
			for (int i = 0; i < emb_dim; ++i)
			{
				v2[i] += v[i];
			}
			++len;
		}
		for (int i = 0; i < emb_dim; ++i)
		{
			v2[i] /= len;
		}
		fprintf(stderr, "Sentence %d S2: UNKNOWN / LENGTH = %d / %d\n", total_sentence, c_unknown, len);

		thetas.push_back(ComputeAngel(v1, v2));
	}

	fprintf(stderr, "total sentences = %d\n", total_sentence);
}

int main()
{
	std::map<std::string, std::array<double, emb_dim>> dict;

	LoadDict(dict);

	std::vector<double> thetas;
	thetas.clear();
	std::vector<LABEL> target;
	target.clear();
	LoadSentence(dict, thetas, target);

//TODO:
	//SVM();

	//print
	{
#if TRAIN
		std::ofstream fout("train.txt");
#else
		std::ofstream fout("test.txt");
#endif
		for (auto i = 0; i < target.size(); ++i)
			fout << target[i] << "\t1:" << thetas[i] << '\n';
		fout.close();
	}
	 
	system("pause");
	return 0;
}