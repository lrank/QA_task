#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <array>
#include <boost/tokenizer.hpp>
#include <algorithm>
#include <boost/numeric/ublas/matrix.hpp>
#include <random>

typedef int LABEL;
const int emb_dim = 200;
std::array<double, emb_dim> UNKNOWN;
typedef std::array<double, emb_dim> wordvector;

void LoadDict(
	std::map<std::string, wordvector>& dict
	)
{
	fprintf(stderr, "Reading word embedding...\n");

	dict.clear();


#ifdef _DEBUG
	const char File_Embedding[] = "c:\\data\\word_embedding_simple_200.txt"; // for debug
#else
	const char File_Embedding[] = "c:\\data\\word_embedding_size_200.txt";
#endif

	std::ifstream fin(File_Embedding);

	std::string new_word;

	//Read *UNKNOWN*
	fin >> new_word;
	UNKNOWN.fill(0);
	for (int i = 0; i < emb_dim; ++i)
	{
		fin >> UNKNOWN[i];
	}

	while (!fin.eof())
	{
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

	fprintf(stderr, "embeding size = %d\n", dict.size());
}


const std::uniform_real_distribution<> dist(0, 1);
std::random_device rd;
std::mt19937 gen(rd());

bool LookUp(
	const std::map<std::string, wordvector>& dict,
	const std::string& word,
	wordvector& v
	)
{
	auto search = dict.find(word);

	if (search == dict.end())
	{

//Comment: (Y.L) think random vector for unknown word could be better here
		for (int i = 0; i < emb_dim; ++i)
			v[i] = dist(gen);

		//v = UNKNOWN;
		return false;
	}
	else
	{
		v = search->second;
		return true;
	}
}

//Return Cosine(\theta)
inline double ComputeAngel(
	const wordvector& v1,
	const wordvector& v2
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

void BuildMatrix(
	const int l1, const int l2,
	const std::vector<wordvector>& v1,
	const std::vector<wordvector>& v2,
	std::vector<double>& matrix
	)
{
	for (int i = 0; i < l1; ++i)
	{
		for (int j = 0; j < l2; ++j)
		{
			matrix.push_back(ComputeAngel(v1[i], v2[j]));
		}
	}
}

inline double ComputeTriAngel(
	const wordvector& v1_0,
	const wordvector& v1_1,
	const wordvector& v1_2,
	const wordvector& v2_0,
	const wordvector& v2_1,
	const wordvector& v2_2
	)
{
	std::array<double, emb_dim * 3> v1, v2;
	for (int i = 0; i < emb_dim; ++i)
	{
		v1[i] = v1_0[i]; v1[i + emb_dim] = v1_1[i]; v1[i + emb_dim] = v1_2[i];
		v1[i] = v2_0[i]; v1[i + emb_dim] = v2_1[i]; v1[i + emb_dim] = v2_2[i];
	}

	double dot = 0, l1 = 0, l2 = 0;
	for (int i = 0; i < emb_dim * 3; ++i)
	{
		dot += v1[i] * v2[i];
		l1 += v1[i] * v1[i];
		l2 += v2[i] * v2[i];
	}

	return dot / sqrt(l1 * l2);
}

void BuildTrigramMatrix(
	const int l1, const int l2,
	const std::vector<wordvector>& v1,
	const std::vector<wordvector>& v2,
	std::vector<double>& matrix
	)
{
	for (int i = 2; i < l1; ++i)
	{
		for (int j = 2; j < l2; ++j)
		{
			matrix.push_back(ComputeTriAngel(v1[i - 2], v1[i - 1], v1[i], v2[j - 2], v2[j - 1], v2[j]));
		}
	}
}


void LoadSentence(
	const std::map<std::string, wordvector>& dict,
	std::vector<LABEL>& target,
	int K
	)
{
	fprintf(stderr, "Load Sentences...\n");

	const char Train_data[] = "C:\\Data\\SemevalData\\SemEval.dev.token";

	std::ifstream fin(Train_data);

	std::vector<int> sentence_length;
	sentence_length.clear();
	std::vector<std::vector<double>> matrice;
	matrice.clear();

	int word_count = 0;
	int total = 0;

	while (!fin.eof())
	{
		LABEL label;
		fin >> label;
		target.push_back(label);
		char ch;
		fin.get(ch);

		//First Sentence
		std::string sentence;
		getline(fin, sentence, '\t');
		boost::char_separator<char> sep{ " \t" };
		boost::tokenizer<boost::char_separator<char>> tok1(sentence, sep);

		int lens1 = 0;
		int c_unknown = 0;

		wordvector v;
		std::vector<wordvector> v1;
		v1.clear();
		for (const auto& it : tok1)
		{
			if (!LookUp(dict, it, v))
				++c_unknown;
			v1.push_back(v);
			++lens1;
		}
		sentence_length.push_back(lens1);

		//fprintf(stderr, "Data ID %zd S1: UNKNOWN / LENGTH = %d / %d\n", sentence_length.size() / 2 + 1, c_unknown, lens1);
		word_count += c_unknown;
		total += lens1;


		//Second Sentence
		getline(fin, sentence, '\n');
		boost::tokenizer<boost::char_separator<char>> tok2(sentence, sep);

		int lens2 = 0;
		c_unknown = 0;
		std::vector<wordvector> v2;
		v2.clear();
		for (const auto& it : tok2)
		{
			if (!LookUp(dict, it, v))
				++c_unknown;
			v2.push_back(v);
			++lens2;
		}
		sentence_length.push_back(lens2);
		//fprintf(stderr, "Data ID %zd S1: UNKNOWN / LENGTH = %d / %d\n", sentence_length.size() / 2 + 1, c_unknown, lens2);
		word_count += c_unknown;
		total += lens2;



		std::vector<double> matrix;
		matrix.clear();
		BuildMatrix(lens1, lens2, v1, v2, matrix);
		matrice.push_back(matrix);

		//Debug
		//fprintf(stderr, "%d\n", matrice.size());
	}

	fprintf(stderr, "total words = %d, missing %d word_vec\n", total, word_count);

	{
		fprintf(stderr, "choose K = ");
		//Dynamically choose K <= median
		
		/*
		std::vector<int> tmp(sentence_length);
		sort(tmp.begin(), tmp.end());
		K = tmp[tmp.size() / 2];
		*/

		K = 30;

		fprintf(stderr, "%d\n", K);
	}


	//Matrix preprocess
	{
		//std::ofstream fout("C:\\Data\\SemevalData\\SemEval.train.cross_unigram");
		std::ofstream fout("C:\\Data\\SemevalData\\SemEval.dev.cross_trigram");
		fout << K * K << '\t' << matrice.size() << '\n';
		
		int n = 0;
		for (auto A = matrice.begin(); A != matrice.end(); ++A)
		{
			fout << target[n / 2];

			int row = sentence_length[n++], col = sentence_length[n++];
			boost::numeric::ublas::matrix<double> T(std::max(row, K), std::max(col, K));
			for (int i = 0, l = 0; i < row; ++i)
				for (int j = 0; j < col; ++j)
					T(i, j) = A->at(l++);

			//Blowup
			if (col < K)
			{
				for (int i = 0; i < row; ++i)
					for (int j = col; j < K; ++j)
						T(i, j) = T(i, j - col);
				col = K;
			}
			if (row < K)
			{
				for (int i = row; i < K; ++i)
					for (int j = 0; j < col; ++j)
						T(i, j) = T(i - row, j);
				row = K;
			}

			//Shrink
			boost::numeric::ublas::matrix<double> M(K, K);
			int div_row = row / K, mod_row = K - row % K;
			int div_col = col / K, mod_col = K - col % K;

			for (int it_row = 0, i = 0, delta_row = div_row; it_row < K; ++it_row, i += delta_row)
			{
				if (it_row == mod_row)
					++delta_row;

				for (int it_col = 0, j = 0, delta_col = div_col; it_col < K; ++it_col, j += delta_col)
				{
					if (it_col == mod_col)
						++delta_col;

					//fprintf(stderr, "%d %d\n", it_row, it_col);
					M(it_row, it_col) = -1e10;
					for (int t1 = i; t1 < i + delta_row; ++t1)
						for (int t2 = j; t2 < j + delta_col; ++t2)
						{
							M(it_row, it_col) = std::max(M(it_row, it_col), T(t1, t2));
						}
				}

			}

			//Flaten M
			for (int i = 0; i < K; ++i)
				for (int j = 0; j < K; ++j)
					fout << ' ' << M(i, j);
			fout << '\n';
		}

		fout.close();
	}
}

int main()
{
	std::map<std::string, std::array<double, emb_dim>> dict;

	LoadDict(dict);

	std::vector<std::array<double, emb_dim>> v1, v2;
	v1.clear();
	v2.clear();
	std::vector<LABEL> target;
	target.clear();

	//K-max-out constant
	int K = 10;
	LoadSentence(dict, target, K);

	system("pause");
	return 0;
}