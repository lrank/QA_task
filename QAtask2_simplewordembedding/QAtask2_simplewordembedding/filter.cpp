#include <fstream>
#include <string>
#include <unordered_set>
#include <boost/tokenizer.hpp>
#include <boost/algorithm/string.hpp>

void to_lower(std::string& s)
{
	for (int i = 0; i < s.length(); ++i)
		if ('A' <= s[i] && s[i] <= 'Z')
			s[i] += 'a' - 'A';
}

//This process filter some comman words, such as 'I', 'am', and will LOWERCASE all the words.
int Filter_main() {

	fprintf(stderr, "Loading Filter Words...\n");
	//stream containing word to ignore;
	std::ifstream fwod("C:\\Data\\word_ignore.txt");
	std::unordered_set<std::string> words;
	words.clear();
	while (!fwod.eof())
	{
		std::string neww;
		fwod >> neww;
		words.insert(neww);
	}
	fwod.close();

	fprintf(stderr, "Filter Sentences...\n");

	const char Train_data[] = "C:\\Data\\SemevalData\\SemEval.train.token";
	const char Filtered_data[] = "C:\\Data\\SemevalData\\SemEval.train.token.filtered";

	std::ifstream fin(Train_data);
	std::ofstream fout(Filtered_data);

	while (!fin.eof())
	{
		int label;
		fin >> label;
		fout << label << '\t';
		std::string s1, s2;
		char ch;
		fin.get(ch);
		assert(ch == '\t');

		getline(fin, s1, '\t');

		boost::char_separator<char> sep{ " \t" };
		boost::tokenizer<boost::char_separator<char>> tok1(s1, sep);
		for (auto& it : tok1)
		{
			std::string tmp = it;
			to_lower(tmp);
			if (words.count(tmp) == 0)
				fout << tmp << ' ';
		}
		fout << '\t';

		getline(fin, s2, '\n');
		boost::tokenizer<boost::char_separator<char>> tok2(s2, sep);
		for (auto& it : tok2)
		{
			std::string tmp = it;
			to_lower(tmp);
			if (words.count(tmp) == 0)
				fout << tmp << ' ';
		}
		fout << '\n';
	}

	return 0;
}