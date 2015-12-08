using System;
using java.io;
using edu.stanford.nlp.process;
using System.IO;

namespace token_data
{
    class Program
    {
        static void Main(string[] args)
        {
            using (TextReader reader = System.IO.File.OpenText("C:\\Data\\semeval_for_elly_titleonly.txt"))
            {
                using (TextWriter writer = System.IO.File.CreateText("C:\\Data\\semeval_for_elly_titleonly_token.txt"))
                {
                    string[] inputdata = reader.ReadToEnd().Split('\n');

                    foreach (string line in inputdata)
                    {
                        string[] sp = line.Split('\t');

                        writer.Write(sp[0] + '\t');

                        var tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
                        var sent2Reader1 = new java.io.StringReader(sp[3]);
                        java.util.List rawWords1 = tokenizerFactory.getTokenizer(sent2Reader1).tokenize();
                        sent2Reader1.close();
                        var sent2Reader2 = new java.io.StringReader(sp[4]);
                        java.util.List rawWords2 = tokenizerFactory.getTokenizer(sent2Reader2).tokenize();
                        sent2Reader2.close();

                        for (int i = 0; i < rawWords1.size(); ++i)
                        {
                            writer.Write(rawWords1.get(i) + " ");
                        }
                        writer.Write('\t');

                        for (int i = 0; i < rawWords2.size(); ++i)
                        {
                            writer.Write(rawWords2.get(i) + " ");
                        }
                        writer.Write('\n');
                    }
                }
            }
            System.Console.ReadKey();

        }

    }
} //namespace
