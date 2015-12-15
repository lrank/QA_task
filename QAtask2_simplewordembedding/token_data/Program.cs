﻿using System;
using java.io;
using edu.stanford.nlp.process;
using System.IO;

namespace token_data
{
    class Program
    {
        static void Main(string[] args)
        {
            using (TextReader reader = System.IO.File.OpenText("C:\\Data\\msr_paraphrase_train.txt"))
            {
                TextWriter writer1 = System.IO.File.CreateText("C:\\Data\\msr_paraphrase_train_s1.token");
                TextWriter writer2 = System.IO.File.CreateText("C:\\Data\\msr_paraphrase_train_s2.token");

                string[] inputdata = reader.ReadToEnd().Split('\n');

                    foreach (string line in inputdata)
                    {
                        string[] sp = line.Split('\t');

                        //writer.Write(sp[0] + '\t');

                        var tokenizerFactory = PTBTokenizer.factory(new CoreLabelTokenFactory(), "");
                        var sent2Reader1 = new java.io.StringReader(sp[3]);
                        java.util.List rawWords1 = tokenizerFactory.getTokenizer(sent2Reader1).tokenize();
                        sent2Reader1.close();
                        var sent2Reader2 = new java.io.StringReader(sp[4]);
                        java.util.List rawWords2 = tokenizerFactory.getTokenizer(sent2Reader2).tokenize();
                        sent2Reader2.close();

                        for (int i = 0; i < rawWords1.size(); ++i)
                        {
                            writer1.Write(rawWords1.get(i) + " ");
                        }
                        writer1.Write('\n');

                        for (int i = 0; i < rawWords2.size(); ++i)
                        {
                            writer2.Write(rawWords2.get(i) + " ");
                        }
                        writer2.Write('\n');
                    }
                writer1.Close();
                writer2.Close();
            }
            System.Console.ReadKey();

        }

    }
} //namespace
