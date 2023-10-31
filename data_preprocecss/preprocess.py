import os
import json
import tqdm
import tsv
import re
import gzip
import zipfile
import csv
import shutil


class Preprocess:
    def __init__(self):
        pass

    def txt2json(self, path, name):  # 获取带转换的txt路径
        with open(path) as file:
            lines = file.readlines()

        with open(name, 'w') as file:
            for line in lines:
                data = line.strip("\n")
                data = data.split()
                data = {
                    "qid": data[0],
                    "docid": data[2],
                    "rank": data[3],
                    "score": data[4],
                    "runstring": data[5]
                }
                json.dump(data, file)
                file.write('\n')

    def tsv2json_query(self, path, name):
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        with open(name, 'w') as f:
            for line in lines:
                data = line.strip("\n")
                data = data.replace('\t', ' ')
                data = data.split(' ', 1)
                data = {
                    "qid": data[0],
                    "query": data[1]
                }
                json.dump(data, f)
                f.write('\n')

    def tsv2json_qrel(self, path, name):
        with open(path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        with open(name, 'w') as f:
            for line in lines:
                data = line.strip("\n")
                data = data.replace('\t', ' ')
                data = data.split(' ')
                data = {
                    "qid": data[0],
                    "docid": data[2],
                    "relevance": data[3]
                }
                json.dump(data, f)
                f.write('\n')

    def preprocess_TREC_2023_Deep_Learning_Document_ranking_dataset(self):

        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  '2023_document_top100.txt')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   '2023_document_top100.json')
        self.txt2json(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  '2023_queries.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   '2023_queries.json')
        self.tsv2json_query(input_path, output_path)

        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  'docv2_dev_top100.txt')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   'docv2_dev_top100.json')
        self.txt2json(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  'docv2_dev_qrels.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   'docv2_dev_qrels.json')
        self.tsv2json_qrel(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  'docv2_dev_queries.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   'docv2_dev_queries.json')
        self.tsv2json_query(input_path, output_path)

        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  'docv2_dev2_top100.txt')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   'docv2_dev2_top100.json')
        self.txt2json(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  'docv2_dev2_qrels.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   'docv2_dev2_qrels.json')
        self.tsv2json_qrel(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Document ranking dataset',
                                  'docv2_dev2_queries.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Document ranking dataset json',
                                   'docv2_dev2_queries.json')
        self.tsv2json_query(input_path, output_path)

    def preprocess_TREC_2023_Deep_Learning_Passage_ranking_dataset(self):

        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  '2023_passage_top100.txt')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   '2023_passage_top100.json')
        self.txt2json(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  '2023_queries.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   '2023_queries.json')
        self.tsv2json_query(input_path, output_path)

        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  'passv2_dev_top100.txt')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   'passv2_dev_top100.json')
        self.txt2json(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  'passv2_dev_qrels.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   'passv2_dev_qrels.json')
        self.tsv2json_qrel(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  'passv2_dev_queries.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   'passv2_dev_queries.json')
        self.tsv2json_query(input_path, output_path)

        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  'passv2_dev2_top100.txt')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   'passv2_dev2_top100.json')
        self.txt2json(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  'passv2_dev2_qrels.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   'passv2_dev2_qrels.json')
        self.tsv2json_qrel(input_path, output_path)
        input_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning\Passage ranking dataset',
                                  'passv2_dev2_queries.tsv')
        output_path = os.path.join(parent_dir, 'TREC 2023 Deep Learning json\Passage ranking dataset json',
                                   'passv2_dev2_queries.json')
        self.tsv2json_query(input_path, output_path)

    def preprocess_ORCAS_qrel(self):

        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))

        input_path = os.path.join(parent_dir, 'ORCAS',
                                  'orcas-doctrain-qrels.tsv')
        output_path = os.path.join(parent_dir, 'ORCAS json',
                                   'orcas-doctrain-qrels.json')
        self.tsv2json_qrel(input_path, output_path)
        input_path = os.path.join(parent_dir, 'ORCAS',
                                  'orcas-doctrain-queries.tsv')
        output_path = os.path.join(parent_dir, 'ORCAS json',
                                   'orcas-doctrain-queries.json')
        self.tsv2json_query(input_path, output_path)

    """
    以下是新加的代码
    """

    def preprocess_NQ(self):

        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        output_path = os.path.join(parent_dir, 'NQ json')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        initial_query_path = os.path.join(os.path.join(parent_dir, 'NQ'), 'queries.jsonl')
        target_query_path = os.path.join(output_path, 'queries.jsonl')
        shutil.copyfile(initial_query_path, target_query_path)
        """
        initial_corpus_path = os.path.join(os.path.join(parent_dir, 'NQ'), 'corpus.jsonl')
        target_corpus_path = os.path.join(output_path, 'corpus.jsonl')
        shutil.copyfile(initial_corpus_path, target_corpus_path)
        """

        initial_test_path = os.path.join(os.path.join(parent_dir, 'NQ'), 'qrels')
        initial_test_path = os.path.join(initial_test_path, 'test.tsv')
        target_test_path = os.path.join(output_path, 'test.jsonl')
        with open(initial_test_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()

        with open(target_test_path, 'w') as f:
            for line in lines:
                if line == lines[0]:
                    continue
                data = line.strip("\n")
                data = data.replace('\t', ' ')
                data = data.split(' ')
                data = {
                    "qid": data[0],
                    "docid": data[1],
                    "relevance": data[2]
                }
                json.dump(data, f)
                f.write('\n')

    def preprocess_fiqa(self):

        parent_dir = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        output_path = os.path.join(parent_dir, 'fiqa json')
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        initial_query_path = os.path.join(os.path.join(parent_dir, 'fiqa'), 'queries.jsonl')
        target_query_path = os.path.join(output_path, 'queries.jsonl')
        shutil.copyfile(initial_query_path, target_query_path)
        initial_corpus_path = os.path.join(os.path.join(parent_dir, 'fiqa'), 'corpus.jsonl')
        target_corpus_path = os.path.join(output_path, 'corpus.jsonl')
        shutil.copyfile(initial_corpus_path, target_corpus_path)

        initial_qrels_path = os.path.join(os.path.join(parent_dir, 'fiqa'), 'qrels')

        initial_test_path = os.path.join(initial_qrels_path, 'test.tsv')
        target_test_path = os.path.join(output_path, 'test.jsonl')
        with open(initial_test_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        with open(target_test_path, 'w') as f:
            for line in lines:
                if line == lines[0]:
                    continue
                data = line.strip("\n")
                data = data.replace('\t', ' ')
                data = data.split(' ')
                data = {
                    "qid": data[0],
                    "docid": data[1],
                    "relevance": data[2]
                }
                json.dump(data, f)
                f.write('\n')

        initial_dev_path = os.path.join(initial_qrels_path, 'dev.tsv')
        target_dev_path = os.path.join(output_path, 'dev.jsonl')
        with open(initial_dev_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        with open(target_dev_path, 'w') as f:
            for line in lines:
                if line == lines[0]:
                    continue
                data = line.strip("\n")
                data = data.replace('\t', ' ')
                data = data.split(' ')
                data = {
                    "qid": data[0],
                    "docid": data[1],
                    "relevance": data[2]
                }
                json.dump(data, f)
                f.write('\n')

        initial_train_path = os.path.join(initial_qrels_path, 'train.tsv')
        target_train_path = os.path.join(output_path, 'train.jsonl')
        with open(initial_train_path, 'r', encoding='UTF-8') as f:
            lines = f.readlines()
        with open(target_train_path, 'w') as f:
            for line in lines:
                if line == lines[0]:
                    continue
                data = line.strip("\n")
                data = data.replace('\t', ' ')
                data = data.split(' ')
                data = {
                    "qid": data[0],
                    "docid": data[1],
                    "relevance": data[2]
                }
                json.dump(data, f)
                f.write('\n')



P = Preprocess()
# P.preprocess_TREC_2023_Deep_Learning_Document_ranking_dataset()
# P.preprocess_TREC_2023_Deep_Learning_Passage_ranking_dataset()
# P.preprocess_ORCAS_qrel()
# P.preprocess_NQ()
P.preprocess_fiqa()