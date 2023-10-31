import json
import ir_datasets
import os
import wget
import gzip
import re
import xmltodict
from huggingface_hub import hf_hub_download
import zipfile
import csv

ALL_QUERY_DATASET = [
    "cast19",
    "cast20",
    "cast21",
    "clariq-fkw",
    "codec",
    "fire08",
    "fire10",
    "fire11",
    "fire12",
    "gecor",
    "gov2",
    "lariq_fkw",
    "orcas_i",
    "raocq",
    "robust04",
    "robust05",
    "trec09",
    "trec10",
    "trec11",
    "trec12",
    "trec13",
    "trec14",
]

def get_gov2_dataset(dataset_name):
    if not os.path.exists("data/" + dataset_name):
        os.makedirs("data/" + dataset_name)

    with open("data/" + dataset_name + "/queries.jsonl", "w", encoding="utf-8") as fw:
        dataset = ir_datasets.load(dataset_name)
        for query in dataset.queries_iter():
            data = {
                "query_id": query.query_id,
                "title": query.title,
                "description": query.description,
                "narrative": query.narrative
            }
            fw.write(json.dumps(data) + "\n")

def get_trec_robust_dataset():
    ### for robust 04
    url = "https://trec.nist.gov/data/robust/04.testset.gz"
    if not os.path.exists("data/robust04"):
        os.makedirs("data/robust04")
    if not os.path.exists("data/robust04/04.testset.gz"):
        wget.download(url, "data/robust04/04.testset.gz")

    ### for robust 05
    url = "https://trec.nist.gov/data/robust/05/05.50.topics.txt"
    if not os.path.exists("data/robust05"):
        os.makedirs("data/robust05")
    if not os.path.exists("data/robust05/05.50.topics.txt"):
        wget.download(url, "data/robust05/05.50.topics.txt")

    for dataset in ["robust04", "robust05"]:
        if dataset == "robust04":
            fr = gzip.open("data/robust04/04.testset.gz", "r")
            fw = open("data/robust04/queries.jsonl", "w", encoding="utf-8")
        else:
            fr = open("data/robust05/05.50.topics.txt", "r", encoding="utf-8")
            fw = open("data/robust05/queries.jsonl", "w", encoding="utf-8")
        data = {}
        next_line = ""
        for line in fr:
            if dataset == "robust04":
                line = line.decode("utf-8")
            line = line.strip()                    
            if line == "":
                next_line = ""
            if line == "</top>":
                fw.write(json.dumps(data) + "\n")
                data = {}
                next_line = ""
            if "<num>" in line:
                data["query_id"] = line[14:]
                continue
            if line == "<title>": 
                next_line = "title"
                continue
            elif "<title>" in line:
                data["title"] = line[8:]
                continue
            if next_line == "title":
                if "title" not in data:
                    data["title"] = line
                else:
                    data["title"] += " " + line
                continue
            if "<desc>" in line:
                next_line = "desc"
                continue
            if next_line == "desc":
                if "description" not in data:
                    data["description"] = line
                else:
                    data["description"] += " " + line
                continue
            if "<narr>" in line:
                next_line = "narr"
                continue
            if next_line == "narr":
                if "narrative" not in data:
                    data["narrative"] = line
                else:
                    data["narrative"] += " " + line
                continue

def get_fire_dataset():
    for dataset in ["fire08", "fire10", "fire11", "fire12"]:
        if dataset == "fire08":
            url = "https://www.isical.ac.in/~fire/data/topics/adhoc/en.topics.176-225.2012.txt"
        elif dataset == "fire10":
            url = "https://www.isical.ac.in/~fire/data/topics/adhoc/en.topics.126-175.2011.txt"
        elif dataset == "fire11":
            url = "https://www.isical.ac.in/~fire/data/topics/adhoc/en.topics.76-125.2010.txt"
        else:
            url = "http://www.isical.ac.in/~fire/data/topics/adhoc/en.topics.26-75.2008.txt"
        if not os.path.exists("data/" + dataset):
            os.makedirs("data/" + dataset)
        if not os.path.exists("data/" + dataset + "/queries.txt" + dataset):
            wget.download(url, "data/" + dataset + "/queries.txt")

        if dataset != "fire11":
            with open("data/" + dataset + "/queries.txt", "r", encoding="utf-8") as fr:
                with open("data/" + dataset + "/queries.jsonl", "w", encoding="utf-8") as fw:
                    data = {}
                    for line in fr:
                        line = line.strip()
                        if "<num>" in line:
                            pattern = r"(?<=<num>).*?(?=</num>)"
                            result = re.findall(pattern, line)
                            data["query_id"] = result[0]
                        elif "<title>" in line:
                            pattern = r"(?<=<title>).*?(?=</title>)"
                            result = re.findall(pattern, line)
                            data["title"] = result[0]
                        elif "<desc>" in line:
                            pattern = r"(?<=<desc>).*?(?=</desc>)"
                            result = re.findall(pattern, line)
                            data["description"] = result[0]
                        elif "<narr>" in line:
                            pattern = r"(?<=<narr>).*?(?=</narr>)"
                            result = re.findall(pattern, line)
                            data["narrative"] = result[0]
                        elif line == "</top>":
                            fw.write(json.dumps(data) + "\n")
                            data = {}
        else:
            with open("data/" + dataset + "/queries.txt", "r", encoding="utf-8") as fr:
                with open("data/" + dataset + "/queries.jsonl", "w", encoding="utf-8") as fw:
                    data = {}
                    next_line = ""
                    for line in fr:
                        line = line.strip()
                        if "<num>" in line:
                            pattern = r"(?<=<num>).*?(?=</num>)"
                            result = re.findall(pattern, line)
                            data["query_id"] = result[0]
                        elif "<title>" in line:
                            if "</title>" in line:
                                pattern = r"(?<=<title>).*?(?=</title>)"
                                result = re.findall(pattern, line)
                                data["title"] = result[0]
                            else:
                                pattern = r"(?<=<title>).*"
                                result = re.findall(pattern, line)
                                data["title"] = result[0]
                        elif "</title>" in line:
                            pattern = r".*?(?=</title>)"
                            result = re.findall(pattern, line)
                            data["title"] += " " + result[0]
                        if "<desc>" in line:
                            next_line = "desc"
                            continue
                        if "</desc>" in line:
                            next_line = ""
                            continue
                        if next_line == "desc":
                            if "description" not in data:
                                data["description"] = line
                            else:
                                data["description"] += " " + line
                            continue
                        if "<narr>" in line:
                            next_line = "narr"
                            continue
                        if "</narr>" in line:
                            next_line = ""
                            continue
                        if next_line == "narr":
                            if "narrative" not in data:
                                data["narrative"] = line
                            else:
                                data["narrative"] += " " + line
                            continue
                        if line == "</top>":
                            fw.write(json.dumps(data) + "\n")
                            data = {}

def get_codec_dataset():
    if not os.path.exists("data/codec"):
        os.makedirs("data/codec")
    if not os.path.exists("data/codec/query_reformulations.txt"):
        print("Please download the data file from https://github.com/grill-lab/CODEC/blob/main/topics/query_reformulations.txt")
    if not os.path.exists("data/codec/topics.json"):
        print("Please download the data file from https://github.com/grill-lab/CODEC/blob/main/topics/topics.json")
    reformulatons = {}
    with open("data/codec/query_reformulations.txt", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            if line[0] not in reformulatons:
                reformulatons[line[0]] = [line[1]]
            else:
                reformulatons[line[0]].append(line[1])
    with open("data/codec/topics.json", "r", encoding="utf-8") as fr:
        with open("data/codec/queries.jsonl", "w", encoding="utf-8") as fw:
            data_all = json.load(fr)
            for key in data_all.keys():
                data = {
                    "query_id": key,
                    "domain": data_all[key]["Domain"],
                    "title": data_all[key]["Query"],
                    "guidelines": data_all[key]["Guidelines"],
                    "reformulations": reformulatons[key]
                }
                fw.write(json.dumps(data) + "\n")

def get_cast19_dataset():
    if not os.path.exists("data/cast19"):
        os.makedirs("data/cast19")
    if not os.path.exists("data/cast19/evaluation_topics_annotated_resolved_v1.0.tsv"):
        print("Please download the data file from https://github.com/daltonj/treccastweb/blob/master/2019/data/evaluation/evaluation_topics_annotated_resolved_v1.0.tsv")
    if not os.path.exists("data/cast19/evaluation_topics_v1.0.json"):
        print("Please download the data file from https://github.com/daltonj/treccastweb/blob/master/2019/data/evaluation/evaluation_topics_v1.0.json")
    reformulations = {}
    with open("data/cast19/evaluation_topics_annotated_resolved_v1.0.tsv", "r", encoding="utf-8") as fr:
        for line in fr:
            line = line.strip().split("\t")
            qid, tid = line[0].split("_")
            reformulated_query = line[1]
            if qid not in reformulations:
                reformulations[qid] = {}
            if tid not in reformulations[qid]:
                reformulations[qid][tid] = reformulated_query
    with open("data/cast19/evaluation_topics_v1.0.json", "r", encoding="utf-8") as fr:
        with open("data/cast19/queries.jsonl", "w", encoding="utf-8") as fw:
            data_all = json.load(fr)
            for sample in data_all:
                data = {}
                data["qid"] = sample["number"]
                data["description"] = ""
                if "description" in sample:
                    data["description"] = sample["description"]
                data["title"] = sample["title"]
                data["turn"] = []
                for t in sample["turn"]:
                    data["turn"].append({
                        "number": t["number"],
                        "title": t["raw_utterance"],
                        "reformulations": [reformulations[str(data["qid"])][str(t["number"])]],
                    })
                fw.write(json.dumps(data) + "\n")

def get_cast20_21_dataset():
    if not os.path.exists("data/cast20"):
        os.makedirs("data/cast20")
    if not os.path.exists("data/cast21"):
        os.makedirs("data/cast21")
    if not os.path.exists("data/cast20/2020_manual_evaluation_topics_v1.0.json"):
        print("Please download the data file from https://github.com/daltonj/treccastweb/blob/master/2020/2020_manual_evaluation_topics_v1.0.json")
    if not os.path.exists("data/cast21/2021_manual_evaluation_topics_v1.0.json"):
        print("Please download the data file from https://github.com/daltonj/treccastweb/blob/master/2021/2021_manual_evaluation_topics_v1.0.json")
    for dataset in ["cast20", "cast21"]:
        if dataset == "cast20":
            file_path = "data/cast20/2020_manual_evaluation_topics_v1.0.json"
        else:
            file_path = "data/cast21/2021_manual_evaluation_topics_v1.0.json"
        with open(file_path, "r", encoding="utf-8") as fr:
            with open("data/" + dataset + "/queries.jsonl", "w", encoding="utf-8") as fw:
                data_all = json.load(fr)
                for sample in data_all:
                    data = {}
                    data["qid"] = sample["number"]
                    data["turn"] = []
                    for t in sample["turn"]:
                        data["turn"].append({
                            "number": t["number"],
                            "title": t["raw_utterance"],
                            "reformulations": [t["manual_rewritten_utterance"]],
                        })
                    fw.write(json.dumps(data) + "\n")

def get_gecor_dataset():
    if not os.path.exists("data/gecor"):
        os.makedirs("data/gecor")
    if not os.path.exists("data/gecor/CamRest676_annotated.json"):
        print("Please download the data file from https://github.com/terryqj0107/GECOR/blob/master/CamRest676_for_coreference_and_ellipsis_resolution/CamRest676_annotated.json")
    with open("data/gecor/CamRest676_annotated.json", "r", encoding="utf-8") as fr:
        with open("data/gecor/queries.jsonl", "w", encoding="utf-8") as fw:
            data_all = json.load(fr)
            for sample in data_all:
                data = {}
                data["goal"] = sample["goal"]["text"]
                data["turn"] = []
                for t in sample["dial"]:
                    data["turn"].append({
                        "number": t["turn"],
                        "title": t["usr"]["transcript"],
                        "reformulations": [t["usr"]["transcript_complete"]],
                        "response": t["sys"]["sent"]
                    })
                fw.write(json.dumps(data) + "\n")

def get_trec_09_14_dataset():
    # for dataset in ["trec09", "trec10", "trec11", "trec12", "trec13", "trec14"]:
    for dataset in ["trec13", "trec14"]:
        if dataset == "trec09":
            trec_url = "https://trec.nist.gov/data/web/09/wt09.topics.full.xml"
        elif dataset == "trec10":
            trec_url = "https://trec.nist.gov/data/web/10/wt2010-topics.xml"
        elif dataset == "trec11":
            trec_url = "https://trec.nist.gov/data/web/11/full-topics.xml"
        elif dataset == "trec12":
            trec_url = "https://trec.nist.gov/data/web/12/full-topics.xml"
        elif dataset == "trec13":
            trec_url = "https://trec.nist.gov/data/web/2013/trec2013-topics.xml"
        else:
            trec_url = "https://trec.nist.gov/data/web/2014/trec2014-topics.xml"
        if not os.path.exists("data/" + dataset):
            os.makedirs("data/" + dataset)
        if not os.path.exists("data/" + dataset + "/" + dataset + ".topics.xml"):
            wget.download(trec_url, "data/" + dataset + "/" + dataset + ".topics.xml")
        
        with open("data/" + dataset + "/" + dataset + ".topics.xml", "r", encoding="utf-8") as fr:
            with open("data/" + dataset + "/queries.jsonl", "w", encoding="utf-8") as fw:
                data_all = fr.read()
                data_all = xmltodict.parse(data_all)
                for sample in data_all["webtrack20" + dataset[-2:]]["topic"]:
                    data = {}
                    data["query_id"] = sample["@number"]
                    data["query_type"] = sample["@type"]
                    data["title"] = sample["query"]
                    if "description" not in sample or sample["description"] == None:
                        description = ""
                    else:
                        description = sample["description"]
                        description = description.split("\n")
                        description = [x.strip() for x in description]
                        description = " ".join(description)
                    data["description"] = description
                    data["subtopic"] = []
                    if "subtopic" not in sample:
                        continue
                    for t in sample["subtopic"]:
                        if "#text" not in t:
                            continue
                        subtopic_title = t["#text"]
                        subtopic_title = subtopic_title.split("\n")
                        subtopic_title = [x.strip() for x in subtopic_title]
                        subtopic_title = " ".join(subtopic_title)
                        data["subtopic"].append({
                            "subtopic_id": t["@number"],
                            "subtopic_type": t["@type"],
                            "subtopic_title": subtopic_title
                        })
                    fw.write(json.dumps(data) + "\n")

def get_orcas_i_dataset():
    url = "https://researchdata.tuwien.ac.at/records/pp7xz-n9a06/files/ORCAS-I-gold.tsv?download=1"
    if not os.path.exists("data/orcas_i"):
        os.makedirs("data/orcas_i")
    if not os.path.exists("data/orcas_i/ORCAS-I-gold.tsv"):
        wget.download(url, "data/orcas_i/ORCAS-I-gold.tsv")
    with open("data/orcas_i/ORCAS-I-gold.tsv", "r", encoding="utf-8") as fr:
        with open("data/orcas_i/queries.jsonl", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                if idx == 0:
                    continue
                line = line.strip().split("\t")
                data = {
                    "query_id": line[0],
                    "title": line[1],
                    "query_type": line[4]
                }
                fw.write(json.dumps(data) + "\n")

def get_clariq_fkw_dataset():
    if not os.path.exists("data/lariq_fkw"):
        os.makedirs("data/lariq_fkw")
    if not os.path.exists("data/clariq-fkw/ClariQ-FKw.tsv.train"):
        print("Please download the data file from https://github.com/isekulic/CQ-generation/blob/main/data/ClariQ-FKw.tsv.train")
    if not os.path.exists("data/clariq-fkw/ClariQ-FKw.tsv.dev"):
        print("Please download the data file from https://github.com/isekulic/CQ-generation/blob/main/data/ClariQ-FKw.tsv.dev")
    with open("data/clariq-fkw/ClariQ-FKw.tsv.train", "r", encoding="utf-8") as fr:
        with open("data/clariq-fkw/queries.train.jsonl", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                if idx == 0:
                    continue
                line = line.strip().split("\t")
                data = {
                    "query_id": line[1],
                    "title": line[2],
                    "clarification": line[4]
                }
                fw.write(json.dumps(data) + "\n")
    with open("data/clariq-fkw/ClariQ-FKw.tsv.dev", "r", encoding="utf-8") as fr:
        with open("data/clariq-fkw/queries.dev.jsonl", "w", encoding="utf-8") as fw:
            for idx, line in enumerate(fr):
                if idx == 0:
                    continue
                line = line.strip().split("\t")
                data = {
                    "query_id": line[1],
                    "title": line[2],
                    "clarification": line[4]
                }
                fw.write(json.dumps(data) + "\n")

def get_rao_qa_dataset():
    if not os.path.exists("data/raocq/clarification_questions_dataset"):
        print("Please download the data file from https://drive.google.com/file/d/1zPmO7tRdqfz3FuiCoLtKqc0yQoKfz1Vz/view, unzip and put it under data/raocq")
    for dataset in ["askubuntu.com", "superuser.com", "unix.stackexchange.com"]:
        anno_dict = {}
        with open("data/raocq/clarification_questions_dataset/data/" + dataset + "/human_annotations", "r", encoding="utf-8") as fr:
            for line in fr:
                line = line.strip().split("\t")
                anno1 = line[0].split(",")
                anno2 = line[1].split(",")
                postid =  anno1[1]
                anno1_conf = int(anno1[4])
                anno2_conf = int(anno2[4])
                if anno1_conf >= anno2_conf:
                    anno_dict[postid] = anno1[2]
                else:
                    anno_dict[postid] = anno2[2]
        post_dict = {}
        with open("data/raocq/clarification_questions_dataset/data/" + dataset + "/post_data.tsv", "r", encoding="utf-8") as fr:
            for idx, line in enumerate(fr):
                if idx == 0:
                    continue
                line = line.strip().split("\t")
                postid = line[0].split("_")[1]
                post_dict[postid] = {
                    "title": line[1],
                    "post": line[2]
                }
        with open("data/raocq/clarification_questions_dataset/data/" + dataset + "/qa_data.tsv", "r", encoding="utf-8") as fr:
            with open("data/raocq/" + dataset + ".queries.jsonl", "w", encoding="utf-8") as fw:
                for idx, line in enumerate(fr):
                    if idx == 0:
                        continue
                    line = line.strip().split("\t")
                    postid = line[0].split("_")[1]
                    questions = line[1:11]
                    answers = line[11:21]
                    if postid in anno_dict:
                        data = {
                            "query_id": postid,
                            "title": post_dict[postid]["title"],
                            "description": post_dict[postid]["post"],
                            "clarification": questions[int(anno_dict[postid])],
                            "answer": answers[int(anno_dict[postid])]
                        }
                        fw.write(json.dumps(data) + "\n")

def get_query2doc_dataset():
    # This dataset is in JSONL format. We do not need further preprocessing.
    repo_id = "intfloat/query2doc_msmarco"
    repo_type = "dataset"

    if not os.path.exists("data/query2doc_msmarco"):
        os.makedirs("data/query2doc_msmarco")

    if not os.path.exists("data/query2doc_msmarco/train.jsonl"):
        hf_hub_download(repo_id=repo_id, filename="train.jsonl", repo_type=repo_type, local_dir="data/query2doc_msmarco/", local_dir_use_symlinks=False)

    if not os.path.exists("data/query2doc_msmarco/test.jsonl"):
        hf_hub_download(repo_id=repo_id, filename="test.jsonl", repo_type=repo_type, local_dir="data/query2doc_msmarco/", local_dir_use_symlinks=False)

    if not os.path.exists("data/query2doc_msmarco/dev.jsonl"):
        hf_hub_download(repo_id=repo_id, filename="dev.jsonl", repo_type=repo_type, local_dir="data/query2doc_msmarco/", local_dir_use_symlinks=False)

def get_qrecc_dataset():
    repo_id = "svakulenk0/qrecc"
    repo_type = "dataset"

    if not os.path.exists("data/qrecc"):
        os.makedirs("data/qrecc")

    if not os.path.exists("data/qrecc/qrecc-training.json"):
        hf_hub_download(repo_id=repo_id, filename="qrecc-training.json", repo_type=repo_type, local_dir="data/qrecc/", local_dir_use_symlinks=False)
    if not os.path.exists("data/qrecc/qrecc-test.json"):
        hf_hub_download(repo_id=repo_id, filename="qrecc-test.json", repo_type=repo_type, local_dir="data/qrecc/", local_dir_use_symlinks=False)

    def process(fin, fout):
        with open(fin, "r", encoding="utf-8") as fr:
            with open(fout, "w", encoding="utf-8") as fw:
                data_all = json.load(fr)
                new_data_all = {}
                for sample in data_all:
                    if sample["Conversation_no"] not in new_data_all:
                        new_data_all[sample["Conversation_no"]] = {}
                        new_data_all[sample["Conversation_no"]]["session_id"] = sample["Conversation_no"]
                        new_data_all[sample["Conversation_no"]]["session"] = []
                    new_data_all[sample["Conversation_no"]]["session"].append({
                        "title": sample["Question"],
                        "reformulations": [sample["Rewrite"]],
                        "answer": sample["Answer"],
                        "number": sample["Turn_no"]
                    })
                for key in new_data_all.keys():
                    data = new_data_all[key]
                    fw.write(json.dumps(data) + "\n")
    
    process("data/qrecc/qrecc-training.json", "data/qrecc/queries.train.jsonl")
    process("data/qrecc/qrecc-test.json", "data/qrecc/queries.test.jsonl")

    
def get_canard_dataset():
    url = "https://obj.umiacs.umd.edu/elgohary/CANARD_Release.zip"

    if not os.path.exists("data/canard"):
        os.makedirs("data/canard")
    
    if not os.path.exists("data/canard/CANARD_Release.zip"):
        wget.download(url, "data/canard/")
    
    with zipfile.ZipFile("data/canard/CANARD_Release.zip", "r") as zip_ref:
        zip_ref.extract("CANARD_Release/dev.json", "data/canard/")
        zip_ref.extract("CANARD_Release/test.json", "data/canard/")
        zip_ref.extract("CANARD_Release/train.json", "data/canard/")
    
    def process(fin, fout):
        with open(fin, "r", encoding="utf-8") as fr:
            with open(fout, "w", encoding="utf-8") as fw:
                data_all = json.load(fr)
                new_data_all = {}
                for sample in data_all:
                    if sample["QuAC_dialog_id"] not in new_data_all:
                        new_data_all[sample["QuAC_dialog_id"]] = {}
                        new_data_all[sample["QuAC_dialog_id"]]["session_id"] = sample["QuAC_dialog_id"]
                        new_data_all[sample["QuAC_dialog_id"]]["context"] = sample["History"]
                        new_data_all[sample["QuAC_dialog_id"]]["session"] = []
                    new_data_all[sample["QuAC_dialog_id"]]["session"].append({
                        "title": sample["Question"],
                        "reformulations": [sample["Rewrite"]],
                        "answer": "",
                        "number": sample["Question_no"]
                    })
                    if sample["Question_no"] > 1:
                        new_data_all[sample["QuAC_dialog_id"]]["session"][sample["Question_no"] - 2]["answer"] = sample["History"][-1]
                for key in new_data_all.keys():
                    data = new_data_all[key]
                    fw.write(json.dumps(data) + "\n")

    process("data/canard/CANARD_Release/train.json", "data/canard/queries.train.jsonl")
    process("data/canard/CANARD_Release/dev.json", "data/canard/queries.dev.jsonl")
    process("data/canard/CANARD_Release/test.json", "data/canard/queries.test.jsonl")

def get_msrp_datset():
    repo_id = "HHousen/msrp"
    repo_type = "dataset"

    if not os.path.exists("data/msrp"):
        os.makedirs("data/msrp")

    if not os.path.exists("data/msrp/train.csv"):
        hf_hub_download(repo_id=repo_id, filename="train.csv", repo_type=repo_type, local_dir="data/msrp/", local_dir_use_symlinks=False)
    if not os.path.exists("data/msrp/test.csv"):
        hf_hub_download(repo_id=repo_id, filename="test.csv", repo_type=repo_type, local_dir="data/msrp/", local_dir_use_symlinks=False)

    def process(fin, fout):
        with open(fin, "r", encoding="utf-8") as fr:
            with open(fout, "w", encoding="utf-8") as fw:
                reader = csv.reader(fr, delimiter=',')
                for idx, row in enumerate(reader):
                    if idx == 0:
                        continue
                    data = {
                        "sentence_1": row[3],
                        "sentence_2": row[4],
                        "label": row[0]
                    }
                    fw.write(json.dumps(data) + "\n")
    
    process("data/msrp/train.csv", "data/msrp/queries.train.jsonl")
    process("data/msrp/test.csv", "data/msrp/queries.test.jsonl")

def get_mimics_dataset():
    if not os.path.exists("data/mimics"):
        os.makedirs("data/mimics")
    
    if not os.path.exists("data/mimics/MIMICS-master.zip"):
        print("Please download the data file from https://github.com/microsoft/MIMICS/archive/refs/heads/master.zip")

    with zipfile.ZipFile("data/mimics/MIMICS-master.zip", "r") as zip_ref:
        zip_ref.extract("MIMICS-master/data/MIMICS-Manual.tsv", "data/mimics/")

    # We only use label >= 2 clarifications
    with open("data/mimics/MIMICS-master/data/MIMICS-Manual.tsv", "r", encoding="utf-8") as fr:
        with open("data/mimics/queries.jsonl", "w", encoding="utf-8") as fw:
            reader = csv.reader(fr, delimiter='\t')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                query = row[0]
                question = row[1]
                clarification = []
                if row[9] != "" and int(row[9]) >= 2:
                    clarification.append(row[2])
                if row[10] != "" and int(row[10]) >= 2:
                    clarification.append(row[3])
                if row[11] != "" and int(row[11]) >= 2:
                    clarification.append(row[4])
                if row[12] != "" and int(row[12]) >= 2:
                    clarification.append(row[5])
                if row[13] != "" and int(row[13]) >= 2:
                    clarification.append(row[6])
                if len(clarification) == 0:
                    continue
                data = {
                    "query": query,
                    "question": question,
                    "clarification": clarification
                }
                fw.write(json.dumps(data) + "\n")

def get_mantis_dataset():
    if not os.path.exists("data/mantis"):
        os.makedirs("data/mantis")
    
    if not os.path.exists("data/mantis/json_dataset_with_intents.7z"):
        print("Please download the data file from https://drive.google.com/file/d/1JI9VAuHllyZxr7XhTYLhx7iI2EVd3-a4/view and unpack it")

    def process(fin, fout):
        with open(fin, "r", encoding="utf-8") as input_file:
            original_data = json.load(input_file)

        new_data = []
        for _, value in original_data.items():
            if value.get("has_intent_labels"):
                session = []
                for utterance in value["utterances"]:
                    if "intent" in utterance:
                        session.append({
                            "trun_id": utterance["utterance_pos"],
                            "title": utterance["utterance"],
                            "query_type": utterance["intent"],
                        })
                new_data.append({
                    "title": value["title"],
                    "session": session,
                })

        # Save jsonl file
        with open(fout, "w", encoding="utf-8") as output_file:
            for data in new_data:
                json.dump(data, output_file, ensure_ascii=False)
                output_file.write('\n')

    process("data/mantis/json_dataset_with_intents/merged_dev_intents.json", "data/mantis/dev.jsonl")
    process("data/mantis/json_dataset_with_intents/merged_test_intents.json", "data/mantis/test.jsonl")
    process("data/mantis/json_dataset_with_intents/merged_train_intents.json", "data/mantis/train.jsonl")

def get_mimics_duo_dataset():
    if not os.path.exists("data/mimics_duo"):
        os.makedirs("data/mimics_duo")
    
    if not os.path.exists("data/mimics_duo/Task2-QualityLabelling.tsv"):
        print("Please download the data file from https://github.com/Leila-Ta/MIMICS-Duo/blob/main/Data/Task2-QualityLabelling.tsv")

    with open("data/mimics_duo/Task2-QualityLabelling.tsv", 
              "r", encoding="utf-8") as fr:
        with open("data/mimics_duo/queries.jsonl", "w", encoding="utf-8") as fw:
            reader = csv.reader(fr, delimiter='\t')
            for idx, row in enumerate(reader):
                if idx == 0:
                    continue
                query = row[0]
                question = row[1]
                clarification = []
                if row[7] != "" and int(row[7]) >= 4:
                    clarification.append(row[2])
                if row[8] != "" and int(row[8]) >= 4:
                    clarification.append(row[3])
                if row[9] != "" and int(row[9]) >= 4:
                    clarification.append(row[4])
                if row[10] != "" and int(row[10]) >= 4:
                    clarification.append(row[5])
                if row[11] != "" and int(row[11]) >= 4:
                    clarification.append(row[6])
                if len(clarification) == 0:
                    continue
                data = {
                    "query": query,
                    "question": question,
                    "clarification": clarification
                }
                fw.write(json.dumps(data) + "\n")

def main():
    # get_gov2_dataset("gov2/trec-tb-2004")
    # get_gov2_dataset("gov2/trec-tb-2005")
    # get_gov2_dataset("gov2/trec-tb-2006")
    # get_trec_robust_dataset()
    # get_fire_dataset()
    # get_codec_dataset()
    # get_cast19_dataset()
    # get_cast20_21_dataset()
    # get_gecor_dataset()
    # get_trec_09_14_dataset()
    # get_orcas_i_dataset()
    # get_clariq_fkw_dataset()
    # get_rao_qa_dataset()
    # get_query2doc_dataset()
    # get_qrecc_dataset()
    # get_canard_dataset()
    # get_msrp_datset()
    get_mimics_dataset()
    # get_mantis_dataset()
    # get_mimics_duo_dataset()

if __name__ == "__main__":
    main()
