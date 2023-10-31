import os
import json
import wget
from IPython import embed
from datasets import load_dataset, VerificationMode, load_from_disk

def get_squad_v2_dataset():
    if not os.path.exists("data/squad_v2"):
        os.makedirs("data/squad_v2")

    dataset = load_dataset("squad_v2", cache_dir="data/squad_v2/", verification_mode=VerificationMode.NO_CHECKS)
    data_path = "data/squad_v2/"
    dataset.save_to_disk(data_path)

    dataset = load_from_disk(data_path)

    def processing(fout, split):
        with open(fout, "w", encoding="utf-8") as fw:
            for sample in dataset[split]:
                answers = []
                for ans in sample["answers"]["text"]:
                    if ans not in answers:
                        answers.append(ans) 
                if len(answers) == 0:
                    continue
                data = {
                    "id": sample["id"],
                    "title": sample["title"],
                    "context": sample["context"],
                    "question": sample["question"],
                    "answers": answers
                }
                fw.write(json.dumps(data) + "\n")
    
    processing("data/squad_v2/train.jsonl", "train")
    processing("data/squad_v2/dev.jsonl", "validation")

def get_hotpotqa_dataset():
    # if not os.path.exists("data/hotpotqa"):
    #     os.makedirs("data/hotpotqa")
    
    # if not os.path.exists("data/hotpotqa/train.json"):
    #     wget.download("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_train_v1.1.json", "data/hotpotqa/train.json")
    
    # if not os.path.exists("data/hotpotqa/dev.json"):
    #     wget.download("http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json", "data/hotpotqa/dev.json")

    def processing(fin, fout):
        with open(fin, "r", encoding="utf-8") as fr:
            with open(fout, "w", encoding="utf-8") as fw:
                data_all = json.load(fr)
                for sample in data_all:
                    sp_facts = []
                    for fact in sample["supporting_facts"]:
                        for paragraph in sample["context"]:
                            if paragraph[0] == fact[0]:
                                if fact[1] >= len(paragraph[1]):
                                    continue
                                sp_fact = paragraph[1][fact[1]]
                                sp_facts.append(sp_fact)
                    all_ps = []
                    for paragraph in sample["context"]:
                        p = paragraph[0] + " <=SEP=> " + "".join(paragraph[1])
                        all_ps.append(p)
                    data = {
                        "id": sample["_id"],
                        "supporting_facts": sp_facts,
                        "context": all_ps,
                        "question": sample["question"],
                        "answers": [sample["answer"]]
                    }
                    fw.write(json.dumps(data) + "\n")

    processing("data/hotpotqa/dev.json", "data/hotpotqa/dev.jsonl")
    processing("data/hotpotqa/train.json", "data/hotpotqa/train.jsonl")

def get_coqa_dataset():
    if not os.path.exists("data/coqa"):
        os.makedirs("data/coqa")
    
    # dataset = load_dataset("coqa", cache_dir="data/coqa/", verification_mode=VerificationMode.NO_CHECKS)
    data_path = "data/coqa/"
    # dataset.save_to_disk(data_path)

    dataset = load_from_disk(data_path)

    def processing(fout, split):
        with open(fout, "w", encoding="utf-8") as fw:
            for sample in dataset[split]:
                assert len(sample["questions"]) == len(sample["answers"]["input_text"])
                data = {
                    "context": sample["story"],
                    "questions": sample["questions"],
                    "answers": sample["answers"]["input_text"]
                }
                fw.write(json.dumps(data) + "\n")

    processing("data/coqa/train.jsonl", "train")
    processing("data/coqa/dev.jsonl", "validation")

def get_msmarco_dataset():
    if not os.path.exists("data/ms_marco"):
        os.makedirs("data/ms_marco")
        os.makedirs("data/ms_marco/dev")
    
    # we do not use the training set as it is too large
    dataset = load_dataset("ms_marco", name="v2.1", split="validation", cache_dir="data/ms_marco/")
    data_path = "data/ms_marco/"
    dataset.save_to_disk(data_path + "dev")

    dataset = load_from_disk(data_path + "dev")
    with open("data/ms_marco/dev.jsonl", "w", encoding="utf-8") as fw:
        for sample in dataset:
            sp_facts = []
            if len(sample["answers"]) == 0 or "no answer" in sample["answers"][0].lower():
                continue
            assert len(sample["passages"]["is_selected"]) == len(sample["passages"]["passage_text"])
            for idx, label in enumerate(sample["passages"]["is_selected"]):
                if label == 1:
                    sp_facts.append(sample["passages"]["passage_text"][idx])
            data = {
                "id": sample["query_id"],
                "context": sample["passages"]["passage_text"],
                "question": sample["query"],
                "answers": sample["answers"],
                "supporting_facts": sp_facts,
            }
            fw.write(json.dumps(data) + "\n")

def get_trivia_qa_dataset():
    if not os.path.exists("data/trivia_qa"):
        os.makedirs("data/trivia_qa")

    # we do not use the training set as it is too large
    dataset = load_dataset("trivia_qa", name="rc", split="validation", cache_dir="data/trivia_qa/")
    data_path = "data/trivia_qa/"
    dataset.save_to_disk(data_path)

    dataset = load_from_disk(data_path)

    with open("data/trivia_qa/dev.jsonl", "w", encoding="utf-8") as fw:
        for sample in dataset:
            assert len(sample["search_results"]["description"]) == len(sample["search_results"]["title"]) == len(sample["search_results"]["search_context"])
            if len(sample["answer"]["normalized_aliases"]) == 0:
                continue
            context = []
            for i in range(len(sample["search_results"]["description"])):
                context.append(sample["search_results"]["title"][i] + " <=SEP=> " + sample["search_results"]["description"][i] + " <=SEP=> " + sample["search_results"]["search_context"][i])
            data = {
                "id": sample["question_id"],
                "question": sample["question"],
                "answers": sample["answer"]["normalized_aliases"],
                "context": context
            }
            fw.write(json.dumps(data) + "\n")

def get_quac_dataset():
    if not os.path.exists("data/quac"):
        os.makedirs("data/quac")

    dataset = load_dataset("quac", cache_dir="data/quac/")
    data_path = "data/quac/"
    dataset.save_to_disk(data_path)

    dataset = load_from_disk(data_path)
    
    def processing(fout, split):
        with open(fout, "w", encoding="utf-8") as fw:
            for sample in dataset[split]:
                answers = []
                for x in sample["orig_answers"]["texts"]:
                    if x != "CANNOTANSWER":
                        answers.append(x)
                    else:
                        answers.append("")
                assert len(sample["questions"]) == len(answers)
                data = {
                    "id": sample["dialogue_id"],
                    "background": sample["wikipedia_page_title"] + " <=SEP=> " + sample["background"],
                    "context": sample["section_title"] + " <=SEP=> " + sample["context"],
                    "questions": sample["questions"],
                    "answers": answers
                }
                fw.write(json.dumps(data) + "\n")

    processing("data/quac/train.jsonl", "train")
    processing("data/quac/dev.jsonl", "validation")

def get_cnndm_dataset():
    if not os.path.exists("data/cnndm"):
        os.makedirs("data/cnndm")

    # dataset = load_dataset("ccdv/cnn_dailymail", name="3.0.0", split="validation", cache_dir="data/cnndm/")
    data_path = "data/cnndm/"
    # dataset.save_to_disk(data_path)

    dataset = load_from_disk(data_path)

    with open("data/cnndm/dev.jsonl", "w", encoding="utf-8") as fw:
        for sample in dataset:
            data = {
                "id": sample["id"],
                "article": sample["article"],
                "summary": sample["highlights"],
            }
            fw.write(json.dumps(data) + "\n")

def get_wikisum_dataset():
    # if not os.path.exists("data/wikisum"):
    #     os.makedirs("data/wikisum")

    # dataset = load_dataset("d0rj/wikisum", cache_dir="data/wikisum/")
    data_path = "data/wikisum/"
    # dataset.save_to_disk(data_path)
    dataset = load_from_disk(data_path)

    def processing(fout, split):
        with open(fout, "w", encoding="utf-8") as fw:
            for sample in dataset[split]:
                data = {
                    "article": sample["title"] + " <=SEP=> " + sample["article"],
                    "summary": sample["summary"],
                }
                fw.write(json.dumps(data) + "\n")

    processing("data/wikisum/train.jsonl", "train")
    processing("data/wikisum/dev.jsonl", "validation")
    processing("data/wikisum/test.jsonl", "test")

def get_multi_news_dataset():
    if not os.path.exists("data/multinews"):
        os.makedirs("data/multinews")

    # dataset = load_dataset("multi_news", cache_dir="data/multinews/")
    data_path = "data/multinews/"
    # dataset.save_to_disk(data_path)
    dataset = load_from_disk(data_path)

    def processing(fout, split):
        with open(fout, "w", encoding="utf-8") as fw:
            for sample in dataset[split]:
                data = {
                    "article": sample["document"].split("|||||"),
                    "summary": sample["summary"],
                }
                fw.write(json.dumps(data) + "\n")

    processing("data/multinews/train.jsonl", "train")
    processing("data/multinews/dev.jsonl", "validation")
    processing("data/multinews/test.jsonl", "test")

def main():
    # get_squad_v2_dataset()
    # get_hotpotqa_dataset()
    # get_coqa_dataset()
    # get_msmarco_dataset()
    # get_trivia_qa_dataset()
    # get_quac_dataset()
    # get_cnndm_dataset()
    get_wikisum_dataset()
    get_multi_news_dataset()

if __name__ == "__main__":

    main()
