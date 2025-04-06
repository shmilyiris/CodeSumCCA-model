import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from utils import load_all_repos, split_dataset, prepare_input_target_pairs
from model.summarizer import CodeSummaryModel
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge
import numpy as np


def evaluate(model_path="../result/t5_code_summary/t5_code_summary_epoch3.pt"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    base_model = CodeSummaryModel()
    dataset = load_all_repos("../data")
    _, _, test_set = split_dataset(dataset)
    x_test, y_test = prepare_input_target_pairs(test_set, base_model)

    smooth_fn = SmoothingFunction().method1
    bleu_scores = []
    meteor_scores = []
    rouge = Rouge()
    rouge_scores = []

    print("[INFO] Generating predictions and evaluating...")
    for i in range(len(x_test)):
        input_text = x_test[i]
        reference = y_test[i]

        input_ids = tokenizer.encode(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
        output_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)

        # Evaluation
        if not prediction.strip():
            continue
        bleu = sentence_bleu([reference.split()], prediction.split(), smoothing_function=smooth_fn)
        meteor = meteor_score([reference.split()], prediction.split())
        rouge_result = rouge.get_scores(prediction, reference)[0]['rouge-l']['f']

        bleu_scores.append(bleu)
        meteor_scores.append(meteor)
        rouge_scores.append(rouge_result)

        print("Input:", input_text[:100].replace('\n', ' '), "...")
        print("Reference:", reference)
        print("Prediction:", prediction)

    print("\n===== Evaluation Results =====")
    print(f"BLEU Score:  {np.mean(bleu_scores):.4f}")
    print(f"METEOR Score:{np.mean(meteor_scores):.4f}")
    print(f"ROUGE-L Score:{np.mean(rouge_scores):.4f}")


if __name__ == '__main__':
    evaluate()