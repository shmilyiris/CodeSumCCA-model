# evaluation.py
from nltk.translate.bleu_score import sentence_bleu
from rouge import Rouge
from nltk.translate.meteor_score import meteor_score

def evaluate_model(model, dataloader, vocab):
    model.eval()
    bleu_scores, meteor_scores, rouge_scores = [], [], []
    rouge = Rouge()
    inv_vocab = {v: k for k, v in vocab.items()}

    with torch.no_grad():
        for code_ids, _, tgt_output in dataloader:
            preds = model.generate(code_ids)
            for pred, tgt in zip(preds, tgt_output):
                pred_tokens = [inv_vocab.get(tok, '') for tok in pred]
                tgt_tokens = [inv_vocab.get(tok.item(), '') for tok in tgt if tok.item() != vocab['<PAD>']]
                bleu = sentence_bleu([tgt_tokens], pred_tokens)
                meteor = meteor_score([' '.join(tgt_tokens)], ' '.join(pred_tokens))
                rouge_l = rouge.get_scores(' '.join(pred_tokens), ' '.join(tgt_tokens))[0]['rouge-l']['f']
                bleu_scores.append(bleu)
                meteor_scores.append(meteor)
                rouge_scores.append(rouge_l)

    print(f"BLEU: {sum(bleu_scores)/len(bleu_scores):.4f}")
    print(f"METEOR: {sum(meteor_scores)/len(meteor_scores):.4f}")
    print(f"ROUGE-L: {sum(rouge_scores)/len(rouge_scores):.4f}")
