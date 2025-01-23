import nltk
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score
from rouge import Rouge

nltk.download('wordnet')
nltk.download('omw-1.4')

# reference = "the cat is on the mat"
# candidate = "the cat is on the mat"
#
# references = [reference]
# candidate_tokens = candidate.split()

references = [['this', 'is', 'a', 'test'], ['this', 'is' 'test']]
candidates = ['this', 'is', 'a', 'test']

bleu_score = sentence_bleu(references, candidates)
print(f"BLEU Score: {bleu_score}")

meteor = meteor_score(references, candidates)
print(f"METEOR Score: {meteor}")

rouge = Rouge()
rouge_score = rouge.get_scores(references[0], candidates)

print(f"ROUGE-L Score: {rouge_score[0]['rouge-l']}")
