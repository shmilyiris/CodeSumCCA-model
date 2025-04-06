import spacy
import numpy as np

class ReadabilityEvaluator:
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def evaluate_readability(self, sentences):
        total_len = []
        total_depth = []
        complete_sent_count = 0

        for sent in sentences:
            doc = self.nlp(sent)
            total_len.append(len(doc))

            # 依存树深度：找到每个 token 向上的最大路径
            def depth(token):
                if token.head == token:
                    return 1
                return 1 + depth(token.head)

            max_depth = max([depth(tok) for tok in doc])
            total_depth.append(max_depth)

            # 是否有主语和谓语（名词和动词）
            has_noun = any(tok.pos_ == "NOUN" for tok in doc)
            has_verb = any(tok.pos_ == "VERB" for tok in doc)
            if has_noun and has_verb:
                complete_sent_count += 1

        return {
            "avg_sentence_length": np.mean(total_len),
            "avg_dependency_depth": np.mean(total_depth),
            "complete_sentence_ratio": complete_sent_count / len(sentences)
        }
