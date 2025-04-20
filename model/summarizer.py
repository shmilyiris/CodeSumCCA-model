import torch
import torch.nn as nn
from transformers import T5Tokenizer, T5ForConditionalGeneration


class CodeSummaryModel(nn.Module):
    def __init__(self, model_name='t5-small'):
        super(CodeSummaryModel, self).__init__()
        self.model_name = model_name
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def preprocess(self, sample):
        """
        将结构化 JSON 数据拼接为结构化文本输入，适用于 encoder-decoder 架构。
        """
        class_info = f"ClassName: {sample['className']}; Modifiers: {' '.join(sample['classModifiers'])}; Package: {sample['packageName']}"
        imports = f"Imports: {', '.join(sample['importList'])}" if sample.get('importList') else ""

        fields = "Fields: " + "; ".join([
            f"{' '.join(f['fieldModifiers'])}{f['fieldType']} {f['fieldName']}"
            for f in sample.get("fieldInfoList", [])
        ]) if sample.get("fieldInfoList") else ""

        methods = "Methods: " + "; ".join([
            f"{' '.join(m['methodModifiers'])}{m['methodReturnType']} {m['methodName']}(); Desc: {m['methodDesc']}"
            for m in sample.get("methodInfoList", [])
        ]) if sample.get("methodInfoList") else ""

        full_input = f"{class_info}. {imports}. {fields}. {methods}"
        return full_input

    def forward(self, input_texts, target_texts=None, max_input_length=512, max_target_length=64):
        inputs = self.tokenizer(
            input_texts,
            max_length=max_input_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        if target_texts is not None:
            targets = self.tokenizer(
                target_texts,
                max_length=max_target_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            outputs = self.model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                labels=targets.input_ids
            )
            return outputs
        else:
            summary_ids = self.model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_target_length
            )
            return self.tokenizer.batch_decode(summary_ids, skip_special_tokens=True)


if __name__ == '__main__':
    model = CodeSummaryModel()
    # 示例用法：
    sample = {
        "className": "AbstractAddressResolver",
        "classModifiers": ["public", "abstract"],
        "packageName": "io.netty.resolver",
        "importList": ["java.net.SocketAddress", "java.util.List"],
        "fieldInfoList": [
            {"fieldModifiers": ["private", "final"], "fieldType": "EventExecutor", "fieldName": "executor"}
        ],
        "methodInfoList": [
            {
                "methodModifiers": ["protected"],
                "methodReturnType": "EventExecutor",
                "methodName": "executor",
                "methodDesc": "Returns the EventExecutor used to notify listeners"
            }
        ]
    }
    input_text = model.preprocess(sample)
    print("[Model Input]:", input_text)
    output = model([input_text])
    print("[Generated Summary]:", output[0])
