cola = {
    "description": "template used by GLUE-CoLA",
    "input_text": "Indicate if the following sentence is grammatically correct or not: \"{sentence}\". Answere 'acceptable' or 'unacceptable'.",
    "target_text": {"0": "unacceptable", "1": "acceptable"},
}

mnli = {
    "input_text": "Does the premise: '{premise}' logically imply, contradict, or is neutral to the hypothesis: '{hypothesis}'? Answere with 'entailment', 'contradiction', or 'neutral'.",
    "target_text": {"0": "entailment", "1": "neutral", "2": "contradiction"},
}

mrpc = {
    "input_text": "Are the following sentences '{sentence1}' and '{sentence2}' conveying the same meaning? Answere with 'yes' or 'no'.",
    "target_text": {"0": "no", "1": "yes"},
}

qnli = {
    "input_text": "Given the context: '{sentence}', does the question '{question}' have an answer based on the information provided? Answer with 'yes' or 'no'.",
    "target_text": {"0": "yes", "1": "no"},
}

qqp = {
    "input_text": "Do the questions '{question1}' and '{question2}' have the same intent? Answere with 'yes' or 'no'.",
    "target_text": {"0": "no", "1": "yes"},
}

rte = {
    "description": "Template used by GLUE-RTE",
    "input_text": "Does the text: '{sentence1}' entail that '{sentence2}' is true? Provide 'yes' or 'no'.",
    "target_text": {"0": "yes", "1": "no"},
}

sst2 = {
    "input_text": "Given the sentence '{sentence}', determine the sentiment. Is it positive or negative?",
    "target_text": {"0": "negative", "1": "positive"},
}

stsb = {
    "input_text": "Consider the sentences '{sentence1}' and '{sentence2}'. On a scale from 1 (completely different) to 5 (completely similar), rate the similarity.",
    "target_text": "{:.1f}",
}

glue_prompt_templates = {
    "cola": cola,
    "mnli": mnli,
    "mrpc": mrpc,
    "qnli": qnli,
    "qqp": qqp,
    "rte": rte,
    "stsb": stsb,
    "sst2": sst2,
}
