EXPERIMENTAL_CONFIG = {
    "humaneval": {
        "question_type": "code",
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"]
    },
    "mbpp": {
        "question_type": "code",
        "operators": ["Custom", "CustomCodeGenerate", "Test", "ScEnsemble"]
    },
    "math": {
        "question_type": "math",
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    },
    "gsm8k": {
        "question_type": "math",
        "operators": ["Custom", "ScEnsemble", "Programmer"]
    },
    "drop": {
        "question_type": "qa",
        "operators": ["Custom", "AnswerGenerate", "ScEnsemble"]
    },
    "hotpotqa": {
        "question_type": "qa",
        "operators": ["Custom", "AnswerGenerate", "ScEnsemble"]
    }
}