# Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the *N*-Back Paradigm
Code for ACL 2025 Findings paper [Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the *N*-Back Paradigm](https://arxiv.org/abs/2412.18120).

## Usage
Depending on the experiment, you will need
* Together AI API key
* OpenAI API key
* Hugging Face `use_auth_token`
```
$ pip install -r requirements.txt
$ python run_experiments.py --api_key_together 123 --api_key_openai 456 --use_auth_token 789
```

## Citation
```
@inproceedings{hu-lewis-2025-language,
    title = "Do Language Models Understand the Cognitive Tasks Given to Them? Investigations with the N-Back Paradigm",
    author = "Hu, Xiaoyang  and
      Lewis, Richard",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2025",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.findings-acl.136/",
    pages = "2665--2677",
    ISBN = "979-8-89176-256-5",
    abstract = "Cognitive tasks originally developed for humans are now increasingly used to study language models. While applying these tasks is often straightforward, interpreting their results can be challenging. In particular, when a model underperforms, it is often unclear whether this results from a limitation in the cognitive ability being tested or a failure to understand the task itself. A recent study argues that GPT 3.5{'}s declining performance on 2-back and 3-back tasks reflects a working memory capacity limit similar to humans (Gong et al., 2024). By analyzing a range of open-source language models of varying performance levels on these tasks, we show that the poor performance is due at least in part to a limitation in task comprehension and task set maintenance. We challenge the best-performing model with progressively harder versions of the task (up to 10-back) and experiment with alternative prompting strategies, before analyzing model attentions. Our larger aim is to contribute to the ongoing conversation around refining methodologies for the cognitive evaluation of language models."
}
```
