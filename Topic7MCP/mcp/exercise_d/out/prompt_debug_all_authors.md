# Exercise D Prompt Debug

- Paper ID: `ARXIV:2210.03629`
- Output file: `Topic7MCP/mcp/exercise_d/out/prompt_debug_all_authors.md`
- Author limit mode: `all`
- System prompt characters: `512`
- User prompt characters: `27585`
- Combined prompt characters: `28099`
- Combined prompt bytes (UTF-8): `28099`
- Approximate combined tokens: `7024`

## System Message

```text
You are generating a structured markdown research report from retrieved paper data. This is not a chat. Write only markdown. Use the provided evidence to produce: 1) a one-paragraph summary of the seed paper, 2) a 'Foundational Works' section with exactly the provided key references, 3) a 'Recent Developments' section with exactly the provided citing papers, and 4) an 'Author Profiles' section with the provided notable works. If any retrieved data is missing, state that explicitly and do not invent details.
```

## User Message

```text
Generate the citation neighborhood report in markdown.
The 'Foundational Works' section must contain exactly 5 entries from the payload.
The 'Recent Developments' section must contain exactly 5 entries from the payload.
The 'Author Profiles' section must contain exactly 7 entries from the payload and no extra authors.

Retrieved research payload:
```json
{
  "paper_id": "ARXIV:2210.03629",
  "generated_on": "2026-03-10",
  "recent_citations_start": "2023-03-11:",
  "author_profile_count": 7,
  "seed_paper": {
    "title": "ReAct: Synergizing Reasoning and Acting in Language Models",
    "year": 2022,
    "publicationDate": "2022-10-06",
    "venue": "International Conference on Learning Representations",
    "url": "https://www.semanticscholar.org/paper/99832586d55f540f603637e458a292406a0ed75d",
    "fieldsOfStudy": [
      "Computer Science"
    ],
    "abstract": "While large language models (LLMs) have demonstrated impressive capabilities across tasks in language understanding and interactive decision making, their abilities for reasoning (e.g. chain-of-thought prompting) and acting (e.g. action plan generation) have primarily been studied as separate topics. In this paper, we explore the use of LLMs to generate both reasoning traces and task-specific actions in an interleaved manner, allowing for greater synergy between the two: reasoning traces help the model induce, track, and update action plans as well as handle exceptions, while actions allow it to interface with external sources, such as knowledge bases or environments, to gather additional information. We apply our approach, named ReAct, to a diverse set of language and decision making tasks and demonstrate its effectiveness over state-of-the-art baselines, as well as improved human interpretability and trustworthiness over methods without reasoning or acting components. Concretely, on question answering (HotpotQA) and fact verification (Fever), ReAct overcomes issues of hallucination and error propagation prevalent in chain-of-thought reasoning by interacting with a simple Wikipedia API, and generates human-like task-solving trajectories that are more interpretable than baselines without reasoning traces. On two interactive decision making benchmarks (ALFWorld and WebShop), ReAct outperforms imitation and reinforcement learning methods by an absolute success rate of 34% and 10% respectively, while being prompted with only one or two in-context examples. Projec [truncated]",
    "authors": [
      "Shunyu Yao",
      "Jeffrey Zhao",
      "Dian Yu",
      "Nan Du"
    ],
    "additionalAuthorCount": 3
  },
  "foundational_works": [
    {
      "title": "Language Models are Few-Shot Learners",
      "year": 2020,
      "publicationDate": "2020-05-28",
      "venue": "Neural Information Processing Systems",
      "url": "https://www.semanticscholar.org/paper/90abbc2cf38462b954ae1b772fac9532e2ccd8b0",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "Recent work has demonstrated substantial gains on many NLP tasks and benchmarks by pre-training on a large corpus of text followed by fine-tuning on a specific task. While typically task-agnostic in architecture, this method still requires task-specific fine-tuning datasets of thousands or tens of thousands of examples. By contrast, humans can generally perform a new language task from only a few examples or from simple instructions - something which current NLP systems still largely struggle to do. Here we show that scaling up language models greatly improves task-agnostic, few-shot performance, sometimes even reaching competitiveness with prior state-of-the-art fine-tuning approaches. Specifically, we train GPT-3, an autoregressive language model with 175 billion parameters, 10x more than any previous non-sparse language model, and test its performance in the few-shot setting. For all tasks, GPT-3 is applied without any gradient updates or fine-tuning, with tasks and few-shot demonstrations specified purely via text interaction with the model. GPT-3 achieves strong performance on many NLP datasets, including translation, question-answering, and cloze tasks, as well as several tasks that require on-the-fly reasoning or domain adaptation, such as unscrambling words, using a novel word in a sentence, or performing 3-digit arithmetic. At the same time, we also identify some datasets where GPT-3's few-shot learning still struggles, as well as some datasets where GPT-3 faces methodological issues related to training on large web corpora. Finally, we find that GPT- [truncated]",
      "authors": [
        "Tom B. Brown",
        "Benjamin Mann",
        "Nick Ryder",
        "Melanie Subbiah"
      ],
      "additionalAuthorCount": 27
    },
    {
      "title": "Improving alignment of dialogue agents via targeted human judgements",
      "year": 2022,
      "publicationDate": "2022-09-28",
      "venue": "arXiv.org",
      "url": "https://www.semanticscholar.org/paper/74eae12620bd1c1393e268bddcb6f129a5025166",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "We present Sparrow, an information-seeking dialogue agent trained to be more helpful, correct, and harmless compared to prompted language model baselines. We use reinforcement learning from human feedback to train our models with two new additions to help human raters judge agent behaviour. First, to make our agent more helpful and harmless, we break down the requirements for good dialogue into natural language rules the agent should follow, and ask raters about each rule separately. We demonstrate that this breakdown enables us to collect more targeted human judgements of agent behaviour and allows for more efficient rule-conditional reward models. Second, our agent provides evidence from sources supporting factual claims when collecting preference judgements over model statements. For factual questions, evidence provided by Sparrow supports the sampled response 78% of the time. Sparrow is preferred more often than baselines while being more resilient to adversarial probing by humans, violating our rules only 8% of the time when probed. Finally, we conduct extensive analyses showing that though our model learns to follow our rules it can exhibit distributional biases.",
      "authors": [
        "Amelia Glaese",
        "Nat McAleese",
        "Maja Trkebacz",
        "John Aslanides"
      ],
      "additionalAuthorCount": 30
    },
    {
      "title": "Text and Patterns: For Effective Chain of Thought, It Takes Two to Tango",
      "year": 2022,
      "publicationDate": "2022-09-16",
      "venue": "arXiv.org",
      "url": "https://www.semanticscholar.org/paper/4988b3d378b79eb8669112620baf1ff4e3e536fd",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "The past decade has witnessed dramatic gains in natural language processing and an unprecedented scaling of large language models. These developments have been accelerated by the advent of few-shot techniques such as chain of thought (CoT) prompting. Specifically, CoT pushes the performance of large language models in a few-shot setup by augmenting the prompts with intermediate steps. Despite impressive results across various tasks, the reasons behind their success have not been explored. This work uses counterfactual prompting to develop a deeper understanding of CoT-based few-shot prompting mechanisms in large language models. We first systematically identify and define the key components of a prompt: symbols, patterns, and text. Then, we devise and conduct an exhaustive set of experiments across four different tasks, by querying the model with counterfactual prompts where only one of these components is altered. Our experiments across three models (PaLM, GPT-3, and CODEX) reveal several surprising findings and brings into question the conventional wisdom around few-shot prompting. First, the presence of factual patterns in a prompt is practically immaterial to the success of CoT. Second, our results conclude that the primary role of intermediate steps may not be to facilitate learning how to solve a task. The intermediate steps are rather a beacon for the model to realize what symbols to replicate in the output to form a factual answer. Further, text imbues patterns with commonsense knowledge and meaning. Our empirical and qualitative analysis reveals that [truncated]",
      "authors": [
        "Aman Madaan",
        "A. Yazdanbakhsh"
      ]
    },
    {
      "title": "Faithful Reasoning Using Large Language Models",
      "year": 2022,
      "publicationDate": "2022-08-30",
      "venue": "arXiv.org",
      "url": "https://www.semanticscholar.org/paper/f0a0e8b6e84207f50db4d24cc4016e40601214ef",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "Although contemporary large language models (LMs) demonstrate impressive question-answering capabilities, their answers are typically the product of a single call to the model. This entails an unwelcome degree of opacity and compromises performance, especially on problems that are inherently multi-step. To address these limitations, we show how LMs can be made to perform faithful multi-step reasoning via a process whose causal structure mirrors the underlying logical structure of the problem. Our approach works by chaining together reasoning steps, where each step results from calls to two fine-tuned LMs, one for selection and one for inference, to produce a valid reasoning trace. Our method carries out a beam search through the space of reasoning traces to improve reasoning quality. We demonstrate the effectiveness of our model on multi-step logical deduction and scientific question-answering, showing that it outperforms baselines on final answer accuracy, and generates humanly interpretable reasoning traces whose validity can be checked by the user.",
      "authors": [
        "Antonia Creswell",
        "M. Shanahan"
      ]
    },
    {
      "title": "BlenderBot 3: a deployed conversational agent that continually learns to responsibly engage",
      "year": 2022,
      "publicationDate": "2022-08-05",
      "venue": "arXiv.org",
      "url": "https://www.semanticscholar.org/paper/a3076ecfed0571fbbb5217a5cc6b4b6f24f6f7dd",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "We present BlenderBot 3, a 175B parameter dialogue model capable of open-domain conversation with access to the internet and a long-term memory, and having been trained on a large number of user defined tasks. We release both the model weights and code, and have also deployed the model on a public web page to interact with organic users. This technical report describes how the model was built (architecture, model and training scheme), and details of its deployment, including safety mechanisms. Human evaluations show its superiority to existing open-domain dialogue agents, including its predecessors (Roller et al., 2021; Komeili et al., 2022). Finally, we detail our plan for continual learning using the data collected from deployment, which will also be publicly released. The goal of this research program is thus to enable the community to study ever-improving responsible agents that learn through interaction.",
      "authors": [
        "Kurt Shuster",
        "Jing Xu",
        "M. Komeili",
        "Da Ju"
      ],
      "additionalAuthorCount": 14
    }
  ],
  "recent_developments": [
    {
      "title": "Enhancing large language models for knowledge graph question answering via multi-granularity knowledge injection and structured reasoning path-augmented prompting",
      "year": 2026,
      "publicationDate": "2026-06-01",
      "venue": "Information Processing & Management",
      "url": "https://www.semanticscholar.org/paper/a06d2796211ff8f39ac052dc06e502264b64b621",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "",
      "authors": [
        "Chuanyang Gong",
        "Zhihua Wei",
        "Wenhao Tao",
        "Duoqian Miao"
      ],
      "missing_data_note": "Abstract unavailable in MCP response."
    },
    {
      "title": "Neuro-symbolic agentic AI: Architectures, integration patterns, applications, open challenges and future research directions",
      "year": 2026,
      "publicationDate": "2026-05-01",
      "venue": "Computer Science Review",
      "url": "https://www.semanticscholar.org/paper/c3f8fcc95ba68d25bc1dd16defa361651563f6c6",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "",
      "authors": [
        "Safayat Bin Hakim",
        "Muhammad Adil",
        "Alvaro Velasquez",
        "H. Song"
      ],
      "missing_data_note": "Abstract unavailable in MCP response."
    },
    {
      "title": "ChatPRE: Knowledge-aware protocol analysis with LLMs for intelligent segmentation",
      "year": 2026,
      "publicationDate": "2026-05-01",
      "venue": "Journal of Network and Computer Applications",
      "url": "https://www.semanticscholar.org/paper/6325b547d8412c98fb2915786acbb57229b33dbe",
      "fieldsOfStudy": [],
      "abstract": "",
      "authors": [
        "Guoyu Huo",
        "Yuyao Huang",
        "Fei Kang",
        "Hui Shu"
      ],
      "missing_data_note": "Abstract unavailable in MCP response."
    },
    {
      "title": "INKER: Adaptive dynamic retrieval augmented generation with internal-external knowledge integration",
      "year": 2026,
      "publicationDate": "2026-04-01",
      "venue": "Information Processing & Management",
      "url": "https://www.semanticscholar.org/paper/67db726c3d3bd292c17a528943b6edc33e2eff61",
      "fieldsOfStudy": [
        "Computer Science"
      ],
      "abstract": "",
      "authors": [
        "Mingjun Zhou",
        "Jiuyang Tang",
        "Weixin Zeng",
        "Xiang Zhao"
      ],
      "missing_data_note": "Abstract unavailable in MCP response."
    },
    {
      "title": "Vision-Language Model-Driven Human-Vehicle Interaction for Autonomous Driving: Status, Challenge, and Innovation",
      "year": 2026,
      "publicationDate": "2026-04-01",
      "venue": "Big Data Mining and Analytics",
      "url": "https://www.semanticscholar.org/paper/908919c745e35a30966d36dc93a38c74a43bd8c6",
      "fieldsOfStudy": [],
      "abstract": "",
      "authors": [
        "Rongfeng Zhao",
        "Aimin Du",
        "Mobing Cai",
        "Zhongpan Zhu"
      ],
      "additionalAuthorCount": 1,
      "missing_data_note": "Abstract unavailable in MCP response."
    }
  ],
  "author_profiles": [
    {
      "author": {
        "name": "Shunyu Yao",
        "authorId": "2093302161"
      },
      "notable_work": {
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "year": 2023,
        "publicationDate": "2023-05-17",
        "venue": "Neural Information Processing Systems",
        "url": "https://www.semanticscholar.org/paper/2f3822eb380b5e753a6d579f31dfc3ec4c4a0820",
        "fieldsOfStudy": [
          "Computer Science"
        ],
        "abstract": "Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices. Our experiments show that ToT significantly enhances language models' problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4% of tasks, our method achieved a success rate of 74%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.",
        "authors": [
          "Shunyu Yao",
          "Dian Yu",
          "Jeffrey Zhao",
          "Izhak Shafran"
        ],
        "additionalAuthorCount": 3
      }
    },
    {
      "author": {
        "name": "Jeffrey Zhao",
        "authorId": "2144551262"
      },
      "notable_work": {
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "year": 2023,
        "publicationDate": "2023-05-17",
        "venue": "Neural Information Processing Systems",
        "url": "https://www.semanticscholar.org/paper/2f3822eb380b5e753a6d579f31dfc3ec4c4a0820",
        "fieldsOfStudy": [
          "Computer Science"
        ],
        "abstract": "Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices. Our experiments show that ToT significantly enhances language models' problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4% of tasks, our method achieved a success rate of 74%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.",
        "authors": [
          "Shunyu Yao",
          "Dian Yu",
          "Jeffrey Zhao",
          "Izhak Shafran"
        ],
        "additionalAuthorCount": 3
      }
    },
    {
      "author": {
        "name": "Dian Yu",
        "authorId": "150978762"
      },
      "notable_work": {
        "title": "BLOOM: A 176B-Parameter Open-Access Multilingual Language Model",
        "year": 2022,
        "publicationDate": "2022-11-09",
        "venue": "arXiv.org",
        "url": "https://www.semanticscholar.org/paper/964bd39b546f0f6625ff3b9ef1083f797807ef2e",
        "fieldsOfStudy": [
          "Computer Science"
        ],
        "abstract": "Large language models (LLMs) have been shown to be able to perform new tasks based on a few demonstrations or natural language instructions. While these capabilities have led to widespread adoption, most LLMs are developed by resource-rich organizations and are frequently kept from the public. As a step towards democratizing this powerful technology, we present BLOOM, a 176B-parameter open-access language model designed and built thanks to a collaboration of hundreds of researchers. BLOOM is a decoder-only Transformer language model that was trained on the ROOTS corpus, a dataset comprising hundreds of sources in 46 natural and 13 programming languages (59 in total). We find that BLOOM achieves competitive performance on a wide variety of benchmarks, with stronger results after undergoing multitask prompted finetuning. To facilitate future research and applications using LLMs, we publicly release our models and code under the Responsible AI License.",
        "authors": [
          "Teven Le Scao",
          "Angela Fan",
          "Christopher Akiki",
          "Ellie Pavlick"
        ],
        "additionalAuthorCount": 387
      }
    },
    {
      "author": {
        "name": "Nan Du",
        "authorId": "2140321952"
      },
      "notable_work": {
        "title": "PaLM: Scaling Language Modeling with Pathways",
        "year": 2022,
        "publicationDate": "2022-04-05",
        "venue": "Journal of machine learning research",
        "url": "https://www.semanticscholar.org/paper/094ff971d6a8b8ff870946c9b3ce5aa173617bfb",
        "fieldsOfStudy": [
          "Computer Science"
        ],
        "abstract": "Large language models have been shown to achieve remarkable performance across a variety of natural language tasks using few-shot learning, which drastically reduces the number of task-specific training examples needed to adapt the model to a particular application. To further our understanding of the impact of scale on few-shot learning, we trained a 540-billion parameter, densely activated, Transformer language model, which we call Pathways Language Model PaLM. We trained PaLM on 6144 TPU v4 chips using Pathways, a new ML system which enables highly efficient training across multiple TPU Pods. We demonstrate continued benefits of scaling by achieving state-of-the-art few-shot learning results on hundreds of language understanding and generation benchmarks. On a number of these tasks, PaLM 540B achieves breakthrough performance, outperforming the finetuned state-of-the-art on a suite of multi-step reasoning tasks, and outperforming average human performance on the recently released BIG-bench benchmark. A significant number of BIG-bench tasks showed discontinuous improvements from model scale, meaning that performance steeply increased as we scaled to our largest model. PaLM also has strong capabilities in multilingual tasks and source code generation, which we demonstrate on a wide array of benchmarks. We additionally provide a comprehensive analysis on bias and toxicity, and study the extent of training data memorization with respect to model scale. Finally, we discuss the ethical considerations related to large language models and discuss potential mitigation strategies.",
        "authors": [
          "A. Chowdhery",
          "Sharan Narang",
          "Jacob Devlin",
          "Maarten Bosma"
        ],
        "additionalAuthorCount": 63
      }
    },
    {
      "author": {
        "name": "Izhak Shafran",
        "authorId": "1697494"
      },
      "notable_work": {
        "title": "Tree of Thoughts: Deliberate Problem Solving with Large Language Models",
        "year": 2023,
        "publicationDate": "2023-05-17",
        "venue": "Neural Information Processing Systems",
        "url": "https://www.semanticscholar.org/paper/2f3822eb380b5e753a6d579f31dfc3ec4c4a0820",
        "fieldsOfStudy": [
          "Computer Science"
        ],
        "abstract": "Language models are increasingly being deployed for general problem solving across a wide range of tasks, but are still confined to token-level, left-to-right decision-making processes during inference. This means they can fall short in tasks that require exploration, strategic lookahead, or where initial decisions play a pivotal role. To surmount these challenges, we introduce a new framework for language model inference, Tree of Thoughts (ToT), which generalizes over the popular Chain of Thought approach to prompting language models, and enables exploration over coherent units of text (thoughts) that serve as intermediate steps toward problem solving. ToT allows LMs to perform deliberate decision making by considering multiple different reasoning paths and self-evaluating choices to decide the next course of action, as well as looking ahead or backtracking when necessary to make global choices. Our experiments show that ToT significantly enhances language models' problem-solving abilities on three novel tasks requiring non-trivial planning or search: Game of 24, Creative Writing, and Mini Crosswords. For instance, in Game of 24, while GPT-4 with chain-of-thought prompting only solved 4% of tasks, our method achieved a success rate of 74%. Code repo with all prompts: https://github.com/princeton-nlp/tree-of-thought-llm.",
        "authors": [
          "Shunyu Yao",
          "Dian Yu",
          "Jeffrey Zhao",
          "Izhak Shafran"
        ],
        "additionalAuthorCount": 3
      }
    },
    {
      "author": {
        "name": "Karthik Narasimhan",
        "authorId": "144958935"
      },
      "notable_work": {
        "title": "Improving Language Understanding by Generative Pre-Training",
        "year": 2018,
        "publicationDate": null,
        "venue": "",
        "url": "https://www.semanticscholar.org/paper/cd18800a0fe0b668a1cc19f2ec95b5003d0a5035",
        "fieldsOfStudy": [
          "Computer Science"
        ],
        "abstract": "",
        "authors": [
          "Alec Radford",
          "Karthik Narasimhan"
        ],
        "missing_data_note": "Abstract unavailable in MCP response."
      }
    },
    {
      "author": {
        "name": "Yuan Cao",
        "authorId": "145144022"
      },
      "notable_work": {
        "title": "Google's Neural Machine Translation System: Bridging the Gap between Human and Machine Translation",
        "year": 2016,
        "publicationDate": "2016-09-26",
        "venue": "arXiv.org",
        "url": "https://www.semanticscholar.org/paper/c6850869aa5e78a107c378d2e8bfa39633158c0c",
        "fieldsOfStudy": [
          "Computer Science"
        ],
        "abstract": "Neural Machine Translation (NMT) is an end-to-end learning approach for automated translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Unfortunately, NMT systems are known to be computationally expensive both in training and in translation inference. Also, most NMT systems have difficulty with rare words. These issues have hindered NMT's use in practical deployments and services, where both accuracy and speed are essential. In this work, we present GNMT, Google's Neural Machine Translation system, which attempts to address many of these issues. Our model consists of a deep LSTM network with 8 encoder and 8 decoder layers using attention and residual connections. To improve parallelism and therefore decrease training time, our attention mechanism connects the bottom layer of the decoder to the top layer of the encoder. To accelerate the final translation speed, we employ low-precision arithmetic during inference computations. To improve handling of rare words, we divide words into a limited set of common sub-word units (\"wordpieces\") for both input and output. This method provides a good balance between the flexibility of \"character\"-delimited models and the efficiency of \"word\"-delimited models, naturally handles translation of rare words, and ultimately improves the overall accuracy of the system. Our beam search technique employs a length-normalization procedure and uses a coverage penalty, which encourages generation of an output sentence that is most likely to cover all the words in the source [truncated]",
        "authors": [
          "Yonghui Wu",
          "M. Schuster",
          "Z. Chen",
          "Quoc V. Le"
        ],
        "additionalAuthorCount": 27
      }
    }
  ]
}
```
```
