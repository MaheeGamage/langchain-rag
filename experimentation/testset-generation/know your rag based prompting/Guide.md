Guide: Statement Extraction Strategy for RAG Dataset GenerationThis guide outlines a step-by-step "statement extraction" process for generating grounded question-answer (Q&A) pairs from a source context, as detailed in Know Your RAG: Dataset Taxonomy and Generation Strategies for Evaluating RAG Systems.This inverted generation process—creating statements (answers) first, then generating questions that map to those statements—helps reduce language model hallucinations and ensures high-quality evaluation data.

Step 1: Theme IdentificationBefore extracting facts, summarize the input context into a single theme. This theme will be injected into subsequent prompts to provide necessary contextualization, preventing the generation of overly generic questions.Prompt:
```
In a few words, extract the main theme behind the following passage: 
[[{context}]]
```

Step 2: Factual Statement ExtractionExtract foundational facts directly from the context. These facts should be atomized units of information that read like direct answers.Prompt:
```
Extract at most five factual statements based on the following passage and its theme. You need to strictly comply with the following guidelines:

* Each statement must be written in the style of an answer to a factual question.
* Each statement must be understandable without the aid of any other source of information.
* Each statement must include contextual information derived from the passage theme.
* Each statement must only contain information that exists in the original passage and theme.
* Each statement must be independent from the other statements.

Generate the statements as a bullet list with the following format:
> Statement
> Statement
etc

Theme: [[{theme}]]
Passage: [[{context}]]
```

Step 3 (Optional): Transforming Statements for Complex QuestionsIf you only want basic fact-based (fact_single) questions, skip to Step 4. If you want to evaluate your system's ability to handle complex queries, you must transform the factual statements into either Summary or Reasoning statements.

Option 3A: Generating Summary StatementsSummary questions ask for multiple pieces of information or a composite overview. Use the factual statements generated from Step 2 ({statements}) and the {theme}.Prompt:
```
Merge the following sentences into three summary statements. 

* Each summary statement must summarise information contained in more than one sentence.
* Each summary statement must be independent and non-overlapping.
* Each summary statement should be a complete sentence.
* Each summary statement can include contextual information contained in the theme below.
* Each summary statement must be understandable without the aid of any other source of information.

Generate the statements as a bullet list with the following format:
> Summary statement
> Summary statement
> Summary statement

Theme: [[{theme}]]
Sentences: [[{statements}]]
```

Option 3B: Generating Reasoning StatementsReasoning questions require inferred conclusions that are logically derived from the text, rather than explicitly stated.Prompt:
```
A reasoning conclusion is an inferred piece of information obtained from critically analysing a group of multiple statements. Reasoning conclusions do not contain information directly contained on any statements.

Generate three reasoning conclusions that can be drawn from the following statements.

* Each conclusion must be independent and non-overlapping.
* Each conclusion should be a complete sentence.
* Each conclusion must be understandable without the aid of any other source of information.
* Each conclusion can include contextual information contained in the theme below.

Generate the conclusions as a bullet list with the following format:
> conclusion
> conclusion
> conclusion
etc

Theme: [[{theme}]]
Statements: [[{statements}]]
```

Step 4: Question GenerationFinally, choose one statement generated from the previous steps (whether it is a factual, summary, or reasoning statement). Use this statement as the definitive "answer" to generate the final question.Prompt:
```
I have a paragraph with the following theme:
[[{theme}]]

From this paragraph, I extracted the following statement:
[[{statement}]]

Generate one question which is answered only by the statement above.

In order to avoid generic questions, use contextual information from the theme to formulate the question.

The question should be concise and in the style of a user asking questions to a search engine.

Generate the question as a bullet list with the following format:
> Question

Do not output anything else other than the question.
```