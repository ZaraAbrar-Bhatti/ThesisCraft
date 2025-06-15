from langchain_core.prompts import PromptTemplate

template = PromptTemplate(
    input_variables=["paper_input", "length_input", "style_input"],
    template="""
You are an expert research assistant. Your task is to summarize and analyze a research paper titled: "{paper_input}".

Please provide a {length_input} summary according to the given {style_input} including the following:
- A brief overview of the problem the paper addresses.
- The methodology used in the study (mention models, datasets, and techniques).
- Key findings or results.
- Code or algorithmic explanations, if mentioned (summarize in plain language).
- Limitations of the study.
- Real-world applications or implications.
- A concluding sentence summarizing the impact of paper.

Make the output clear, structured, and readable for someone with a technical background but who hasn’t read the paper.
"""
)

template.save('template.json')
