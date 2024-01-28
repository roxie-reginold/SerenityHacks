import langchain_openai as lch_openai
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

OPENAI_API_KEY = "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"

def generate_result(emotion, scale):
    llm = lch_openai.OpenAI(api_key=OPENAI_API_KEY, temperature=0.7)

    prompt_template = PromptTemplate(
        input_variables=['emotion', 'scale'],
        template="I am feeling {emotion}, at a level {scale} on a scale of 1-10. Give me activity suggestions."
    )

    name_chain = LLMChain(llm=llm, prompt=prompt_template)

    response = name_chain({'emotion': emotion, 'scale': int(scale)}) 
    return response

if __name__ == "__main__":
    print(generate_result("sad", "9"))
