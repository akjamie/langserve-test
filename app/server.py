import os

import dotenv
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langchain.output_parsers import OutputFixingParser
from langchain_community.chat_models import ChatTongyi
# from langchain_community.chat_models import ChatTongyi
from langchain_core.output_parsers.pydantic import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts.chat import SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate
from langserve import add_routes

from app import QaRequest, QaResponse

dotenv.load_dotenv(dotenv_path="../.env")
app = FastAPI()


@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")


system_message_prompt = SystemMessagePromptTemplate.from_template("""
    You are a helpful, professional assistant named Cob, you will answer every query from user.
""")

human_message_prompt = HumanMessagePromptTemplate.from_template("""
input: {user_input}
""")

# to user the pydantic output parser
parser = PydanticOutputParser(pydantic_object=QaResponse)
ai_message_prompt = AIMessagePromptTemplate.from_template(template="""Process the user's query and return result following with the output format instructions.
""", partial_variables={"format_instructions": parser.get_format_instructions()})

# Declare a chain
prompt = ChatPromptTemplate.from_messages([system_message_prompt, ai_message_prompt, human_message_prompt, ])

chat_llm = ChatTongyi(model_name="qwen-plus", temperature=0.5, dashscope_api_key=os.environ['DASHSCOPE_API_KEY'], )

# llm = HuggingFaceEndpoint(
#     # repo_id="Qwen/Qwen2-7B-Instruct",
#     repo_id="Qwen/Qwen2-1.5B-Instruct",
#     task="text-generation",
#     max_new_tokens=1024,
#     do_sample=False,
#     repetition_penalty=1.03,
# )
# chat_llm = ChatHuggingFace(llm=llm)

output_parser = OutputFixingParser.from_llm(parser=parser, llm=chat_llm)
# output_parser = RetryWithErrorOutputParser.from_llm(parser=parser, llm=chat_llm)
chain = prompt | chat_llm | output_parser
# chain = prompt | chat_llm

add_routes(
    app,
    chain.with_types(input_type=QaRequest, output_type=QaResponse),
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True,
    playground_type="default",
    path="/ask"
)

# @app.post("/ask", response_model=OutputChat)
# def ask_chat(input_chat: InputChat):
#     if isinstance(chat_llm, ChatHuggingFace):
#         os.environ["https_proxy"] = "http://127.0.0.1:7890"
#         os.environ["http_proxy"] = "http://127.0.0.1:7890"
#         os.environ["no_proxy"] = "localhost,127.0.0.1"
#     # Get the raw output from the final chain
#     raw_output = chain.invoke(input_chat.user_input)
#
#     # Use the output parser to process the output
#     # raw_output = output_parser.parse_with_prompt(raw_output,ai_message_prompt)
#
#     return raw_output


if __name__ == "__main__":
    from langchain.globals import set_debug

    set_debug(True)
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
