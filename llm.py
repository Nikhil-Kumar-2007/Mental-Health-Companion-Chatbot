from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field, validator
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os



summary = []
recent_chats = []
#load_dotenv()
token = os.getenv("HUGGINGFACEHUB_API_TOKEN")




output_llm = HuggingFaceEndpoint(
    model = "zai-org/GLM-4.7",
    task = "text-generation",
    huggingfacehub_api_token = token,
    provider = "novita",
    temperature = 1.5
)
output_model = ChatHuggingFace(llm = output_llm)

summary_llm = HuggingFaceEndpoint(
    model = "zai-org/GLM-4.7",
    task = "text-generation",
    huggingfacehub_api_token = token,
    provider = "novita",
    temperature = 0.5
)
summary_model = ChatHuggingFace(llm = summary_llm)


str_parser = StrOutputParser()

chat_template_output = ChatPromptTemplate([
    ("system" , '''As a compassionate AI companion, use reflective listening and highlight user achievements without toxic positivity. Redirect medical queries to doctors. For any non-mental health topics—except for greetings and for subject or any specific talk reply in polite manner: 'I am a mental health companion chatbot. you can share mental health-related concerns with me.' In case of a crisis, immediately urge the user to talk to their parents, friends, or relatives.'''),

    MessagesPlaceholder(variable_name = "memory"),

    ("human" , "{user_msg}"),
])




class Schema(BaseModel):
    user : str = Field(description = "Only summarize messages where role == 'human'")
    assistant : str = Field(description = "Only summarize messages where role == 'AI'")



summary_parser = PydanticOutputParser(pydantic_object = Schema)

chat_template_summary = ChatPromptTemplate([
    ("system" , '''You are a memory summarization assistant. You will receive chat messages as (role, message) where role is "human" or "assistant". Summarize the two roles separately: first, concisely summarize all human messages, preserving the user’s intent, emotions, concerns, and goals without adding new information; then, concisely summarize all assistant messages, keeping only the key responses, tone, and guidance style without expanding advice. Do not mix roles.\n{format_instruction} and no extra text.
'''),
    ("human" , "{complete_chat}")
    ],

    partial_variables = {
        "format_instruction" : summary_parser.get_format_instructions()
    }
)




chain_output = chat_template_output | output_model | str_parser
chain_summary = chat_template_summary | summary_model | summary_parser



def ai_assistant_reply(user_msg):

    assistant_reply = chain_output.invoke({'user_msg' : user_msg, "memory" : summary + recent_chats})

    recent_chats.append(HumanMessage(content = user_msg))
    recent_chats.append(AIMessage(content = assistant_reply))


    if len(recent_chats) >= 4 and len(summary) != 0:
        summ = chain_summary.invoke({"complete_chat" : summary + recent_chats})
        summary.clear()
        summary.append(HumanMessage(content = summ['user']))
        summary.append(AIMessage(content = summ['assistant']))
        recent_chats.clear()

    return assistant_reply
