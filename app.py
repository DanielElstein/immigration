import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import pinecone

st.set_page_config(page_title="Immigration Q&A", layout="wide", initial_sidebar_state="expanded")

st.header("Immigration Q&A")


custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background-color: #ADD8E6;
    }
    .anchor svg {
        display: none;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Create two columns: one for the image and one for the text
col1, col2 = st.columns([1, 3])

# Display the image in the left column
image_path = 'statue.png'
col1.image(image_path, width=350)

# Display the text in the right column
col2.markdown("""
Ask a question about immigration to the United States in any language, and our artificial intelligence will answer based on the [USCIS policy manual](https://www.uscis.gov/policy-manual).

Haga una pregunta sobre inmigración a los Estados Unidos en cualquier idioma y nuestra inteligencia artificial responderá según el manual del USCIS.

用任何语言提问有关美国移民的问题，我们的人工智能将根据美国公民及移民服务局的手册进行回答。

Magtanong tungkol sa imigrasyon sa Estados Unidos sa anumang wika, at sasagutin ng aming artificial intelligence batay sa USCIS handbook.

Hỏi về nhập cư vào Hoa Kỳ bằng bất kỳ ngôn ngữ nào, và trí thông minh nhân tạo của chúng tôi sẽ trả lời dựa trên sách hướng dẫn của USCIS.

Báa ìbéèrè nípa ìjìmìn-ìlú sí Amẹ́ríkà ní èdè kankan, kí àṣà ìmọ̀ ọ̀rọ̀ wa ṣe àpèsè láti ìwé kikọ USCIS.

*Legal Disclaimer: This platform is meant for informational purposes only. It is not affiliated with USCIS or any other governmental organization, and is not a substitute for professional legal advice. The answers provided are based on the USCIS policy manual and may not cover all aspects of your specific situation. For personalized guidance, please consult an immigration attorney.*

""")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "langchaintest2"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Create a form
with st.form(key="my_form"):
    query = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Submit")

conversation_text = ""

if submit_button:
    template = """
    System: Play the role of a friendly immigration lawyer. Respond to questions in detail, in the same language as the human's most recent question. If they ask a question in Spanish, you should answer in Spanish. If they ask a question in French, you should answer in French. And so on, for every language.
    
    Human: ¿Cómo puedo obtener una visa para ingresar a los Estados Unidos?
    
    Lawyer: Si busca una visa para ingresar a EE. UU. y reside fuera del país, podría necesitar una entrevista. Todos los no ciudadanos deben ser inspeccionados y admitidos o en libertad condicional. Puede presentar una solicitud de naturalización junto con el Formulario I-131 sin costo para solicitar libertad condicional por razones humanitarias o de beneficio público. USCIS coordinará la fecha y lugar de la entrevista. Respondo preguntas en inglés, avíseme si necesita más ayuda en ese idioma.
    
    Human: How can I get a visa to the United States?
    
    Lawyer: If you are seeking a visa to enter the United States, you may need to appear for an interview in the United States if you reside outside the country and have separated from the military. All noncitizens must be inspected and admitted or paroled in order to enter the United States. If you are seeking parole into the United States, you may file a naturalization application concurrently with an Application for Travel Document (Form I-131) without a fee to seek an advance parole document for a humanitarian or significant public benefit parole before entering the United States, if necessary. USCIS will coordinate with you to schedule an interview date and location. 
    
    {conversation_text}
    
    Human: {query}

    Lawyer: """

    if query:
        prompt = template.format(query=query, conversation_text=conversation_text)
        docs = docsearch.similarity_search(query, include_metadata=True)

        from langchain.llms import OpenAI
        from langchain.chains.question_answering import load_qa_chain

        llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
        conversation = ConversationChain(
            llm=llm, verbose=True, memory=ConversationBufferMemory()
        )
        chain = load_qa_chain(llm, chain_type="stuff")

        with st.spinner('Processing your question...'):
            result = conversation.predict(input=prompt)

        st.header("Answer")
        st.write(result)
        conversation.memory.push(prompt, result)
        conversation_text += f"Human: {query}\n\n"
        conversation_text += f"Lawyer: {result}\n\n"
