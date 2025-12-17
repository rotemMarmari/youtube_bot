from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter 
from langchain_community.vectorstores import FAISS  
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import LLMChain
from yt_functions import get_transcript

url = 'https://www.youtube.com/watch?v=SjvS20YqUJ4'


def process(transcript):
    '''
    process the transcript to extract text and start time
    '''
    txt = ""
    
    # Loop through each entry in the transcript
    for i in transcript:
        try:
            # Append the text and its start time to the output string
            txt += f"Text: {i.text} Start: {i.start}\n"
        except AttributeError:
            # If there is an issue accessing 'text' or 'start', skip this entry
            pass
            
    return txt


def chunk_transcript(processed_transcript, chunk_size=200, chunk_overlap=20):
    '''
    Chunk the processed transcript into smaller segments for embedding
    '''
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = text_splitter.split_text(processed_transcript)
    return chunks


def create_faiss_index(chunks, embedding_model):
    """
    Create a FAISS index from text chunks using the specified embedding model.
    
    :param chunks: List of text chunks
    :param embedding_model: The embedding model to use
    :return: FAISS index
    """

    return FAISS.from_texts(chunks, embedding_model)


embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)


def perform_similarity_search(faiss_index, query, k=3):
    """
    Search for specific queries within the embedded transcript using the FAISS index similarity search.
    
    :param faiss_index: The FAISS index containing embedded text chunks
    :param query: The text input for the similarity search
    :param k: The number of similar results to return (default is 3)
    :return: List of similar results
    """

    results = faiss_index.similarity_search(query, k=k)
    return results


def create_summary_prompt():
    """
    Create a PromptTemplate for summarizing a YouTube video transcript.
    
    :return: PromptTemplate object
    """

    template = """
    <|begin_of_text|><|start_header_id|>system<|end_header_id|>
    You are an AI assistant tasked with summarizing YouTube video transcripts. Provide concise, informative summaries that capture the main points of the video content.

    Instructions:
    1. Summarize the transcript in a single concise paragraph.
    2. Ignore any timestamps in your summary.
    3. Focus on the spoken content (Text) of the video.

    Note: In the transcript, "Text" refers to the spoken words in the video, and "start" indicates the timestamp when that part begins in the video.<|eot_id|><|start_header_id|>user<|end_header_id|>
    Please summarize the following YouTube video transcript:

    {transcript}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
    """
    
    prompt = PromptTemplate(
        input_variables=["transcript"],
        template=template
    )
    
    return prompt


def create_summary_chain(llm, prompt, verbose=True):
    """
    Create an LLMChain for generating summaries.
    
    :param llm: Language model instance
    :param prompt: PromptTemplate instance
    :param verbose: Boolean to enable verbose output (default: True)
    :return: LLMChain instance
    """
    return LLMChain(llm=llm, prompt=prompt, verbose=verbose)


def retrieve(query, faiss_index, k=7):
    """
    Retrieve relevant context from the FAISS index based on the user's query.

    Parameters:
        query (str): The user's query string.
        faiss_index (FAISS): The FAISS index containing the embedded documents.
        k (int, optional): The number of most relevant documents to retrieve (default is 3).

    Returns:
        list: A list of the k most relevant documents (or document chunks).
    """
    relevant_context = faiss_index.similarity_search(query, k=k)
    return relevant_context


def create_qa_prompt_template():
    """
    Create a PromptTemplate for question answering based on video content.
    Returns:
        PromptTemplate: A PromptTemplate object configured for Q&A tasks.
    """
    
    qa_template = """
    You are an expert assistant providing detailed answers based on the following video content.
    Relevant Video Context: {context}
    Based on the above context, please answer the following question:
    Question: {question}
    """

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template=qa_template
    )
    return prompt_template


def create_qa_chain(llm, prompt_template, verbose=True):
    """
    Create an LLMChain for question answering.

    Args:
        llm: Language model instance
            The language model to use in the chain (e.g., WatsonxGranite).
        prompt_template: PromptTemplate
            The prompt template to use for structuring inputs to the language model.
        verbose: bool, optional (default=True)
            Whether to enable verbose output for the chain.

    Returns:
        LLMChain: An instantiated LLMChain ready for question answering.
    """
    
    return LLMChain(llm=llm, prompt=prompt_template, verbose=verbose)


def generate_answer(question, faiss_index, qa_chain, k=7):
    """
    Retrieve relevant context and generate an answer based on user input.

    Args:
        question: str
            The user's question.
        faiss_index: FAISS
            The FAISS index containing the embedded documents.
        qa_chain: LLMChain
            The question-answering chain (LLMChain) to use for generating answers.
        k: int, optional (default=3)
            The number of relevant documents to retrieve.

    Returns:
        str: The generated answer to the user's question.
    """

    relevant_context = retrieve(question, faiss_index, k=k)

    answer = qa_chain.predict(context=relevant_context, question=question)

    return answer


# Initialize an empty string to store the processed transcript after fetching and preprocessing
processed_transcript = ""

def summarize_video(video_url):
    """
    Title: Summarize Video

    Description:
    This function generates a summary of the video using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.

    Returns:
        str: The generated summary of the video or a message indicating that no transcript is available.
    """
    global fetched_transcript, processed_transcript
    
    
    if video_url:
        # Fetch and preprocess transcript
        fetched_transcript = get_transcript(video_url)
        processed_transcript = process(fetched_transcript)
    else:
        return "Please provide a valid YouTube URL."

    if processed_transcript:

        llm = OllamaLLM(
            model="llama3.2:3b",
            temperature=0.2,
        )

        summary_prompt = create_summary_prompt()
        summary_chain = create_summary_chain(llm, summary_prompt)

        summary = summary_chain.run({"transcript": processed_transcript})
        return summary
    else:
        return "No transcript available. Please fetch the transcript first."
    


def answer_question(video_url, user_question):
    """
    Title: Answer User's Question

    Description:
    This function retrieves relevant context from the FAISS index based on the user’s query 
    and generates an answer using the preprocessed transcript.
    If the transcript hasn't been fetched yet, it fetches it first.

    Args:
        video_url (str): The URL of the YouTube video from which the transcript is to be fetched.
        user_question (str): The question posed by the user regarding the video.

    Returns:
        str: The answer to the user's question or a message indicating that the transcript 
             has not been fetched.
    """
    global fetched_transcript, processed_transcript

    if not processed_transcript:
        if video_url:
            fetched_transcript = get_transcript(video_url)
            processed_transcript = process(fetched_transcript)
        else:
            return "Please provide a valid YouTube URL."

    if processed_transcript and user_question:
        chunks = chunk_transcript(processed_transcript)

        llm = OllamaLLM(
            model="llama3.2:3b",
            temperature=0.2,
        )

        embedding_model = embeddings
        faiss_index = create_faiss_index(chunks, embedding_model)

        qa_prompt = create_qa_prompt_template()
        qa_chain = create_qa_chain(llm, qa_prompt)

        answer = generate_answer(user_question, faiss_index, qa_chain)
        return answer
    else:
        return "Please provide a valid question and ensure the transcript has been fetched."
    
