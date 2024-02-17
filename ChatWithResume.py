import argparse
import torch
from auto_gptq import AutoGPTQForCausalLM
from langchain import HuggingFacePipeline, PromptTemplate
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from pdf2image import convert_from_path
from transformers import AutoTokenizer, TextStreamer, pipeline

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def load_sample_questions(file_path):
    with open(file_path, "r") as file:
        questions = file.readlines()
    return [question.strip() for question in questions]

def main(data_dir, questions_file):
    # Load resumes from a directory
    data_loader = PyPDFDirectoryLoader(data_dir)
    all_resume = data_loader.load()

    # Split resumes into smaller text chunks
    split_text = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    all_texts = split_text.split_documents(all_resume)

    # Create a Chroma database from the text chunks
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="hkunlp/instructor-large", model_kwargs={"device": DEVICE}
    )
    resume_db = Chroma.from_documents(all_texts, embeddings, persist_directory="resume_db")

    # Initialize GPT-based conversational model
    model_name_or_path = "TheBloke/Llama-2-13B-chat-GPTQ"
    model_basename = "model"
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
    model = AutoGPTQForCausalLM.from_quantized(
        model_name_or_path,
        revision="gptq-4bit-128g-actorder_True",
        model_basename=model_basename,
        use_safetensors=True,
        trust_remote_code=True,
        inject_fused_attention=False,
        device=DEVICE,
        quantize_config=None,
    )

    # Initialize text streamer
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    # Initialize text generation pipeline
    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        temperature=0,
        top_p=0.95,
        repetition_penalty=1.15,
        streamer=streamer,
    )

    # Initialize HuggingFace pipeline
    llm = HuggingFacePipeline(pipeline=text_pipeline, model_kwargs={"temperature": 0})

    # Define prompt template
    system_prompt = "Use the following pieces of context to answer questions about the candidate's Resume., "
    def generate_prompt(prompt, system_prompt):
        return f"""[INST] <<SYS>> 
        {system_prompt} 
        <</SYS>> 
        {prompt} [/INST]
        """.strip()

    template = generate_prompt(
        """
        {context}
        Question: {question}
        """,
        system_prompt=system_prompt,
    )
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])

    # Define QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=resume_db.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )

    # Load sample questions
    if questions_file:
        sample_questions = load_sample_questions(questions_file)
    else:
        sample_questions = [
            "What is the candidate's education?",
            "Give me the top 3 skills from the Resume.",
            "What are the key technologies the candidate has worked on?"
            "Of the lot, pick the best Resume that matches the input job description"
        ]

    # Query the QA chain with sample questions
    for question in sample_questions:
        qa_chain(question)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chat with Resumes Intelligently")
    parser.add_argument("data_dir", type=str, help="Path to the directory containing resumes.")
    parser.add_argument("--questions_file", type=str, default="", help="Path to the file containing sample questions.")
    args = parser.parse_args()
    main(args.data_dir, args.questions_file)
