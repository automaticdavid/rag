import os
import sys
import logging
import google.cloud.logging
import asyncio
import asyncpg
from google.cloud.sql.connector import Connector, IPTypes
from langchain_community import VertexAIEmbeddings
from google.cloud import aiplatform
from pgvector.asyncpg import register_vector
from langchain.chains.summarize import load_summarize_chain
from langchain_google_vertexai import VertexAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate 
from langchain_community.document_loaders import Document
from google.cloud import texttospeech
from google.cloud import storage
from flask import Flask, request

project_id = "dcl-dev-ai"  
database_password = os.environ['pg_cx']
region = "europe-west4" 
instance_name = "dcl-dev-ai-pgvec"  
database_name = "toys"  
database_user = "postgres"  
min_price = 25  
max_price = 100 
blob_mp3 = "dcl-mp3"

aiplatform.init(project=f"{project_id}", location=f"{region}")
logclient = google.cloud.logging.Client()
logclient.setup_logging()


app = Flask(__name__)

@app.route("/", methods=['POST'])
def toy():
    request_json = request.get_json(silent=True)
    if request_json and 'toy' in request_json:
        toy = request_json['toy']
    else:
        sys.exit(1)
    matches = asyncio.run(run_query(toy))
    answer = ask_gemini(toy, matches)
    # logging.warning(answer)
    #speech(answer, blob_mp3)
    answer = matches
    logging.warning(answer)
    return("\n" + "OK" + "\n")
    
async def run_query(toy):
    matches = await vector_query(toy)
    return(matches)

async def vector_query(toy):
    embeddings_service = VertexAIEmbeddings(model_name="textembedding-gecko@001")
    qe = embeddings_service.embed_query(toy)
    matches = []
    loop = asyncio.get_running_loop()
    async with Connector(loop=loop) as connector:
        # Create connection to Cloud SQL database.
        conn: asyncpg.Connection = await connector.connect_async(
            f"{project_id}:{region}:{instance_name}",  # Cloud SQL instance connection name
            "asyncpg",
            ip_type = IPTypes.PSC,
            user=f"{database_user}",
            password=f"{database_password}",
            db=f"{database_name}",
        )
        await register_vector(conn)
        similarity_threshold = 0.1
        num_matches = 50
        # Find similar products to the query using cosine similarity search
        # over all vector embeddings. This new feature is provided by `pgvector`.
        results = await conn.fetch(
            """
                            WITH vector_matches AS (
                              SELECT product_id, 1 - (embedding <=> $1) AS similarity
                              FROM product_embeddings
                              WHERE 1 - (embedding <=> $1) > $2
                              ORDER BY similarity DESC
                              LIMIT $3
                            )
                            SELECT product_name, list_price, description FROM products
                            WHERE product_id IN (SELECT product_id FROM vector_matches)
                            AND list_price >= $4 AND list_price <= $5
                            """,
            qe,
            similarity_threshold,
            num_matches,
            min_price,
            max_price,
        )
        if len(results) == 0:
            raise Exception("Did not find any results. Adjust the query parameters.")
        for r in results:
            # Collect the description for all the matched similar toy products.
            matches.append(
                f"""The name of the toy is {r["product_name"]}.
                          The price of the toy is ${round(r["list_price"], 2)}.
                          Its description is below:
                          {r["description"]}."""
            )
        await conn.close()
    logging.warning(matches)
    return(matches)

def ask_gemini(user_query, matches):
    llm = VertexAI(model_name="gemini-pro")
    map_prompt_template = """
              You will be given a detailed description of a toy product.
              This description is enclosed in triple backticks (```).
              Using this description only, extract the name of the toy,
              the price of the toy and its features.

              ```{text}```
              SUMMARY:
              """
    map_prompt = PromptTemplate(template=map_prompt_template, input_variables=["text"])
    combine_prompt_template = """
                You will be given a detailed description different toy products
                enclosed in triple backticks (```) and a question enclosed in
                double backticks(``).
                Select one toy that is most relevant to answer the question.
                Using that selected toy description, answer the following
                question.
                You should only use the information in the description.
                Your answer should include the name of the toy and a summary of its feature.
                Your answer should be less than 75 words.
                Your answer should not contain markup and should be just plain text that is easy to read out loud. 
                Your answer must end with the price. 

                
                Description:
                ```{text}```

                
                Question:
                ``{user_query}``

                
                Answer:
                """
    combine_prompt = PromptTemplate(
        template=combine_prompt_template, input_variables=["text", "user_query"]
    )
    docs = [Document(page_content=t) for t in matches]
    chain = load_summarize_chain(
        llm, chain_type="map_reduce", map_prompt=map_prompt, combine_prompt=combine_prompt
    )
    logging.warning(chain)
    answer = chain.run(
        { "input_documents": docs,
        "user_query": user_query }
    )
    logging.warning(answer)
    return(answer)

def speech(text, bucket_name):
    client = texttospeech.TextToSpeechClient()
    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Studio-O",
        ssml_gender=texttospeech.SsmlVoiceGender.FEMALE,
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob("output.mp3")
    with blob.open("wb") as f:
        f.write(response.audio_content)
    logging.warning('Audio content written to file')
