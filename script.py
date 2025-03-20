from google.cloud import bigquery
from openai import AzureOpenAI
import os
from bertopic import BERTopic
from dotenv import load_dotenv
from langdetect import detect
import json
import time
from flask import Flask, request, jsonify, send_file
import logging

start_time = time.time()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

service_account_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

if not service_account_path:
    raise RuntimeError("Service account credentials not found!")

client = bigquery.Client()

api_key = os.getenv("API_KEY")

model = "gpt-4o" 

llm_client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint="https://derai-vision.openai.azure.com/",
)

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "ok"})


@app.route('/download')
def download():
    file_path = "/tmp/topics.json"
    return send_file(file_path, as_attachment=True)


@app.route('/generate_topics', methods=['POST'])
def generate_topics():
    
    data = request.get_json()
    start_date = data.get('start_date')
    end_date = data.get('end_date')

    project_id = 'ingka-online-analytics-prod'
    dataset_id = 'app_data_v2'
    table_id = 'app_surveys'

    table_ref = f'{project_id}.{dataset_id}.{table_id}'

    query = f"""
        SELECT
            date, 
            answer_translated
        FROM {table_ref}
        WHERE date BETWEEN '{start_date}' AND '{end_date}'
            AND answer_translated IS NOT NULL AND rating != 0
        ORDER BY date DESC
    """

    query_job = client.query(query)

    reviews = [row['answer_translated'] for row in query_job]
    timestamps = [row['date'] for row in query_job]

    ## Identify and remove non-english reviews

    logger.info("Reviews before processing: %d", len(reviews))
    logger.info("Processing reviews...")

    filtered_reviews = []
    filtered_timestamps = []
    removed_reviews = []

    for review, timestamp in zip(reviews, timestamps):
        try:
            if detect(review) == 'en' and len(review.split()) > 1 and len(review) >= 10:
                filtered_reviews.append(review)
                filtered_timestamps.append(timestamp)
            else:
                removed_reviews.append(review)
        except:
            removed_reviews.append(review)

    reviews = filtered_reviews

    logger.info("Reviews after processing: %d", len(reviews))

    logger.info("Running BERTopic...")
    
    topic_model = BERTopic()

    topics, probabilities = topic_model.fit_transform(reviews)

    topics = topic_model.get_topics()

    number_of_topics = len(topics)
    
    logger.info(f"Generated %d topics", number_of_topics)

    hierarchical_topics = topic_model.hierarchical_topics(reviews)

    logger.info("Getting main topics...")
    
    topics_at_depth = get_topics_at_depth(hierarchical_topics, 3)
    
    for topic in topics_at_depth:
        logger.info(f"ID: %s, Name: %s, Distance: %s", topic[0], topic[1], topic[2])

    logger.info("Adjusting main topics...")

    topics = adjust_topics(hierarchical_topics, topics_at_depth, 1)

    logger.info("Generating subtopics...")

    subtopics = get_subtopics_for_topics(hierarchical_topics, topics, 1)

    subtopic_topics = get_leaves(subtopics, hierarchical_topics)

    subtopic_reviews = get_reviews_by_subtopic(subtopic_topics, topic_model, reviews)

    logger.info("Naming topics and creating json file...")

    output_file = "/tmp/topics.json"
    
    create_json_structure(subtopics, subtopic_reviews, output_file)

    end_time = time.time()

    logger.info("Execution time: %s seconds (%.2f minutes)", end_time - start_time, (end_time - start_time) / 60)
    
    return send_file(output_file, as_attachment=True, download_name="topicsAPI.json")



def get_topics_at_depth(df, depth):
    from collections import deque
    
    # Build adjacency list with stored distance (no accumulation)
    adjacency = {}
    for _, row in df.iterrows():
        adjacency[row['Parent_ID']] = [
            (row['Child_Left_ID'], row['Child_Left_Name'], row['Distance']),
            (row['Child_Right_ID'], row['Child_Right_Name'], row['Distance'])
        ]
    
    root_id = df.iloc[0]['Parent_ID']
    root_name = df.iloc[0]['Parent_Name']
    
    # BFS
    queue = deque([(root_id, root_name, 0)])  # (id, name, depth)
    result = []
    
    while queue:
        node_id, node_name, curr_depth = queue.popleft()
        children = adjacency.get(node_id, [])
        
        for child_id, child_name, child_distance in children:
            child_depth = curr_depth + 1
            if child_depth == depth:
                result.append((child_id, child_name, child_distance))
            elif child_depth < depth:
                queue.append((child_id, child_name, child_depth))
    
    return result


def adjust_topics(df, topics, threshold):
    from collections import defaultdict

    # Build child->parent and parent->children maps
    child_to_parent = {}
    parent_to_children = defaultdict(list)
    for _, row in df.iterrows():
        p_id, p_name, p_dist = row['Parent_ID'], row['Parent_Name'], row['Distance']
        cl_id, cl_name = row['Child_Left_ID'], row['Child_Left_Name']
        cr_id, cr_name = row['Child_Right_ID'], row['Child_Right_Name']
        
        child_to_parent[cl_id] = (p_id, p_name, p_dist)
        child_to_parent[cr_id] = (p_id, p_name, p_dist)
        parent_to_children[p_id].append((cl_id, cl_name, p_dist))
        parent_to_children[p_id].append((cr_id, cr_name, p_dist))

    # Start with the current topics in a set
    final_topics = set(topics)
    
    # Below-threshold topics
    below_threshold = [t for t in topics if t[2] < threshold]

    # For each below-threshold topic, pair it with another topic of the same distance,
    # remove both, then add the parent. Then remove the highest-distance topic and add its children.
    for bt_id, bt_name, bt_dist in below_threshold:
        if (bt_id, bt_name, bt_dist) not in final_topics:
            continue

        # Find another topic with the same distance
        same_dist_candidates = [
            t for t in final_topics
            if t[2] == bt_dist and t != (bt_id, bt_name, bt_dist)
        ]
        if not same_dist_candidates:
            continue

        # Remove the below-threshold topic and its same-distance candidate
        same_dist_topic = same_dist_candidates[0]
        final_topics.remove((bt_id, bt_name, bt_dist))
        final_topics.remove(same_dist_topic)

        # Add the parent of the below-threshold topic
        parent = child_to_parent.get(bt_id, (bt_id, bt_name, bt_dist))
        final_topics.add(parent)

        # Find the highest-distance topic, remove it, and add its children
        if final_topics:
            highest_topic = max(final_topics, key=lambda x: x[2])
            final_topics.remove(highest_topic)
            h_id, h_name, h_dist = highest_topic
            for ch_id, ch_name, ch_dist in parent_to_children.get(h_id, []):
                final_topics.add((ch_id, ch_name, ch_dist))

    return list(final_topics)


def get_subtopics_for_topics(df, topics, threshold):
    """
    For each topic in 'topics', find subtopics by going up to 2 levels down a binary tree:
      1) If the topic's direct children (level 1) have distance < threshold, return those 2 children.
      2) Otherwise, go one more level (level 2) and return those 4 descendants

    Returns a dict: { "topic_id:topic_name": [ (child_id, child_name, distance), ... ] }
    """
    from collections import defaultdict, deque

    parent_to_children = defaultdict(list)
    for _, row in df.iterrows():
        p_id = row['Parent_ID']
        parent_to_children[p_id].append((row['Child_Left_ID'], row['Child_Left_Name'], row['Distance']))
        parent_to_children[p_id].append((row['Child_Right_ID'], row['Child_Right_Name'], row['Distance']))

    def collect_descendants(root_id, max_level=2):
        queue = deque([(root_id, 0)])
        levels_nodes = defaultdict(list)
        while queue:
            node_id, lvl = queue.popleft()
            for (cid, cname, cdist) in parent_to_children.get(node_id, []):
                levels_nodes[lvl + 1].append((cid, cname, cdist))
                if lvl + 1 < max_level:
                    queue.append((cid, lvl + 1))

        for level in range(1, max_level + 1):
            nodes = levels_nodes.get(level, [])
            if not nodes:
                return []
            if any(n[2] < threshold for n in nodes) or level == max_level:
                return nodes
        return []

    result = {}
    for (t_id, t_name, t_dist) in topics:
        subtopics = collect_descendants(t_id)
        result[(t_id, t_name)] = subtopics
    return result


def get_leaves(topic_structure, hierarchical_topics):
    """
    For each subtopic, get its ID and retrieve the 'Topics' attribute from the hierarchical_topics dataframe.

    Parameters:
    - topic_structure: Dictionary containing topics and their subtopics.
    - hierarchical_topics: DataFrame containing hierarchical topic information.

    Returns:
    - Dictionary with subtopic IDs as keys and their 'Topics' attributes as values.
    """
    subtopic_topics = {}

    for main_topic, subtopics in topic_structure.items():
        for subtopic in subtopics:
            subtopic_id = subtopic[0]
            # Find the row in the dataframe with the matching subtopic ID
            row = hierarchical_topics[hierarchical_topics['Parent_ID'] == subtopic_id]
            if not row.empty:
                subtopic_topics[subtopic_id] = row.iloc[0]['Topics']
            else:
                subtopic_topics[subtopic_id] = [int(subtopic_id)]

    return subtopic_topics


def get_reviews_by_subtopic(subtopic_topics, topic_model, documents):
    """
    Get reviews associated with each subtopic ID in the subtopic_topics dictionary.

    Parameters:
    - subtopic_topics: Dictionary with subtopic IDs as keys and list of topic IDs as values.
    - topic_model: Trained BERTopic model.
    - documents: List of all input documents to the BERTopic model.

    Returns:
    - Dictionary with subtopic IDs as keys and list of reviews as values.
    """
    subtopic_reviews = {}

    # Get topic assignments for each document
    topic_assignments = topic_model.transform(documents)[0]

    for subtopic_id, topic_ids in subtopic_topics.items():
        # Filter documents based on the topic IDs
        associated_docs = [doc for doc, assigned_topic in zip(documents, topic_assignments) if assigned_topic in topic_ids]
        subtopic_reviews[subtopic_id] = associated_docs

    return subtopic_reviews


def get_topic_keyword(cluster_words):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful expert summarizer that identifies and generates a concise, broad topic word for each cluster of words.\n"
                "The topic word should capture the essence of all the words in the cluster.\n"
                "Merge similar or related words into a single, broader category.\n"
                "Use singular words unless a plural form is necessary.\n"                
                "Use only one word. 2 or 3 words can be used only when they are part of a composite word and are better to represent the idea of the topic (e.g.: ease of use).\n"
                "If you identify a verb as a topic, use the noun version (e.g., use 'order' instead of 'ordering').\n"
                "Generalize the topic word; for example, if you encounter 'saleswoman' or 'salesman', abstract it to 'staff'.\n"
                "Provide the output as a single word."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please read the following cluster of words carefully and generate a single topic word that captures the essence of all the words.\n"
                "The topic word should be broad and general, capturing the essence of the cluster's main points without being overly specific or redundant.\n"
                "The topics could be either nouns that refers to a certain characteristic of the product of spefic features or parts of the product (e.g.: click & collect, email redeem, etc.)\n"
                f"Cluster: {', '.join(cluster_words)}\n"
                "Topic word(s):"
            ),
        },
    ]

    response = ' '
    
    # Generate the topic word using the language model
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=5,
        temperature=0.4,
        n=1,
        stop=None,
    )

    # Extract and return the topic word
    return response.choices[0].message.content.strip()


def get_subtopic_keyword(topic_keyword, cluster_words):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful expert summarizer that identifies and generates a concise, broad subtopic word for each cluster of words.\n"
                "The topic word should capture the essence of all the words in the cluster.\n"
                "The words you choose can be specific, since they are a specialization of a broader topic word.\n" 
                "Use singular words unless a plural form is necessary.\n"                
                "Use only one word unless 2 or 3 words are better to represent the idea of the subtopic.\n"
                "If you identify a verb as a subtopic, use the noun version (e.g., use 'order' instead of 'ordering').\n"
                "Generalize the topic word; for example, if you encounter 'saleswoman' or 'salesman', abstract it to 'staff'.\n"
                f"Provide the output as: '{topic_keyword} - <Subtopic word>'."
            ),
        },
        {
            "role": "user",
            "content": (
                "Please read the following cluster of words carefully and generate a single subtopic word that captures the essence of all the words.\n"
                "The subtopic is a specification of the broader topic, therefore it should be about an aspect that the customers mention and that is related to the broader topic.\n"
                "The topics could be either nouns that refers to a certain characteristic of the product of spefic features or parts of the product (e.g.: click & collect, email redeem, etc.)\n"
                f"The broader topic word is: {topic_keyword}\n"
                f"Cluster: {', '.join(cluster_words)}\n"
                "Topic word(s):"
            ),
        },
    ]

    response = ' '

    # Generate the topic word using the language model
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=10,
        temperature=0.4,
        n=1,
        stop=None,
    )

    # Extract and return the topic word
    return response.choices[0].message.content.strip()


def get_review_summary_short(reviews, llm_client, model, selected_subtopic):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a skilled summarizer specializing in customer feedback analysis.\n"
                "Your role is to identify and concisely summarize the main themes, sentiments, and frequently mentioned points in customer reviews.\n"
                "The reviews provided are related to an IKEA service and may discuss various aspects such as product quality, delivery, customer service, payment, or store experience.\n" 
                "The summary you generate will be used by coworkers to understand in a few words what the reviews are talking about.\n"           
                "Provide the output as a short text summary with no more than 70 words. Do not exceed this limit.\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Please read carefully the following customer reviews and generate a summary of the main aspects that customers are discussing.\n"
                "The summary should be as concise as possible, only reporting the main aspects.\n"
                f"The summary should focus on this particular topic: {selected_subtopic}. Ensure that all aspects of the text directly relate to this topic, without introducing unrelated information.\n"
                "In case an aspect is mentioned in many reviews, the summary should include 'Many customers' to highlight that it is a common positive/negative point.\n"
                "Focus on the most significant details that are repeated or impactful.\n"
                "I will provide you with the reviews and you will generate the summary.\n"
                f"Reviews: {reviews}\n"
                "Summary:\n"
            ),
        },
    ]

    # Generate the topic word using the language model
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=100,
        temperature=0.5,
        n=1,
        stop=None,
    )

    # Extract and return the topic word
    return response.choices[0].message.content.strip()


def get_review_summary_long(reviews, llm_client, model, selected_subtopic):
    messages = [
        {
            "role": "system",
            "content": (
                "You are a skilled summarizer specializing in customer feedback analysis.\n"
                "Your role is to identify and concisely summarize the main themes, sentiments, and frequently mentioned points in customer reviews.\n"
                "The reviews provided are related to an IKEA service and may discuss various aspects such as product quality, delivery, customer service, payment, or store experience.\n" 
                "The summaries you generate will be used by coworkers to understand comprehensively the main positive and negative aspects of the reviews.\n"           
                "If a group of reviews does not contain any positive aspects, you can skip the positive points section.\n"           
                "If a group of reviews does not contain any negative aspects, you can skip the negative points section.\n"           
                "Provide the output in the following format: \n"
                "<b>Positive points:</b>\n • Point 1 \n • Point 2 \n ...\n"
                "<b>Negative points:</b>\n • Point 1 \n • Point 2 \n ...\n"
            ),
        },
        {
            "role": "user",
            "content": (
                "Please read carefully the following customer reviews and generate summaries of the main aspects that customers are discussing.\n"
                "The summary should be comprehensive, touching the main aspects mentioned by customer reviews.\n"
                f"The summary should focus on this particular topic: {selected_subtopic}. Ensure that all aspects of the text directly relate to this topic, without introducing unrelated information.\n"
                "In case an aspect is mentioned in many reviews, the summary should include 'Many customers' to highlight that it is a common positive/negative point.\n"
                f"Reviews: {reviews}\n"
                "Summary:\n"
            ),
        },
    ]

    # Generate the topic word using the language model
    response = llm_client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=250,
        temperature=0.5,
        n=1,
        stop=None,
    )

    # Extract and return the topic word
    return response.choices[0].message.content.strip()


def create_json_structure(subtopics_structure, subtopic_reviews, output_file):
    """
    Create a JSON structure with topics, subtopics, and reviews, and save it to a file.

    Parameters:
    - subtopics_structure: Dictionary containing topics and their subtopics.
    - subtopic_reviews: Dictionary with subtopic IDs as keys and list of reviews as values.
    - output_file: Path to the output JSON file.
    """
    json_structure = {}

    for main_topic, subtopics in subtopics_structure.items():
        topic_name = main_topic[1]
        topic_keyword = get_topic_keyword(topic_name)
        subtopic_ids = [subtopic[0] for subtopic in subtopics]
        merged_reviews = [review for subtopic_id in subtopic_ids for review in subtopic_reviews.get(subtopic_id, [])]
        topic_short_summary = get_review_summary_short(merged_reviews, llm_client, model, topic_keyword)
        topic_long_summary = get_review_summary_long(merged_reviews, llm_client, model, topic_keyword)
        json_structure[main_topic[1]] = {
            "Keyword": topic_keyword,
            "Short summary": topic_short_summary,
            "Long summary": topic_long_summary,
            "Subtopics": {}
        }

        print(f"Processing main topic: {main_topic[0]} - {topic_name}")

        for subtopic in subtopics:
            subtopic_id = subtopic[0]
            subtopic_name = subtopic[1]
            subtopic_keyword = get_subtopic_keyword(topic_keyword, subtopic_name)
            reviews = subtopic_reviews.get(subtopic_id, [])
            subtopic_short_summary = get_review_summary_short(reviews, llm_client, model, subtopic_keyword)
            subtopic_long_summary = get_review_summary_long(reviews, llm_client, model, subtopic_keyword)
            reviews = list(set(reviews))
            json_structure[main_topic[1]]["Subtopics"][subtopic_name] = {
                "Subtopic_keyword": subtopic_keyword,
                "Short summary": subtopic_short_summary,
                "Long summary": subtopic_long_summary,
                "Reviews": reviews
            }

            print(f"  Subtopic ID: {subtopic_id} - {subtopic_name}")
            print(f"    Keyword: {subtopic_keyword}")
            print(f"    Number of reviews: {len(reviews)}")

    with open(output_file, 'w') as f:
        json.dump(json_structure, f, indent=4)

    print(f"JSON structure saved to {output_file}")



if __name__ == '__main__':
    app.run(port=int(os.environ.get("PORT", 8080)),host='0.0.0.0',debug=True)