from google.cloud import bigquery
from openai import AzureOpenAI
import os
from bertopic import BERTopic
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from dotenv import load_dotenv
from langdetect import detect
import json
import time

start_time = time.time()

os.environ["TOKENIZERS_PARALLELISM"] = "false"

client = bigquery.Client()

load_dotenv()

api_key = os.getenv("API_KEY")

llm_client = AzureOpenAI(
    api_key=api_key,
    api_version="2023-07-01-preview",
    azure_endpoint="https://derai-vision.openai.azure.com/",
)



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
        json_structure[main_topic[1]] = {
            "keyword": topic_keyword,
            "subtopics": {}
        }

        print(f"Processing main topic: {main_topic[0]} - {topic_name}")

        for subtopic in subtopics:
            subtopic_id = subtopic[0]
            subtopic_name = subtopic[1]
            subtopic_keyword = get_subtopic_keyword(topic_keyword, subtopic_name)
            reviews = subtopic_reviews.get(subtopic_id, [])

            json_structure[main_topic[1]]["subtopics"][subtopic_name] = {
                "subtopic_keyword": subtopic_keyword,
                "reviews": reviews
            }

            print(f"  Subtopic ID: {subtopic_id} - {subtopic_name}")
            print(f"    Keyword: {subtopic_keyword}")
            print(f"    Number of reviews: {len(reviews)}")

    with open(output_file, 'w') as f:
        json.dump(json_structure, f, indent=4)

    print(f"JSON structure saved to {output_file}")



if __name__ == "__main__":

    project_id = 'ingka-online-analytics-prod'
    dataset_id = 'app_data_v2'
    table_id = 'app_surveys'

    table_ref = f'{project_id}.{dataset_id}.{table_id}'

    ## Query to test with a fixed number of reviews per day

    num_reviews = 10000
    num_reviews_per_day = 300

    query_test = f"""
        WITH ranked_reviews AS (
            SELECT 
                date, 
                answer_translated,
                ROW_NUMBER() OVER (PARTITION BY date ORDER BY date DESC) as row_num
            FROM {table_ref}
            WHERE answer_translated IS NOT NULL AND rating != 0
        )
        SELECT *
        FROM ranked_reviews
        WHERE row_num <= {num_reviews_per_day}
        ORDER BY date DESC
        LIMIT {num_reviews}
    """

    ## With 6 months of data, the number of reviews will be between 2M and 3M
    ### Of this, only around 200k have a non-null answer_translated

    query_1_month = f"""
        SELECT
            date, 
            answer_translated
        FROM {table_ref}
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 1 MONTH) AND current_date()
            AND answer_translated IS NOT NULL AND rating != 0
        ORDER BY date DESC
    """

    query_3_months = f"""
        SELECT
            date, 
            answer_translated
        FROM {table_ref}
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 3 MONTH) AND current_date()
            AND answer_translated IS NOT NULL AND rating != 0
        ORDER BY date DESC
    """

    query_6_months = f"""
        SELECT
            date, 
            answer_translated
        FROM {table_ref}
        WHERE date BETWEEN DATE_SUB(CURRENT_DATE(), INTERVAL 6 MONTH) AND current_date()
            AND answer_translated IS NOT NULL AND rating != 0
        ORDER BY date DESC
    """

    query_job = client.query(query_test)

    reviews = [row['answer_translated'] for row in query_job]
    timestamps = [row['date'] for row in query_job]

    ## Identify and remove non-english reviews
    ### For 6 months of data, this takes around 10 minutes 


    print("Reviews before processing: ", len(reviews))
    print("Processing reviews...")
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
    timestamps = filtered_timestamps

    print("Reviews after processing: ", len(reviews))


    formatted_timestamps = [ts.strftime("%Y-%m-%d") for ts in timestamps]

    stop_words = set(stopwords.words('english')).union(set(ENGLISH_STOP_WORDS))

    processed_reviews = [' '.join([word for word in word_tokenize(review.lower()) if word.isalnum() and word not in stop_words]) for review in reviews]

    ## Limiting the number of topics with nr_topics does not work
    # nr_topics_before = 'Auto'
    print("Running BERTopic...")
    topic_model = BERTopic()

    # Fit the model on the reviews
    topics, probabilities = topic_model.fit_transform(reviews)

    nr_topics_after = 'auto'

    # Further reduce topics if needed
    # topic_model.reduce_topics(reviews, nr_topics=nr_topics_after)

    topics_over_time = topic_model.topics_over_time(reviews, formatted_timestamps, datetime_format="%Y-%m-%d", nr_bins=10)
    topics = topic_model.get_topics()

    topic_info = topic_model.get_topic_info()
    all_topic_names = '; '.join(topic_info['Name'])

    number_of_topics = len(topics)
    print(f"Generated {number_of_topics} topics")
    hierarchical_topics, Z = topic_model.hierarchical_topics(processed_reviews)



    print("Getting main topics...")
    topics_at_depth = get_topics_at_depth(hierarchical_topics, 3)
    for topic in topics_at_depth:
        print(f"ID: {topic[0]}, Name: {topic[1]}, Distance: {topic[2]}")

    print("Adjusting main topics...")



    topics = adjust_topics(hierarchical_topics, topics_at_depth, 1)

    print("Generating subtopics...")


    subtopics = get_subtopics_for_topics(hierarchical_topics, topics, 1)



    subtopic_topics = get_leaves(subtopics, hierarchical_topics)



    subtopic_reviews = get_reviews_by_subtopic(subtopic_topics, topic_model, reviews)


    model = "gpt-4o" 

    print("Naming topics and creating json file...")

    output_file = 'topics.json'
    create_json_structure(subtopics, subtopic_reviews, output_file)

    end_time = time.time()

    print(f"Execution time: {end_time - start_time} seconds ({(end_time - start_time) / 60} minutes)")