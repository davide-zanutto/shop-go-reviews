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