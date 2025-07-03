import streamlit as st

# Define the pages
main_page = st.Page("main_page.py", title="Main Page", icon="🎈")
page_2 = st.Page("page_2.py", title="Page 2", icon="❄️")
page_3 = st.Page("page_3.py", title="Page 3", icon="🎉")

# Set up navigation
pg = st.navigation([main_page, page_2, page_3])

# Run the selected page
pg.run()


'''
    if ask_button and query:
        if len(st.session_state.doc_embeddings) != 0:
            question_vec = get_embedding(client,query)
            top_indices, similarities  = get_top_k_similar_docs(question_vec, st.session_state.doc_embeddings)
            top_docs = [st.session_state.doc_chunks[i] for i in top_indices]
            # Construct a prompt
            context = "\n".join(top_docs)

            # prompt before the prompt
            # you need to setup the prompt structure that allow to response on purpose to the user
            prompt = f"""You are a helpful assistant. Use the following context to answer the question:

            Context:
            {context}

            Question: {query}
            Answer:"""

            completion = client.chat.completions.create(
                model="gpt-4.1-nano",
                messages=[
                    {"role": "system", "content": "You are an assistant who is helping answer a questions. Please answer as if you are talking to a 8 years old children"},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
            )

            st.text_area("Answer:", completion.choices[0].message.content, height=400)
        else:
            st.warning("Please embedded a website first.")
'''