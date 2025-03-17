import streamlit as st
import openai
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import matplotlib.pyplot as plt
import os
import pickle

openai.api_key = os.getenv('OPENAI_API_KEY')

#テキストをOpenAIでベクトル化
def create_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding


#グラフから類似したノードを検索
def find_most_similar_node(question_embedding, node_embeddings):
    best_node = None
    best_score = -1

    for node, embedding in node_embeddings.items():
        similarity = cosine_similarity(
            [question_embedding], [embedding]
        )[0][0]

        if similarity > best_score:
            best_score = similarity
            best_node = node

    return best_node, best_score

#グラフ検索結果をOpenAIに渡して回答生成
def ask_openai(question, context):
    system = f"""
        あなたは楽天ステイによって生み出された優秀なチャットボットです。
    """
    prompt = f"""
        以下の関連情報を参考にして、ユーザーの質問に答えてください。

        --- 関連情報 ---
        {context}

        --- ユーザーの質問 ---
        {question}
    """
    print("プロンプト：", prompt,"\n\n")
    msg=st.session_state['messages']+[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt}
        ]

    print("メッセージ：", msg) 
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=msg
    )
    return response.choices[0].message.content

def get_related_nodes(graph, node):
    """指定したノードの関連ノードを取得"""
    if node in graph.nodes:
        return list(graph.neighbors(node))
    return []

##################################3

# タイトルの表示
st.title('SRS bot')

#セッション初期化
if 'messages' not in st.session_state:
    st.session_state['messages'] = []

# チャット履歴の表示
for message in st.session_state['messages']:
    with st.chat_message(message['role']):
        st.markdown(message['content'])

# ユーザーからの入力受付
if prompt := st.chat_input('メッセージを入力してください'):
    with st.chat_message('user'):
        st.markdown(prompt)

    #グラフ読み込み
    with open("./graph.pkl", mode="rb") as f:
        G = pickle.load(f)

    with open("./embeddings.pkl", mode="rb") as f:
        node_embeddings = pickle.load(f)

    # AIの応答を生成中であることを示すスピナーの表示
    with st.chat_message('assistant'):
        with st.spinner('考え中...'):
            # 質問をベクトル化
            print("---------------------------------------\n")
            question_embedding = create_embedding(prompt)

            # 最も関連性の高いノードを検索
            best_match, similarity_score = find_most_similar_node(question_embedding, node_embeddings)
            print(f"最も関連するノード: {best_match}, 類似度: {similarity_score}\n")

            # 類似ノードの関連ノードを取得
            related_nodes = get_related_nodes(G, best_match)
            print("類似ノード：",related_nodes,"\n\n")

            #文脈生成
            context=f"最も関連するノード：{best_match}"
            for node in related_nodes:
                context = f"{context}\n{best_match} - {nx.get_edge_attributes(G, 'relationship')[best_match, node]} -> {node}"

            # 回答生成
            answer = ask_openai(prompt, context)
            st.markdown(answer)
            print("回答：", answer,"\n\n")


    # ユーザーのメッセージを履歴に追加
    st.session_state['messages'].append({'role': 'user', 'content': prompt})
    # AIのメッセージを履歴に追加
    st.session_state['messages'].append({'role': 'assistant', 'content': answer})