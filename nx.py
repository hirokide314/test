import openai
import json
import networkx as nx
import matplotlib.pyplot as plt
from langchain.text_splitter import CharacterTextSplitter
import os
import pickle

openai.api_key = os.getenv('OPENAI_API_KEY')

#テキストをOpenAIで解析してjson形式で返す
def extract_graph_structure(text):
    prompt = f"""
    以下のHTML全体を詳細に解析して全ての情報をノード（概念）とエッジ（関係）に分解してください。
    出力は'{'ではじまり'}'で終わり、などの余計な文字がないjson形式にしてくだい。
     出力フォーマット:
    {{
      "nodes": [
        {{"id": "デュポン", "label": "Company"}},
        {{"id": "化学産業", "label": "Industry"}}
      ],
      "edges": [
        {{"source": "デュポン", "target": "化学産業", "relationship": "関連"}}
      ]
    }}
        
    ここから下の全てが解析対象のHTMLです:
    {text}
    """
    # print(prompt)

    response = openai.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    print(response.choices[0].message.content)

    return json.loads(response.choices[0].message.content)

#json形式の情報をグラフにする
def create_graph_from_json(G,data):
  for node in data["nodes"]:
      G.add_node(node["id"], label=node["label"])

  # エッジを追加
  for edge in data["edges"]:
      G.add_edge(edge["source"], edge["target"], relationship=edge["relationship"])

  return G

def create_embedding(text):
    response = openai.embeddings.create(
        input=text,
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

####################################################################

text_splitter = CharacterTextSplitter(
    separator="\n",         # チャンクの区切り文字
    chunk_size=1000,          # チャンクの最大文字数
    chunk_overlap=200,        # チャンク間の重複する文字数
    length_function=len,      # 文字数で分割
    is_separator_regex=False, # separatorを正規表現として扱う場合はTrue
)

#file読み込み
f = open('./source.txt', encoding='UTF-8')
text_data=f.read()
f.close()

G = nx.DiGraph()
# Confluence のチャンクをグラフデータに変換
texts = text_splitter.create_documents([text_data])
i=1
for t in texts:
  print(f"{t}\n{i}/{len(texts)}\n")
  i=i+1
  json_data = extract_graph_structure(t)
  G = create_graph_from_json(G, json_data)

# 描画
pr = nx.pagerank(G)
# plt.figure(figsize=(10,10))
pos = nx.spring_layout(G,k=0.3)
# nx.draw_networkx_edges(G, pos=pos)
# nx.draw_networkx_nodes(G, pos=pos, node_color=list(pr.values()), cmap=plt.cm.Reds, node_size=[5000*v for v in pr.values()], label=list(G.nodes))
nx.draw(G, with_labels=True, font_family="MS Gothic") 
# plt.show()
plt.savefig("./graph.png")

# 既存のグラフのノードをベクトル化
node_embeddings = {}
i=1
for node in G.nodes:
    print(f"{i}/{len(G.nodes)} {node}")
    i=i+1
    node_embeddings[node] = create_embedding(node)

# グラフ保存
with open("./graph.pkl", mode="wb") as f:
  pickle.dump(G, f)
  

# ベクトル保存
with open("./embeddings.pkl", mode="wb") as f:
  pickle.dump(node_embeddings, f)
  
