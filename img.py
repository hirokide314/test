import streamlit as st
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

st.title("IMAGE")
G = nx.DiGraph()
G.add_node("あ")
G.add_node("い")
G.add_node("う")
G.add_node("え")
G.add_node("お")
G.add_node("か")
G.add_node("き")
G.add_edge("あ","い")
G.add_edge("え","い")
G.add_edge("あ","う")
G.add_edge("う","お")
G.add_edge("う","え")
G.add_edge("う","か")
G.add_edge("か","き")

fm.fontManager.addfont('ipaexg.ttf')

pr = nx.pagerank(G)
plt.rcParams["font.family"] = "IPAGothic"
# plt.figure(figsize=(10,10))
pos = nx.spring_layout(G,k=0.3)
nx.draw_networkx_edges(G, pos=pos)
nx.draw_networkx_nodes(G, pos=pos, node_color=list(pr.values()), cmap=plt.cm.Reds, node_size=[5000*v for v in pr.values()], label=list(G.nodes))
nx.draw_networkx_labels(G, pos=pos, font_family="IPAGothic")
plt.savefig("./graph.png")

st.image("./graph.png")
