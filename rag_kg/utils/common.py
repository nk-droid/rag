import networkx as nx
import plotly.graph_objects as go

def visualize_kg(result, output_file="kg_visualization.html"):
    G = nx.DiGraph()

    for idx, record in enumerate(result):
        if idx == 100:
            break
        entity1 = record["Entity1"]
        entity2 = record["Entity2"]
        rel = record["Relationship"]
        
        G.add_node(entity1, type=record["Type1"])
        G.add_node(entity2, type=record["Type2"])
        G.add_edge(entity1, entity2, relationship=rel)

    if len(G.nodes()) == 0:
        print("Warning: Graph has no nodes.")
        return

    pos = nx.spring_layout(G, seed=42)

    edge_x, edge_y, edge_text = [], [], []
    for src, dst, data in G.edges(data=True):
        x0, y0 = pos[src]
        x1, y1 = pos[dst]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        edge_text.append(f"{src} → {dst} ({data['relationship']})")

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color="#888"),
        mode="lines",
        hoverinfo="text",
        text=edge_text * (len(edge_x) // 3)
    )

    node_x, node_y, node_text, node_colors = [], [], [], []
    node_types = set(nx.get_node_attributes(G, 'type').values())
    color_map = {node_type: i for i, node_type in enumerate(node_types)}

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_type = G.nodes[node]["type"]
        node_text.append(f"{node} ({node_type})")
        node_colors.append(color_map[node_type])

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        marker=dict(
            size=15,
            color=node_colors,
            colorscale='Viridis',
            line=dict(width=2, color='DarkSlateGrey')
        ),
        text=node_text,
        textposition="top center",
        hoverinfo="text"
    )

    fig = go.Figure(data=[edge_trace, node_trace])
    fig.update_layout(
        showlegend=False,
        title="Knowledge Graph Visualization",
        title_x=0.5,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(255,255,255,1)',
        height=600,
        width=800,
        margin=dict(l=20, r=20, t=40, b=20)
    )

    fig.write_html(output_file)
    print(f"Visualization saved as {output_file}")