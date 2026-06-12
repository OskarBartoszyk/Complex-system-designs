import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── dane ──────────────────────────────────────────────────────────
edge_index = torch.tensor([[0, 1, 1, 2],
                            [1, 0, 2, 1]], dtype=torch.long)
x = torch.tensor([[-1], [0], [1]], dtype=torch.float)
data = Data(x=x, edge_index=edge_index)

# ── konwersja PyG → NetworkX ───────────────────────────────────────
G = to_networkx(data, to_undirected=True)

# ── layout węzłów ─────────────────────────────────────────────────
pos = nx.spring_layout(G, seed=42)

# ── kolory węzłów według cechy x ──────────────────────────────────
feature_values = [x[i].item() for i in range(data.num_nodes)]
colors = []
for v in feature_values:
    if v < 0:
        colors.append('#FF6B6B')   # czerwony  → x = -1
    elif v == 0:
        colors.append('#FFD93D')   # żółty     → x =  0
    else:
        colors.append('#6BCB77')   # zielony   → x =  1

# ── etykiety: numer węzła + wartość cechy ─────────────────────────
labels = {i: f"węzeł {i}\nx = {x[i].item():.0f}" for i in range(data.num_nodes)}

# ── figura: 3 panele ──────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Wizualizacja grafu PyTorch Geometric", fontsize=15, fontweight='bold', y=1.02)

# ═══════════════════════════════════════════════════════
# Panel 1 – sam graf z etykietami i cechami
# ═══════════════════════════════════════════════════════
ax1 = axes[0]
ax1.set_title("Graf + cechy węzłów", fontsize=12, fontweight='bold', pad=10)

nx.draw_networkx_edges(G, pos, ax=ax1, width=2.5, edge_color='#555555', alpha=0.7)
nx.draw_networkx_nodes(G, pos, ax=ax1, node_color=colors, node_size=1800, alpha=0.95)
nx.draw_networkx_labels(G, pos, labels=labels, ax=ax1, font_size=10, font_weight='bold')

# legenda kolorów
patches = [
    mpatches.Patch(color='#FF6B6B', label='x = -1'),
    mpatches.Patch(color='#FFD93D', label='x =  0'),
    mpatches.Patch(color='#6BCB77', label='x =  1'),
]
ax1.legend(handles=patches, loc='upper left', fontsize=9, framealpha=0.9)
ax1.axis('off')

# ═══════════════════════════════════════════════════════
# Panel 2 – edge_index jako tabela
# ═══════════════════════════════════════════════════════
ax2 = axes[1]
ax2.set_title("edge_index  [2 × 4]", fontsize=12, fontweight='bold', pad=10)
ax2.axis('off')

col_labels = ['kol 0', 'kol 1', 'kol 2', 'kol 3']
row_labels  = ['wiersz 0\n(źródła)', 'wiersz 1\n(cele)']
table_data  = [
    ['0', '1', '1', '2'],
    ['1', '0', '2', '1'],
]
edge_meanings = ['0 → 1', '1 → 0', '1 → 2', '2 → 1']

table = ax2.table(
    cellText=table_data,
    rowLabels=row_labels,
    colLabels=col_labels,
    loc='center',
    cellLoc='center',
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.4, 2.2)

# kolorowanie komórek
header_color  = '#4A4A8A'
source_color  = '#FFD6D6'
target_color  = '#D6F0FF'

for (row, col), cell in table.get_celld().items():
    cell.set_edgecolor('#AAAAAA')
    if row == 0:                          # nagłówek kolumn
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')
    elif col == -1:                       # nagłówek wierszy
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')
    elif row == 1:                        # wiersz źródeł
        cell.set_facecolor(source_color)
    elif row == 2:                        # wiersz celów
        cell.set_facecolor(target_color)

# znaczenia krawędzi pod tabelą
ax2.text(0.5, 0.12,
         "Krawędzie: " + "  |  ".join(edge_meanings),
         ha='center', va='center', transform=ax2.transAxes,
         fontsize=10, color='#333333',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', edgecolor='#CCCCCC'))

# ═══════════════════════════════════════════════════════
# Panel 3 – macierz cech x jako tabela
# ═══════════════════════════════════════════════════════
ax3 = axes[2]
ax3.set_title("Macierz cech  x  [3 × 1]", fontsize=12, fontweight='bold', pad=10)
ax3.axis('off')

feat_row_labels = ['węzeł 0', 'węzeł 1', 'węzeł 2']
feat_col_labels = ['cecha 0']
feat_data = [['-1'], ['0'], ['1']]
feat_colors_cells = [
    ['#FF6B6B'],
    ['#FFD93D'],
    ['#6BCB77'],
]

feat_table = ax3.table(
    cellText=feat_data,
    rowLabels=feat_row_labels,
    colLabels=feat_col_labels,
    cellColours=feat_colors_cells,
    loc='center',
    cellLoc='center',
)
feat_table.auto_set_font_size(False)
feat_table.set_fontsize(12)
feat_table.scale(1.4, 2.5)

for (row, col), cell in feat_table.get_celld().items():
    cell.set_edgecolor('#AAAAAA')
    if row == 0 or col == -1:
        cell.set_facecolor(header_color)
        cell.set_text_props(color='white', fontweight='bold')

# info pod tabelą
ax3.text(0.5, 0.12,
         f"num_nodes = {data.num_nodes}   |   num_features = {data.num_features}",
         ha='center', va='center', transform=ax3.transAxes,
         fontsize=10, color='#333333',
         bbox=dict(boxstyle='round,pad=0.4', facecolor='#F0F0F0', edgecolor='#CCCCCC'))

plt.tight_layout()
plt.savefig('graf_wizualizacja.png', dpi=150, bbox_inches='tight')
plt.show()