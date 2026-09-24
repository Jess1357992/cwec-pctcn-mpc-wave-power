#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Te 估計誤差的兩張圖。

主文圖：雙面板 (a) R² vs Te 偏差、(b) RMSE 增幅 vs Te 偏差。
補充圖：五條 R² 隨前置時間變化的曲線，0% 基準線加粗。

資料來源（不重跑模型、不重新訓練）：
    outputs/te_context_error/te_error_summary.csv
    outputs/te_context_error/te_error_by_horizon.csv

依 Jessica 2026-09-11 的規格：資料點以直線相連，不做平滑擬合，
暫不加誤差棒（目前只有單一模型的彙整結果，尚未計算不確定性區間）。
圖內不放標題，說明一律寫在 caption（Elsevier 慣例）。
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT = HERE / 'outputs' / 'te_context_error'

# ---------------------------------------------------------------- 樣式
plt.rcParams.update({
    'font.size': 9,
    'font.family': 'DejaVu Sans',
    'axes.linewidth': 0.8,
    'axes.edgecolor': '#666666',
    'xtick.color': '#444444',
    'ytick.color': '#444444',
    'savefig.facecolor': 'white',
})
INK = '#222222'        # 文字用墨色，不用序列顏色
MUTED = '#666666'

# 發散配色：冷色 = 低估、中性灰 = 基準、暖色 = 高估；
# 每一臂由淺到深表示偏差量增大（發散配色檢查的是明度單調，不是相鄰 CVD）。
DIVERGING = {
    -20: ('#14507F', 'v'),
    -10: ('#6BA3D0', '^'),
      0: ('#3D3D3D', 'o'),
     10: ('#E8926B', 's'),
     20: ('#A33A16', 'D'),
}
SINGLE = '#1F6FB4'     # 單一序列用主文既有的藍（同 Fig. 7 / Fig. 8）


def panel_label(ax, text):
    ax.set_title(text, loc='left', fontweight='bold', color=INK, fontsize=10,
                 pad=8)


def tidy(ax):
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


# ================================================================ 主文圖
s = pd.read_csv(OUT / 'te_error_summary.csv', encoding='utf-8-sig')
s = s.sort_values('Te_error_pct').reset_index(drop=True)
x = s['Te_error_pct'].to_numpy(float)

fig, ax = plt.subplots(1, 2, figsize=(7.6, 3.3))

# --- (a) R² vs Te 偏差
a = ax[0]
a.axhline(0.0, ls='--', lw=1.0, color=MUTED, zorder=1)
a.annotate('R² = 0', xy=(x.min() - 4, 0.0), xytext=(0, 4),
           textcoords='offset points', ha='left', va='bottom',
           fontsize=8, color=MUTED)
a.plot(x, s['R2_NSE'], '-', lw=1.8, color=SINGLE, zorder=2)
a.plot(x, s['R2_NSE'], 'o', ms=7, color=SINGLE,
       markeredgecolor='white', markeredgewidth=1.2, zorder=3)
panel_label(a, '(a)  Forecast skill')
a.set_xlabel('$T_e$ estimation bias (%)', color=INK)
a.set_ylabel('R² / NSE', color=INK)
a.set_xticks(x)
a.set_xlim(x.min() - 4.5, x.max() + 4.5)
a.set_ylim(-0.055, 0.105)
tidy(a)

# --- (b) RMSE 增幅 vs Te 偏差
b = ax[1]
inc = s['RMSE_increase_pct'].to_numpy(float)
b.plot(x, inc, '-', lw=1.8, color=SINGLE, zorder=2)
b.plot(x, inc, 'o', ms=7, color=SINGLE,
       markeredgecolor='white', markeredgewidth=1.2, zorder=3)
for xi, yi in zip(x, inc):
    below = (yi == 0.0)          # 基準點標在下方，避開折線頂點
    dx = {-10.0: -6, 10.0: 6}.get(float(xi), 0)   # ±10% 往外挪，不壓在折線上
    b.annotate('%.2f%%' % yi, xy=(xi, yi), xytext=(dx, -13 if below else 11),
               textcoords='offset points',
               ha='right' if dx < 0 else ('left' if dx > 0 else 'center'),
               va='top' if below else 'bottom', fontsize=8, color=INK)
panel_label(b, '(b)  Relative RMSE penalty')
b.set_xlabel('$T_e$ estimation bias (%)', color=INK)
b.set_ylabel('RMSE increase vs. 0% bias (%)', color=INK)
b.set_xticks(x)
b.set_xlim(x.min() - 4.5, x.max() + 4.5)
b.set_ylim(-1.3, 6.6)
tidy(b)

fig.tight_layout()
for ext, kw in (('png', dict(dpi=600)), ('pdf', {})):
    fig.savefig(OUT / ('te_error_main.%s' % ext), bbox_inches='tight',
                facecolor='white', **kw)

# ========================================================== 補充資料圖
h = pd.read_csv(OUT / 'te_error_by_horizon.csv', encoding='utf-8-sig')

fig2, ax2 = plt.subplots(figsize=(6.6, 3.8))
ax2.axhline(0.0, ls='--', lw=1.0, color=MUTED, zorder=1)
for bias in [-20, -10, 0, 10, 20]:
    g = h[h['Te_error_pct'] == bias].sort_values('seconds_ahead')
    colour, marker = DIVERGING[bias]
    baseline = (bias == 0)
    ax2.plot(g['seconds_ahead'], g['R2_NSE'], '-', marker=marker,
             ms=6 if not baseline else 7,
             lw=2.4 if baseline else 1.5,
             color=colour, markeredgecolor='white', markeredgewidth=0.9,
             label=('%+d%%' % bias).replace('-', '\u2212') if bias
                   else '0% (baseline)',
             zorder=4 if baseline else 3)
ax2.set_xlabel('forecast lead time (s)', color=INK)
ax2.set_ylabel('R² / NSE', color=INK)
ax2.set_xticks(np.arange(0.2, 1.7, 0.2))
ax2.set_xlim(0.03, 1.67)
leg = ax2.legend(title='$T_e$ estimation bias', fontsize=8, title_fontsize=8,
                 frameon=False, ncol=2)
leg.get_title().set_color(INK)
tidy(ax2)
fig2.tight_layout()
for ext, kw in (('png', dict(dpi=600)), ('pdf', {})):
    fig2.savefig(OUT / ('te_error_by_horizon.%s' % ext), bbox_inches='tight',
                 facecolor='white', **kw)

print('saved to', OUT)
