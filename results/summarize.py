#!/usr/bin/env python3
"""Print the tables in RESULTS.md from the curated files in this directory.

    python results/summarize.py            # markdown tables to stdout
    python results/summarize.py --chart    # also write five_mode_run/charts/ragas_by_mode.png

Every number is an aggregate of values stored in results/ (RAGAS scores, timestamps,
retrieval counts, API-reported latencies). Nothing is re-scored.
"""
import argparse
import itertools
import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RUN = HERE / 'five_mode_run'
MODES = ['local', 'global', 'naive', 'hybrid', 'mix']
METRICS = ['faithfulness', 'answer_relevancy', 'context_recall', 'context_precision']


def md(df, floatfmt='{:.3f}'):
    """Render a DataFrame as a GitHub markdown table."""
    cols = [str(c) for c in df.columns]
    lines = ['| ' + ' | '.join([df.index.name or ''] + cols) + ' |',
             '|' + '---|' * (len(cols) + 1)]
    for idx, row in df.iterrows():
        cells = []
        for v in row:
            if isinstance(v, float):
                cells.append('' if pd.isna(v) else floatfmt.format(v))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join([str(idx)] + cells) + ' |')
    return '\n'.join(lines)


def mean_n(s):
    return f'{s.mean():.3f}' if s.count() == len(s) else f'{s.mean():.3f} (N={s.count()})'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--chart', action='store_true')
    args = ap.parse_args()

    df = pd.read_csv(RUN / 'per_mode.csv')
    by = df.groupby('mode')

    print('## RAGAS by mode (mean over questions; N=63 unless shown)\n')
    t = pd.DataFrame({m: {k: mean_n(by.get_group(m)[k]) for k in METRICS + ['ragas_score']} for m in MODES}).T
    t.index.name = 'mode'
    print(md(t))

    print('\n## Median RAGAS by mode\n')
    t = by[METRICS + ['ragas_score']].median().reindex(MODES)
    t.index.name = 'mode'
    print(md(t, '{:.2f}'))

    print('\n## Retrieval and latency by mode\n')
    t = pd.DataFrame({
        'duration_s median': by['duration_ms'].median() / 1000,
        'duration_s mean': by['duration_ms'].mean() / 1000,
        'N timed': by['duration_ms'].count().astype(str),
        'context chars (mean)': by['context_length'].mean().round(0).astype(int).astype(str),
        'entities (mean)': by['entities'].mean(),
        'relations (mean)': by['relations'].mean(),
        'chunks kept (mean)': by['chunks'].mean(),
        'questions with 0 chunks': by['chunks'].apply(lambda s: int((s == 0).sum())).astype(str),
        'chunks from the asked company': (by['own_ticker_chunks'].sum() / by['chunks'].sum()).map('{:.0%}'.format),
        'answer chars (mean)': by['answer_length'].mean().round(0).astype(int).astype(str),
    }).reindex(MODES)
    t.index.name = 'mode'
    print(md(t, '{:.1f}'))

    p = df.pivot(index='question_id', columns='mode', values='ragas_score')[MODES]
    print('\n## Head-to-head on ragas_score (row mode vs column mode: wins-ties-losses, mean difference)\n')
    rows = {}
    for a, b in itertools.permutations(MODES, 2):
        d = p[a] - p[b]
        rows.setdefault(a, {})[b] = f'{(d > 0).sum()}-{(d == 0).sum()}-{(d < 0).sum()} ({d.mean():+.3f})'
    t = pd.DataFrame(rows).T.reindex(index=MODES, columns=MODES).fillna('—')
    t.index.name = 'mode'
    print(md(t))

    best = p.eq(p.max(axis=1), axis=0)
    print('\n## Best mode per question (ties credit every tied mode)\n')
    t = pd.DataFrame({'questions where best': best.sum().reindex(MODES).astype(str)})
    t.index.name = 'mode'
    print(md(t))
    print(f'\nQuestions with a tie for best: {int((best.sum(axis=1) > 1).sum())}. '
          f'Spread (best minus worst mode) per question: median {(p.max(axis=1) - p.min(axis=1)).median():.3f}, '
          f'max {(p.max(axis=1) - p.min(axis=1)).max():.3f}.')

    meta = df.drop_duplicates('question_id').set_index('question_id')[['ticker', 'category', 'reasoning']]
    pm = p.join(meta)
    for col, title in [('ticker', 'ticker'), ('category', 'FinDER category'), ('reasoning', 'FinDER reasoning flag')]:
        print(f'\n## Mean ragas_score by {title}\n')
        t = pm.groupby(col)[MODES].mean()
        t.insert(0, 'N', pm.groupby(col).size().astype(str))
        t.index.name = col
        print(md(t, '{:.2f}'))

    print('\n## Mean ragas_score split by whether the mode kept any document chunk\n')
    t = df.assign(kept=df.chunks.gt(0).map({True: 'chunks > 0', False: 'no chunks'}))
    t = t.groupby(['mode', 'kept'])['ragas_score'].agg(['mean', 'count']).unstack('kept')
    out = pd.DataFrame({k: [f"{t.loc[m, ('mean', k)]:.3f} (N={int(t.loc[m, ('count', k)])})"
                            if not pd.isna(t.loc[m, ('mean', k)]) else '' for m in MODES]
                        for k in ['chunks > 0', 'no chunks']}, index=MODES)
    out.index.name = 'mode'
    print(md(out))

    print('\n## Mean ragas_score of every mode, grouped by how many chunks naive kept for the question\n')
    naive_chunks = df[df['mode'] == 'naive'].set_index('question_id')['chunks']
    bucket = pd.cut(naive_chunks, [-1, 0, 4, 10], labels=['0', '1-4', '5-10'])
    t = p.groupby(bucket, observed=True).mean()
    t.insert(0, 'N', p.groupby(bucket, observed=True).size().astype(str))
    t.index.name = 'naive chunks'
    print(md(t, '{:.2f}'))

    rep = pd.read_csv(RUN / 'ragas_repeat_scoring.csv')
    rep = rep[rep.status == 'success']
    print(f'\n## Same answer and context scored twice ({len(rep)} question-mode pairs)\n')
    rows = {}
    for k in METRICS + ['ragas_score']:
        d = (rep[f'{k}_pilot'] - rep[f'{k}_final']).abs().dropna()
        rows[k] = {'pairs': str(len(d)), 'mean abs diff': d.mean(), 'max abs diff': d.max(),
                   'pairs with abs diff >= 0.5': str(int((d >= 0.5).sum()))}
    t = pd.DataFrame(rows).T
    t.index.name = 'metric'
    print(md(t))

    g = pd.read_csv(HERE / 'gemini_latency_benchmark' / 'calls.csv')
    gb = g.groupby('mode')
    print('\n## Gemini-only latency benchmark (10 questions x 5 modes, one call each)\n')
    t = pd.DataFrame({
        'calls': gb.size().astype(str),
        'succeeded': gb['success'].sum().astype(int).astype(str),
        'response_time median (s)': gb['response_time'].median(),
        'input tokens median': gb['input_tokens'].median().round(0).astype(int).astype(str),
        'output tokens median': gb['output_tokens'].median().round(0).astype(int).astype(str),
    }).reindex(MODES)
    t.index.name = 'mode'
    print(md(t, '{:.1f}'))
    for _, r in g[g['note'].fillna('') != ''].iterrows():
        print(f"\n- {r['question_id']} / {r['mode']}: {r['note']}.")

    c = pd.read_csv(HERE / 'mix_only_custom_judge' / 'evaluation_scores.csv')
    print(f'\n## Mix-only run, custom judge (N={len(c)})\n')
    t = pd.DataFrame({k: {'mean': c[k].mean(), 'judge prompt 1': c[f'{k}_eval1'].mean(),
                          'judge prompt 2': c[f'{k}_eval2'].mean()}
                      for k in ['answer_accuracy', 'context_relevance', 'groundedness']}).T
    t.index.name = 'score'
    print(md(t))

    loc = pd.read_csv(HERE / 'local_llm_run' / 'finder_lightrag_results.csv')
    errs = loc['lightrag_generated_answer'].str.startswith('Error').sum()
    print(f'\n## Local llama.cpp run\n\n{len(loc)} queries, {errs} answers are error strings.')

    if args.chart:
        chart(df)


def chart(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    surface, ink, ink2, muted, grid, bar = '#fcfcfb', '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#2a78d6'
    cols = METRICS + ['ragas_score']
    titles = ['Faithfulness', 'Answer relevancy', 'Context recall', 'Context precision', 'Mean of the four']
    means = df.groupby('mode')[cols].mean().reindex(MODES)
    counts = df.groupby('mode')[cols].count().reindex(MODES)

    fig, axes = plt.subplots(1, 5, figsize=(13, 3.2), sharey=True, facecolor=surface)
    ypos = range(len(MODES))[::-1]
    for ax, col, title in zip(axes, cols, titles):
        ax.set_facecolor(surface)
        vals = means[col].values
        ax.barh(list(ypos), vals, height=0.55, color=bar, zorder=2)
        for y, v, m in zip(ypos, vals, MODES):
            n = counts.loc[m, col]
            label = f'{v:.2f}' + ('' if n == len(df[df['mode'] == m]) else f' (N={n})')
            ax.text(v + 0.02, y, label, va='center', ha='left', fontsize=9, color=ink2)
        ax.set_xlim(0, 1.25)
        ax.set_xticks([0, 0.5, 1])
        ax.set_title(title, fontsize=10, color=ink, loc='left')
        ax.grid(axis='x', color=grid, linewidth=0.8, zorder=0)
        ax.grid(axis='y', visible=False)
        for s in ['top', 'right', 'left']:
            ax.spines[s].set_visible(False)
        ax.spines['bottom'].set_color('#c3c2b7')
        ax.tick_params(colors=muted, labelsize=9, length=0)
    axes[0].set_yticks(list(ypos))
    axes[0].set_yticklabels(MODES, color=ink2, fontsize=10)
    fig.suptitle('RAGAS scores by retrieval mode - 63 FinDER questions, 10 companies '
                 '(judge ministral-14b-2512; means, N=63 unless shown)',
                 x=0.01, ha='left', fontsize=10.5, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    out = RUN / 'charts' / 'ragas_by_mode.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=110, facecolor=surface)
    print(f'\nwrote {out.relative_to(HERE.parent)}')


if __name__ == '__main__':
    main()
