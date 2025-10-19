import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from scipy.stats import pearsonr
from io import StringIO

st.set_page_config(page_title='🎥 Balancing Fame and Talent in Bollywood', layout='wide')

GOLD = '#D4AF37'
RED = '#D72638'
ACCENT = '#912F40'

@st.cache_data
def load_data(path='BollywoodActorRanking (2).csv'):
    # Read CSV and treat common NULL token as NaN. If the file is missing,
    # generate a small synthetic sample so the app can run without the data file.
    try:
        df = pd.read_csv(path, na_values=['NULL', 'null', 'NaN'])
    except FileNotFoundError:
        # Generate deterministic synthetic sample data
        rng = np.random.default_rng(42)
        n = 12
        actors = [f'Actor {i}' for i in range(1, n + 1)]
        movieCount = rng.integers(1, 60, size=n)
        avg_ratings = rng.uniform(4.0, 9.0, size=n)
        ratingSum = avg_ratings * movieCount
        googleHits = rng.integers(1000, 1000000, size=n)
        df = pd.DataFrame({
            'actor': actors,
            'movieCount': movieCount,
            'ratingSum': ratingSum,
            'googleHits': googleHits
        })
        st.warning('Data file not found; running with generated sample data.')
    # If dataset uses different actor column name, normalize to 'actor'
    if 'actorName' in df.columns and 'actor' not in df.columns:
        df = df.rename(columns={'actorName': 'actor'})
    if 'actorId' in df.columns and 'actor' not in df.columns:
        # keep actorId but ensure actor exists
        df['actor'] = df.get('actor', df.get('actorName', None))

    # Basic cleaning: drop exact duplicate rows, then drop rows missing essential fields
    df = df.drop_duplicates()
    # Ensure essential columns exist
    required = ['actor', 'movieCount', 'ratingSum']
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Required column missing: {c}")

    # Coerce numeric columns
    df['movieCount'] = pd.to_numeric(df['movieCount'], errors='coerce')
    df['ratingSum'] = pd.to_numeric(df['ratingSum'], errors='coerce')
    # Remove rows with missing essential values
    df = df.dropna(subset=['actor', 'movieCount', 'ratingSum']).copy()

    # Remove non-positive movie counts to avoid div by zero
    df = df[df['movieCount'] > 0].copy()

    # Compute avgRating
    df['avgRating'] = df['ratingSum'] / df['movieCount']

    # Handle normalized columns: if present but on a 0-10 scale (max>1.1) -> rescale to 0-1
    norm_cols = ['normalizedMovieRank', 'normalizedGoogleRank', 'normalizedRating']
    for col in norm_cols:
        if col in df.columns:
            # ensure numeric
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if df[col].notna().any():
                col_min = df[col].min()
                col_max = df[col].max()
                if col_max > 1.1:
                    # rescale to 0-1
                    df[col] = (df[col] - col_min) / (col_max - col_min + 1e-9)
        else:
            # attempt to create a normalized proxy from raw columns
            base = None
            if col == 'normalizedMovieRank' and 'movieCount' in df.columns:
                base = df['movieCount']
            if col == 'normalizedGoogleRank' and 'googleHits' in df.columns:
                base = pd.to_numeric(df['googleHits'], errors='coerce')
            if col == 'normalizedRating' and 'avgRating' in df.columns:
                base = df['avgRating']
            if base is not None:
                bmin, bmax = base.min(), base.max()
                if bmax > bmin:
                    df[col] = (base - bmin) / (bmax - bmin)
                else:
                    df[col] = 0.5

    # Fill any remaining NaNs in normalized columns with the column mean
    for col in norm_cols:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = 0.0

    # Final KPI columns
    df['fameScore'] = 0.5 * df['normalizedGoogleRank'] + 0.5 * df['normalizedMovieRank']
    df['talentScore'] = df['normalizedRating']
    df['balanceScore'] = 1 - (df['fameScore'] - df['talentScore']).abs()
    return df


df = load_data()

def sidebar_filters(df):
    st.sidebar.header('Filters & Settings')
    min_movies, max_movies = int(df['movieCount'].min()), int(df['movieCount'].max())
    movie_count_range = st.sidebar.slider('Movie count range', min_movies, max_movies, (min_movies, max_movies))
    min_rating, max_rating = float(df['avgRating'].min()), float(df['avgRating'].max())
    rating_range = st.sidebar.slider('Average rating range', min_rating, max_rating, (min_rating, max_rating))
    actor_search = st.sidebar.multiselect('Select actors (optional)', options=df['actor'].tolist())
    st.sidebar.markdown('---')
    st.sidebar.markdown('Dataset: `BollywoodActorRanking (2).csv`')
    st.sidebar.markdown('Developer: Muhamed Ahmad')
    st.sidebar.markdown('---')
    st.sidebar.header('About')
    st.sidebar.markdown('This dashboard analyzes alignment between public attention (fame) and movie performance (talent) for Bollywood actors.\n\nData source: internal compilation / public scraping (CSV provided).')
    return movie_count_range, rating_range, actor_search

movie_count_range, rating_range, actor_search = sidebar_filters(df)

# Apply filters
mask = (
    (df['movieCount'] >= movie_count_range[0]) & (df['movieCount'] <= movie_count_range[1]) &
    (df['avgRating'] >= rating_range[0]) & (df['avgRating'] <= rating_range[1])
)
if actor_search:
    mask = mask | df['actor'].isin(actor_search)

df_filtered = df[mask].copy()

st.markdown(f"<div style='background:linear-gradient(90deg, {GOLD}, {RED}); padding:20px; border-radius:8px'>"
            "<h1 style='color:white;'>🎥 Balancing Fame and Talent in Bollywood</h1>"
            "<h4 style='color:#fff5e6;'>An interactive analysis of actors’ popularity and performance.</h4></div>", unsafe_allow_html=True)

st.markdown('This dashboard helps entertainment journalists and studio executives explore whether actors\' public fame aligns with movie quality. Use the sidebar to filter actors and ranges.')

# KPI cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric('Average Fame Score', f"{df_filtered['fameScore'].mean():.2f}")
with col2:
    st.metric('Average Talent Score', f"{df_filtered['talentScore'].mean():.2f}")
with col3:
    try:
        corr, p = pearsonr(df_filtered['fameScore'], df_filtered['talentScore'])
    except Exception:
        corr, p = np.nan, np.nan
    st.metric('Fame vs Talent (Pearson r)', f"{corr:.2f}")
with col4:
    top_balanced = df_filtered.sort_values('balanceScore', ascending=False).iloc[0]
    st.metric('Most Balanced Actor', top_balanced['actor'])

st.markdown('---')

# Scatter plot
st.subheader('Fame vs Talent — Scatter')
highlight_above = st.checkbox('Highlight actors above-average in both metrics', value=True)
selected_actor = st.selectbox('Zoom to actor (optional)', options=[''] + df_filtered['actor'].tolist())

# create scatter with customdata for controlled hover formatting
fig = px.scatter(df_filtered, x='fameScore', y='talentScore', color='balanceScore', color_continuous_scale='reds',
                 labels={'fameScore':'Fame Score', 'talentScore':'Talent Score'},
                 custom_data=['actor', 'avgRating'])
# Hovertemplate: show actor and rounded numeric values only on hover
fig.update_traces(hovertemplate='<b>%{customdata[0]}</b><br>avgRating=%{customdata[1]:.2f}<br>Fame=%{x:.2f}<br>Talent=%{y:.2f}<extra></extra>')
if highlight_above:
    avg_fame = df_filtered['fameScore'].mean()
    avg_talent = df_filtered['talentScore'].mean()
    above = df_filtered[(df_filtered['fameScore'] > avg_fame) & (df_filtered['talentScore'] > avg_talent)]
    # Add highlighted markers but do not show text labels by default; show details on hover
    if not above.empty:
        fig.add_scatter(x=above['fameScore'], y=above['talentScore'], mode='markers',
                        marker=dict(symbol='star', size=12, color=GOLD), name='Above Avg Both',
                        customdata=above[['actor', 'avgRating']].to_numpy(),
                        hovertemplate='<b>%{customdata[0]}</b><br>avgRating=%{customdata[1]:.2f}<br>Fame=%{x:.2f}<br>Talent=%{y:.2f}<extra></extra>')

if selected_actor:
    actor_row = df_filtered[df_filtered['actor'] == selected_actor]
    if not actor_row.empty:
        fig.add_scatter(x=actor_row['fameScore'], y=actor_row['talentScore'], mode='markers',
                        marker=dict(size=18, color=ACCENT), name='Selected Actor',
                        customdata=actor_row[['actor', 'avgRating']].to_numpy(),
                        hovertemplate='<b>%{customdata[0]}</b><br>avgRating=%{customdata[1]:.2f}<br>Fame=%{x:.2f}<br>Talent=%{y:.2f}<extra></extra>')

st.plotly_chart(fig, use_container_width=True)

st.markdown('---')

# Top Actor Comparison
st.subheader('Top 10 Comparison — Fame vs Talent')
colA, colB = st.columns(2)
top_fame = df_filtered.sort_values('fameScore', ascending=False).head(10)
top_talent = df_filtered.sort_values('talentScore', ascending=False).head(10)
with colA:
    st.markdown('Top 10 by Fame Score')
    figA = px.bar(top_fame[::-1], x='fameScore', y='actor', orientation='h', color_discrete_sequence=[RED])
    st.plotly_chart(figA, use_container_width=True)
with colB:
    st.markdown('Top 10 by Talent Score')
    figB = px.bar(top_talent[::-1], x='talentScore', y='actor', orientation='h', color_discrete_sequence=[GOLD])
    st.plotly_chart(figB, use_container_width=True)

st.markdown('---')

# Correlation analysis
st.subheader('Correlation Analysis')
st.markdown(f"**Pearson correlation (r):** {corr:.2f} (p = {p:.2f})")
if not np.isnan(corr):
    interpretation = 'positive' if corr > 0 else 'negative' if corr < 0 else 'no'
    st.markdown(f"Interpretation: There is a {interpretation} correlation between fame and talent in the current selection.")

st.markdown('---')

# Actor detail view
st.subheader('Actor Detail View')
actor_detail = st.selectbox('Choose actor for details', options=[''] + df['actor'].tolist())
if actor_detail:
    row = df[df['actor'] == actor_detail].iloc[0]
    st.metric('Avg Rating', f"{row['avgRating']:.2f}")
    st.metric('Fame Score', f"{row['fameScore']:.2f}")
    st.metric('Talent Score', f"{row['talentScore']:.2f}")
    st.metric('Balance Score', f"{row['balanceScore']:.2f}")
    # mini profile chart
    small = pd.DataFrame({
        'metric':['fame','talent','balance'],
        'value':[row['fameScore'], row['talentScore'], row['balanceScore']]
    })
    fig_small = px.bar(small, x='value', y='metric', orientation='h', color='metric', color_discrete_map={'fame':RED,'talent':GOLD,'balance':ACCENT})
    st.plotly_chart(fig_small, use_container_width=True)

st.markdown('---')

# Insight summary and report generation
st.subheader('Insight Summary')
insights = []
mean_fame = df['fameScore'].mean()
mean_talent = df['talentScore'].mean()
if mean_fame > mean_talent:
    insights.append('On average, fame scores are slightly higher than talent scores in this dataset.')
elif mean_fame < mean_talent:
    insights.append('On average, talent scores are slightly higher than fame scores in this dataset.')
else:
    insights.append('On average, fame and talent scores are similar.')

top_fame_actor = df.sort_values('fameScore', ascending=False).iloc[0]['actor']
top_talent_actor = df.sort_values('talentScore', ascending=False).iloc[0]['actor']
insights.append(f'Top fame: {top_fame_actor}; Top talent: {top_talent_actor}.')

st.markdown('\n'.join(['- ' + s for s in insights]))

def generate_report(df):
    out = StringIO()
    out.write('# Technical Report — Balancing Fame and Talent in Bollywood\n\n')
    out.write('Audience & Context\n')
    out.write('This dashboard is aimed at entertainment journalists and studio executives. It helps evaluate alignment between public fame and movie quality.\n\n')
    out.write('KPIs Chosen\n')
    out.write('- fameScore: composite of normalized Google attention and movie rank.\n')
    out.write('- talentScore: normalized rating.\n')
    out.write('- balanceScore: 1 - |fame - talent| so higher = more balanced.\n\n')
    out.write('Dashboard Structure\n')
    out.write('- Header, KPI metrics, Scatter, Top comparisons, Correlation, Actor detail, Insight summary.\n\n')
    out.write('Data Cleaning Steps\n')
    out.write('- Removed duplicates and nulls.\n')
    out.write('- Removed actors with movieCount <= 0.\n')
    out.write('- Normalized googleHits using min-max scaling.\n\n')
    out.write('Design Choices\n')
    out.write('- Cinematic palette: golds and reds for visual impact.\n')
    out.write('- Interactive filters in sidebar.\n\n')
    out.write('Reflection\n')
    out.write('- What worked well: quick comparisons and drill-down.\n')
    out.write('- Challenges: mock data may not reflect real-world distributions.\n')
    out.write('- Next: incorporate time-series and social sentiment.\n\n')
    out.write('References\n')
    out.write('- Mock dataset: Bollywood FameStats (demo).\n')
    out.write('- Libraries: pandas, plotly, streamlit, scipy.\n')
    out.write('- Tools: ChatGPT used for scaffolding.\n')
    return out.getvalue()

report_text = generate_report(df)
st.download_button('Download Technical Report', data=report_text, file_name='report.md', mime='text/markdown')