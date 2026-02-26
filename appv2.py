import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from geopy.distance import geodesic
import pydeck as pdk
import re

st.set_page_config(layout="wide")

title_col, info_col = st.columns([11, 1])
with title_col:
    st.markdown("""
    <div style='display:flex;align-items:center;gap:12px;margin-bottom:8px;'>
        <span style='font-size:2.6rem;'>🚌</span>
        <span style='font-size:2.6rem;font-weight:800;line-height:1.15;letter-spacing:-0.5px;'>STPH Stop Optimizer</span>
    </div>
    """, unsafe_allow_html=True)
with info_col:
    st.markdown("""
    <style>
    div[data-testid="stPopover"] button {
        width: 45px !important;
        height: 45px !important;
        min-height: 38px !important;
        border-radius: 50% !important;
        padding: 0 !important;
        background: transparent !important;
        border: none !important;
        box-shadow: none !important;
        margin-top: 12px !important;
        cursor: pointer !important;
        transition: opacity 0.2s ease, transform 0.2s ease !important;
        color: transparent !important;
        font-size: 0 !important;
    }
    div[data-testid="stPopover"] button::after {
        content: "" !important;
        display: block !important;
        width: 100px !important;
        height: 100px !important;
        background-image: url("https://cdn-icons-png.freepik.com/256/69/69544.png?semt=ais_hybrid viewBox='0 0 100 100'%3E%3Ccircle cx='50' cy='50' r='44' fill='none' stroke='%23222' stroke-width='6'/%3E%3Ccircle cx='50' cy='28' r='6' fill='%23222'/%3E%3Cpath d='M44 42 Q38 44 40 48 L44 48 Q44 44 50 43 L50 75 Q44 74 42 78 L58 78 Q56 74 50 75 L50 43 Q56 44 56 48 L60 48 Q62 44 56 42 Z' fill='%23222'/%3E%3C/svg%3E") !important;
        background-size: contain !important;
        background-repeat: no-repeat !important;
        background-position: center !important;
    }
    div[data-testid="stPopover"] button:hover {
        opacity: 0.65 !important;
        transform: scale(1.1) !important;
    }
    </style>
    """, unsafe_allow_html=True)
    with st.popover("i"):
        st.markdown("""
### About This App
The **STPH Stop Optimizer** analyzes passenger boarding and alighting data to suggest optimal bus stop locations.

---

### 📂 Input Files

**Boarding & Alighting CSV** *(required)*
- Must contain coordinate columns: `lon`/`lng`/`longitude` and `lat`/`latitude`
- Must contain boarding/alighting columns: `isBoarding`/`board` and `isAlighting`/`alight` *(case-insensitive)*
- `isBoarding` / `isAlighting` should be `True`/`False`

**Reference Stops CSV** *(optional)*
- Existing bus stop locations to snap clusters to
- Accepted formats:
  - `WKT` column: `POINT (lon lat)`
  - `X` and `Y` columns (longitude, latitude)
  - Optional `name` column for stop labels

---

### 🔄 How It Works

1. **Upload** your boarding & alighting data
2. **Explore** the map — view passenger density heatmap and color-coded boarding/alighting points
3. **Run Clustering** — HDBSCAN groups nearby passenger activity into candidate stop locations
4. **Snap Threshold** — if reference stops are loaded, clusters within this distance (meters) snap to the nearest existing stop
5. **Download** the final optimized stop list as CSV

---

### 🗺️ Map Legend
| Color | Meaning |
|---|---|
| 🟢 Green | Boarding points / Snapped stops |
| 🔴 Red | Alighting points / New proposed stops |
| 🟠 Orange | Reference stops |
| Gradient | Passenger density heatmap |
        """)

# ----------------------------
# Inject a legend that floats OVER the pydeck map
# ----------------------------
LEGEND_STYLE = """
<style>
div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"]
  > div[data-testid="element-container"] > div.stMarkdown > div > div.map-legend-float {
    position: relative;
    z-index: 100;
    pointer-events: none;
}
.map-legend-float {
    margin-top: -460px;
    margin-right: 10px;
    float: right;
    display: inline-block;
    background: rgba(15, 20, 30, 0.75);
    backdrop-filter: blur(6px);
    -webkit-backdrop-filter: blur(6px);
    color: #f0f0f0;
    padding: 7px 10px;
    border-radius: 7px;
    font-size: 11px;
    font-family: sans-serif;
    line-height: 1.4;
    border: 1px solid rgba(255,255,255,0.1);
    min-width: 0;
    pointer-events: none;
}
.map-legend-float .legend-title {
    font-weight: 700;
    font-size: 9px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 6px;
}
.map-legend-float .legend-row {
    display: flex;
    align-items: center;
    gap: 6px;
    margin-bottom: 4px;
}
.map-legend-float .legend-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.map-legend-float .legend-label { font-size: 11px; }
.map-legend-float .legend-desc  { display: none; }
.map-legend-float .legend-divider {
    border-top: 1px solid rgba(255,255,255,0.1);
    margin: 5px 0;
}
.map-legend-float .gradient-bar {
    width: 60px; height: 7px; border-radius: 2px; flex-shrink: 0;
    background: linear-gradient(90deg,
        rgba(0,0,255,.7), rgba(0,200,255,.8), rgba(0,255,100,.85),
        rgba(255,255,0,.9), rgba(255,140,0,.95), rgba(255,0,0,1));
}
</style>
"""

def make_ba_legend(show_points, show_heatmap):
    rows = ""
    if show_points:
        rows += """
        <div class='legend-row'>
            <span class='legend-dot' style='background:#22c55e'></span>
            <div><div class='legend-label'>Boarding</div>
                 <div class='legend-desc'>Passengers getting on the bus</div></div>
        </div>
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ef4444'></span>
            <div><div class='legend-label'>Alighting</div>
                 <div class='legend-desc'>Passengers getting off the bus</div></div>
        </div>"""
    if show_heatmap:
        rows += """
        <div class='legend-divider'></div>
        <div><div class='legend-label' style='margin-bottom:4px;'>Density</div>
             <div style='display:flex;align-items:center;gap:4px;'>
                 <div class='gradient-bar'></div>
             </div>
        </div>"""
    return f"<div class='map-legend-float'><div class='legend-title'>Legend</div>{rows}</div>"


def make_opt_legend(has_reference):
    if has_reference:
        rows = """
        <div class='legend-row'>
            <span class='legend-dot' style='background:#00ff00'></span>
            <div><div class='legend-label'>Snapped to Reference</div>
                 <div class='legend-desc'>Matched to existing stop within snap threshold</div></div>
        </div>
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ff4444'></span>
            <div><div class='legend-label'>New Optimized Stop</div>
                 <div class='legend-desc'>Proposed stop — no nearby reference found</div></div>
        </div>
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ffa500'></span>
            <div><div class='legend-label'>Reference Stop</div>
                 <div class='legend-desc'>Existing stop from uploaded reference dataset</div></div>
        </div>"""
    else:
        rows = """
        <div class='legend-row'>
            <span class='legend-dot' style='background:#ff4444'></span>
            <div><div class='legend-label'>Optimized Stop</div>
                 <div class='legend-desc'>Proposed stop derived from passenger clusters</div></div>
        </div>"""
    return f"<div class='map-legend-float'><div class='legend-title'>Legend</div>{rows}</div>"


# ----------------------------
# Helper: Flexible column resolver
# ----------------------------
def find_column(df_columns, candidates):
    """
    Returns the first matching column name from `candidates` (case-insensitive).
    `candidates` is an ordered list of preferred names.
    Returns None if nothing matches.
    """
    col_map = {c.lower(): c for c in df_columns}
    for candidate in candidates:
        if candidate.lower() in col_map:
            return col_map[candidate.lower()]
    return None


def resolve_coords_and_flags(df):
    """
    Detects lon, lat, boarding, and alighting columns flexibly.
    Renames them to standard internal names: lon, lat, isBoarding, isAlighting.
    Returns (df_normalized, error_message_or_None).
    """
    cols = df.columns.tolist()

    lon_col = find_column(cols, ['lon', 'lng', 'longitude', 'long'])
    lat_col = find_column(cols, ['lat', 'latitude'])
    board_col = find_column(cols, ['isBoarding', 'boarding', 'board'])
    alight_col = find_column(cols, ['isAlighting', 'alighting', 'alight'])

    missing = []
    if lon_col is None:
        missing.append("longitude (tried: lon, lng, longitude, long)")
    if lat_col is None:
        missing.append("latitude (tried: lat, latitude)")
    if board_col is None:
        missing.append("boarding (tried: isBoarding, boarding, board)")
    if alight_col is None:
        missing.append("alighting (tried: isAlighting, alighting, alight)")

    if missing:
        return None, "Could not find required columns:\n- " + "\n- ".join(missing)

    rename_map = {
        lon_col:   'lon',
        lat_col:   'lat',
        board_col: 'isBoarding',
        alight_col:'isAlighting',
    }
    # Only rename if the detected name differs from the target
    rename_map = {k: v for k, v in rename_map.items() if k != v}
    df = df.rename(columns=rename_map)

    # Normalize boolean-like values (handles True/False strings, 1/0, yes/no)
    for flag_col in ['isBoarding', 'isAlighting']:
        col_data = df[flag_col]
        if col_data.dtype == object:
            df[flag_col] = col_data.str.strip().str.lower().isin(['true', '1', 'yes'])
        else:
            df[flag_col] = col_data.astype(bool)

    return df, None


# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload Boarding & Alighting CSV File", type=["csv"])

# ----------------------------
# Upload Reference Stops (Optional)
# ----------------------------
reference_file = st.file_uploader("Upload Reference Stops CSV (Optional)", type=["csv"], key="reference")
reference_stops = None

if reference_file is not None:
    reference_df = pd.read_csv(reference_file)
    if 'WKT' in reference_df.columns:
        def parse_wkt(wkt_string):
            match = re.search(r'POINT \(([0-9.-]+) ([0-9.-]+)\)', str(wkt_string))
            if match:
                return float(match.group(1)), float(match.group(2))
            return None, None
        reference_df[['ref_lon', 'ref_lat']] = reference_df['WKT'].apply(
            lambda x: pd.Series(parse_wkt(x))
        )
    elif 'X' in reference_df.columns and 'Y' in reference_df.columns:
        reference_df['ref_lon'] = reference_df['X']
        reference_df['ref_lat'] = reference_df['Y']
    else:
        st.error("Reference file must contain either 'WKT' or 'X'/'Y' columns")

    if reference_stops is None and 'ref_lon' in reference_df.columns:
        reference_df = reference_df.dropna(subset=['ref_lon', 'ref_lat'])
        if 'name' in reference_df.columns:
            reference_stops = reference_df[['ref_lon', 'ref_lat', 'name']].copy()
        else:
            reference_df['name'] = [f"Reference Stop {i+1}" for i in range(len(reference_df))]
            reference_stops = reference_df[['ref_lon', 'ref_lat', 'name']].copy()
        st.success(f"Reference stops loaded: {len(reference_stops)} stops")


# ----------------------------
# Main App
# ----------------------------
if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)

    df, col_error = resolve_coords_and_flags(raw_df)
    if col_error:
        st.error(col_error)
        st.stop()

    df = df[(df['isBoarding'] == True) | (df['isAlighting'] == True)]
    df = df.dropna(subset=['lon', 'lat'])

    st.success("Data Loaded Successfully!")

    boarding_df  = df[df['isBoarding']  == True].copy()
    alighting_df = df[df['isAlighting'] == True].copy()

    st.write(f"Total points: {len(df)}  |  🟢 Boarding: {len(boarding_df)}  |  🔴 Alighting: {len(alighting_df)}")

    MAP_STYLES = {
        "Carto Dark":      "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        "Carto Light":     "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
        "Road (built-in)": "road",
    }

    # ----------------------------
    # Section 1: Boarding & Alighting Map (collapsible)
    # ----------------------------
    with st.expander("📍 Boarding & Alighting Points", expanded=True):
        st.markdown("## 📍 Boarding & Alighting Points")
        col1, col2 = st.columns([3, 1])
        with col2:
            st.markdown("**Map Options**")
            map_style_choice   = st.selectbox("Map Style", list(MAP_STYLES.keys()), index=0)
            selected_map_style = MAP_STYLES[map_style_choice]
            show_points        = st.checkbox("Show Points",  value=True)
            show_heatmap       = st.checkbox("Show Heatmap", value=True)
            heatmap_intensity  = st.slider("Heatmap Intensity",   1,  20,  5)
            heatmap_radius     = st.slider("Heatmap Radius (px)", 10, 80, 30)

        layers = []
        if show_heatmap:
            layers.append(pdk.Layer(
                "HeatmapLayer", data=df[['lon', 'lat']].copy(),
                get_position=['lon', 'lat'], aggregation="SUM",
                intensity=heatmap_intensity, radius_pixels=heatmap_radius,
                threshold=0.05, pickable=False,
                color_range=[
                    [0,   0,   255, 120], [0,   200, 255, 160],
                    [0,   255, 100, 180], [255, 255, 0,   200],
                    [255, 140, 0,   220], [255, 0,   0,   255],
                ],
            ))
        if show_points:
            if len(boarding_df) > 0:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=boarding_df,
                    get_position=['lon', 'lat'], get_radius=40,
                    get_fill_color=[34, 197, 94, 210], pickable=True,
                ))
            if len(alighting_df) > 0:
                layers.append(pdk.Layer(
                    "ScatterplotLayer", data=alighting_df,
                    get_position=['lon', 'lat'], get_radius=40,
                    get_fill_color=[239, 68, 68, 210], pickable=True,
                ))

        with col1:
            st.markdown(LEGEND_STYLE, unsafe_allow_html=True)
            st.pydeck_chart(pdk.Deck(
                map_style=selected_map_style,
                initial_view_state=pdk.ViewState(
                    latitude=df['lat'].mean(), longitude=df['lon'].mean(),
                    zoom=12, pitch=0,
                ),
                layers=layers,
                tooltip={"text": "Lat: {lat}\nLon: {lon}"},
            ))
            st.markdown(make_ba_legend(show_points, show_heatmap), unsafe_allow_html=True)

        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.metric("Total Passenger Events", len(df))
        with stat_col2:
            st.metric("🟢 Boarding", len(boarding_df))
        with stat_col3:
            st.metric("🔴 Alighting", len(alighting_df))

    # ----------------------------
    # Run Clustering button — between the two sections
    # ----------------------------
    if st.button("🔄 Run Clustering"):
        st.info("Clustering in progress...")
        coordinates = df[['lon', 'lat']].values
        clusterer = hdbscan.HDBSCAN(min_cluster_size=10, min_samples=3)
        df['Cluster'] = clusterer.fit_predict(np.radians(coordinates))
        df = df[df['Cluster'] != -1]
        centroids = df.groupby('Cluster')[['lon', 'lat']].mean().reset_index()

        def filter_min_distance(centroids_df, min_distance_m=200):
            kept = []
            for _, row in centroids_df.iterrows():
                point = (row['lat'], row['lon'])
                if all(geodesic(point, (k['lat'], k['lon'])).meters >= min_distance_m for k in kept):
                    kept.append(row)
            return pd.DataFrame(kept)

        st.session_state['filtered_centroids'] = filter_min_distance(centroids, 200)
        st.session_state['n_clusters'] = df['Cluster'].nunique()
        st.session_state['n_points']   = len(df)

    # ----------------------------
    # Section 2: Optimized Stops (collapsible)
    # ----------------------------
    if 'filtered_centroids' in st.session_state:
        with st.expander("🔴 Optimized Stops", expanded=True):
            st.markdown("## 🔴 Optimized Stops")
            filtered_centroids = st.session_state['filtered_centroids']

            st.caption(f"Clustered points: {st.session_state['n_points']}  |  Clusters found: {st.session_state['n_clusters']}")

            # Snap threshold
            if reference_stops is not None:
                snap_threshold = st.slider("Snap Threshold (meters)", 50, 500, 150, 25)
                snapped_centroids, unsnapped_centroids = [], []
                for _, centroid in filtered_centroids.iterrows():
                    centroid_point = (centroid['lat'], centroid['lon'])
                    min_distance = float('inf')
                    nearest_ref = None
                    for _, ref in reference_stops.iterrows():
                        distance = geodesic(centroid_point, (ref['ref_lat'], ref['ref_lon'])).meters
                        if distance < min_distance:
                            min_distance = distance
                            nearest_ref = ref
                    if min_distance <= snap_threshold:
                        snapped_centroids.append({
                            'Cluster': centroid['Cluster'],
                            'lon': nearest_ref['ref_lon'], 'lat': nearest_ref['ref_lat'],
                            'snapped': True, 'snap_distance': min_distance,
                            'reference_name': nearest_ref['name']
                        })
                    else:
                        unsnapped_centroids.append({
                            'Cluster': centroid['Cluster'],
                            'lon': centroid['lon'], 'lat': centroid['lat'],
                            'snapped': False, 'snap_distance': None, 'reference_name': None
                        })
                final_centroids = pd.DataFrame(snapped_centroids + unsnapped_centroids)
                st.caption(f"✅ Snapped: {len(snapped_centroids)}  |  🆕 New stops: {len(unsnapped_centroids)}  |  Total: {len(final_centroids)}")
            else:
                final_centroids = filtered_centroids.copy()
                final_centroids['snapped'] = False
                st.caption(f"Total optimized stops: {len(final_centroids)}")

            if len(final_centroids) > 0:
                opt_layers = []
                if reference_stops is not None:
                    opt_layers.append(pdk.Layer(
                        "ScatterplotLayer", data=reference_stops,
                        get_position=['ref_lon', 'ref_lat'],
                        get_color=[255, 165, 0, 200], get_radius=50, pickable=True,
                    ))
                if 'snapped' in final_centroids.columns:
                    snapped   = final_centroids[final_centroids['snapped'] == True]
                    unsnapped = final_centroids[final_centroids['snapped'] == False]
                    if len(snapped) > 0:
                        opt_layers.append(pdk.Layer(
                            "ScatterplotLayer", data=snapped,
                            get_position=['lon', 'lat'], get_color=[0, 255, 0, 200],
                            get_radius=70, pickable=True,
                        ))
                    if len(unsnapped) > 0:
                        opt_layers.append(pdk.Layer(
                            "ScatterplotLayer", data=unsnapped,
                            get_position=['lon', 'lat'], get_color=[255, 0, 0, 200],
                            get_radius=70, pickable=True,
                        ))
                else:
                    opt_layers.append(pdk.Layer(
                        "ScatterplotLayer", data=final_centroids,
                        get_position=['lon', 'lat'], get_color=[255, 0, 0, 200], get_radius=70,
                    ))

                st.pydeck_chart(pdk.Deck(
                    map_style=selected_map_style,
                    initial_view_state=pdk.ViewState(
                        latitude=final_centroids['lat'].mean(),
                        longitude=final_centroids['lon'].mean(),
                        zoom=13, pitch=0,
                    ),
                    layers=opt_layers,
                ))
                st.markdown(make_opt_legend(reference_stops is not None), unsafe_allow_html=True)

                csv = final_centroids.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "⬇️ Download Optimized Stops CSV", csv,
                    "optimized_stops.csv", "text/csv"
                )
            else:
                st.warning("No centroids found after filtering!")

else:
    st.info("Please upload a CSV file to begin.")