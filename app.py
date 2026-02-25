import streamlit as st
import pandas as pd
import numpy as np
import hdbscan
from geopy.distance import geodesic
import pydeck as pdk

st.set_page_config(layout="wide")
st.title("🚌STPH Stop Optimizer")

# ----------------------------
# Upload CSV
# ----------------------------
uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    # ----------------------------
    # Filter boarding/alighting
    # ----------------------------
    df = df[(df['isBoarding'] == True) | (df['isAlighting'] == True)]
    df = df.dropna(subset=['lon','lat'])
    
    st.success("Data Loaded Successfully!")
    st.write(f"Total points: {len(df)}")  # Debug info
    
    st.subheader("📍 Boarding & Alighting Points")
    
    # Use the renamed columns
    st.pydeck_chart(pdk.Deck(
        initial_view_state=pdk.ViewState(
            latitude=df['lat'].mean(),
            longitude=df['lon'].mean(),
            zoom=12,
        ),
        layers=[
            pdk.Layer(
                "ScatterplotLayer",
                data=df,
                get_position=['lon', 'lat'],
                get_radius=70,
                get_fill_color=[0, 0, 255, 180],
                pickable=True,
            )
        ],
    ))

    # ----------------------------
    # Clustering Button
    # ----------------------------
    if st.button("Run Clustering"):
        st.info("Clustering in progress...")
        
        coordinates = df[['lon', 'lat']].values
        
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=10,
            min_samples=3
        )
        
        df['Cluster'] = clusterer.fit_predict(np.radians(coordinates))
        df = df[df['Cluster'] != -1]
        
        st.write(f"Points after clustering: {len(df)}")  # Debug info
        st.write(f"Number of clusters: {df['Cluster'].nunique()}")  # Debug info
        
        centroids = df.groupby('Cluster')[['lon', 'lat']].mean().reset_index()
        
        # ----------------------------
        # Minimum Distance Filter
        # ----------------------------
        def filter_min_distance(centroids_df, min_distance_m=200):
            kept = []
            for _, row in centroids_df.iterrows():
                point = (row['lat'], row['lon'])  # geodesic expects (lat, lon)
                if all(
                    geodesic(point, (k['lat'], k['lon'])).meters >= min_distance_m
                    for k in kept
                ):
                    kept.append(row)
            return pd.DataFrame(kept)
        
        filtered_centroids = filter_min_distance(centroids, 200)
        
        st.write(f"Filtered centroids: {len(filtered_centroids)}")  # Debug info
        
        st.success("Clustering Completed!")
        st.subheader("🔴 Clustered Stops")
        
        if len(filtered_centroids) > 0:
            # Removed map_style parameter to use default map
            st.pydeck_chart(pdk.Deck(
                initial_view_state=pdk.ViewState(
                    latitude=filtered_centroids['lat'].mean(),
                    longitude=filtered_centroids['lon'].mean(),
                    zoom=13,
                    pitch=0,
                ),
                layers=[
                    pdk.Layer(
                        "ScatterplotLayer",
                        data=filtered_centroids,
                        get_position=['lon', 'lat'],
                        get_color=[255, 0, 0, 200],
                        get_radius=70,
                    )
                ],
            ))
            
            # Download clustered centroids
            csv = filtered_centroids.to_csv(index=False).encode('utf-8')
            st.download_button(
                "Download Clustered Stops CSV",
                csv,
                "clustered_stops.csv",
                "text/csv"
            )
        else:
            st.warning("No centroids found after filtering!")

else:
    st.info("Please upload a CSV file to begin.")